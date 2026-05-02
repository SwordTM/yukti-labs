from __future__ import annotations

import json
import os
import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

from openai import OpenAI
from pydantic import ValidationError

from schema.models import ComponentManifest, PaperMetadata

from .arxiv_resolver import ArxivPaper
from .pdf_parser import ParsedPaper
from .prompts import (
    EXTRACTION_SYSTEM_PROMPT, 
    USER_MESSAGE_TEMPLATE, 
    TAXONOMY_SYSTEM_PROMPT, 
    TAXONOMY_USER_TEMPLATE,
    FOCUS_SYSTEM_PROMPT,
    FOCUS_USER_TEMPLATE
)

DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-01")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_TEXT_CHARS = 80_000

# Kind-based tiers for topological inference.
# Components in the same tier can be processed in parallel.
_TIER_ORDER = [
    {"input_embedding", "positional_encoding"},
    {"masking"},
    {"linear_projection", "attention", "multi_head_attention"},
    {"softmax"},
    {"residual"},
    {"layernorm", "rmsnorm"},
    {"feedforward"},
    {"output_head"},
]


def _infer_depends_on(raw_json: dict) -> dict:
    """Best-effort pass to fill in missing depends_on links and remove orphans."""
    comps = raw_json.get("components") or []
    if not comps:
        return raw_json

    id_list = [c["id"] for c in comps if isinstance(c, dict) and "id" in c]

    # Check if graph is already well-connected
    non_root = [c for c in comps if isinstance(c, dict) and c.get("depends_on")]
    if len(non_root) > len(comps) * 0.5:
        # Already somewhat connected, but let's still run orphan removal later
        pass
    else:
        # Low connectivity, try tensor matching
        tcs = raw_json.get("tensor_contracts") or []
        tc_out: dict[str, set] = {}
        tc_in: dict[str, set] = {}
        for tc in tcs:
            if not isinstance(tc, dict): continue
            cid = tc.get("component_id", "")
            if cid:
                tc_out[cid] = set(tc.get("output_shapes", {}).keys())
                tc_in[cid] = set(tc.get("input_shapes", {}).keys())

        for b_id in id_list:
            comp = next(c for c in comps if c.get("id") == b_id)
            if comp.get("depends_on"): continue # Skip if already has deps
            
            b_inputs = tc_in.get(b_id, set())
            if not b_inputs or b_inputs == {"x"}: continue
            
            for a_id in id_list:
                if a_id == b_id: continue
                a_outputs = tc_out.get(a_id, set())
                if a_outputs and b_inputs & a_outputs:
                    if "depends_on" not in comp: comp["depends_on"] = []
                    if a_id not in comp["depends_on"]:
                        comp["depends_on"].append(a_id)

    # Identify all IDs that are referenced by someone else
    all_refs = set()
    for comp in comps:
        if isinstance(comp, dict):
            all_refs.update(comp.get("depends_on") or [])

    final_components = []
    input_kinds = {"input_embedding", "positional_encoding"}

    for i, comp in enumerate(comps):
        if not isinstance(comp, dict):
            final_components.append(comp)
            continue

        cid = comp.get("id")
        kind = comp.get("kind")
        deps = comp.get("depends_on") or []
        is_referenced = cid in all_refs

        # Case A: True Orphan (no parents, no children, not an input)
        # These are usually sub-mechanisms that shouldn't be top-level nodes.
        if not deps and not is_referenced and kind not in input_kinds:
            logger.info(f"Removing orphaned sub-mechanism node: {cid}")
            continue

        # Case B: Broken Chain or Semantic Anchor needed
        # (no parents, but either has children or is a known auxiliary type)
        is_auxiliary = kind in {"positional_encoding", "masking"}
        if not deps and (is_referenced or is_auxiliary) and kind not in {"input_embedding"} and i > 0:
            # Look for a logical predecessor
            prev_comp = comps[i-1] if isinstance(comps[i-1], dict) else None
            prev_id = prev_comp.get("id") if prev_comp else None
            
            # Special logic for Decoder inputs:
            # If we see a second embedding/encoding block, it's likely the decoder starting.
            # It might depend on the Encoder Output (last of the first stack).
            if prev_id:
                comp["depends_on"] = [prev_id]
                logger.info(f"Semantically anchored {cid} to {prev_id}")

        final_components.append(comp)

    raw_json["components"] = final_components
    return raw_json


def _deduplicate_ids(raw_json: dict) -> dict:
    """Rename any duplicate component IDs and update all depends_on references."""
    comps = raw_json.get("components") or []
    seen: dict[str, int] = {}
    rename_map: dict[str, str] = {}

    for comp in comps:
        if not isinstance(comp, dict):
            continue
        original = comp.get("id", "")
        if not original:
            continue
        if original in seen:
            seen[original] += 1
            new_id = f"{original}_{seen[original]}"
            rename_map[original] = rename_map.get(original, original)  # keep first as-is
            comp["id"] = new_id
            logger.info(f"Deduplicated ID: '{original}' -> '{new_id}'")
        else:
            seen[original] = 1

    # Update depends_on references that pointed to a now-renamed original ID
    # (we only rename the SECOND+ occurrence, so depends_on refs to the first stay valid)
    # Nothing to do for the first occurrence — only subsequent duplicates were renamed.
    return raw_json


_VALID_KINDS = {
    "input_embedding", "positional_encoding", "linear_projection", "attention",
    "multi_head_attention", "feedforward", "layernorm", "rmsnorm", "residual",
    "softmax", "masking", "output_head", "other",
}

_KIND_ALIASES: dict[str, str] = {
    "embedding": "input_embedding",
    "token_embedding": "input_embedding",
    "word_embedding": "input_embedding",
    "positional_embedding": "positional_encoding",
    "position_encoding": "positional_encoding",
    "pos_encoding": "positional_encoding",
    "self_attention": "attention",
    "cross_attention": "attention",
    "scaled_dot_product_attention": "attention",
    "multi_head_self_attention": "multi_head_attention",
    "multihead_attention": "multi_head_attention",
    "mha": "multi_head_attention",
    "feed_forward": "feedforward",
    "ffn": "feedforward",
    "mlp": "feedforward",
    "position_wise_ffn": "feedforward",
    "layer_norm": "layernorm",
    "layer_normalization": "layernorm",
    "rms_norm": "rmsnorm",
    "rms_normalization": "rmsnorm",
    "skip_connection": "residual",
    "residual_connection": "residual",
    "add_norm": "residual",
    "softmax_attention": "softmax",
    "causal_mask": "masking",
    "attention_mask": "masking",
    "padding_mask": "masking",
    "linear": "linear_projection",
    "projection": "linear_projection",
    "linear_layer": "linear_projection",
    "output_projection": "linear_projection",
    "lm_head": "output_head",
    "language_model_head": "output_head",
    "classification_head": "output_head",
}


def _normalize_kind(kind: str) -> str:
    k = kind.lower().strip()
    if k in _KIND_ALIASES:
        return _KIND_ALIASES[k]
    return k # Allow novel kinds from LLM


def _normalize_invariant_kind(kind: str) -> str:
    valid = {"weight_tying", "causal_mask", "residual_connection", "init_scheme", "normalization_placement", "scaling", "other"}
    aliases = {
        "residual": "residual_connection",
        "weight_sharing": "weight_tying",
        "tied_weights": "weight_tying",
        "normalization": "normalization_placement",
        "norm_placement": "normalization_placement",
        "initialization": "init_scheme",
        "scale": "scaling",
        "mask": "causal_mask",
    }
    k = kind.lower().strip()
    if k in valid:
        return k
    if k in aliases:
        return aliases[k]
    return "other"


class ComponentExtractorError(RuntimeError):
    pass


def _truncate(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: int(limit * 0.75)]
    tail = text[-int(limit * 0.25) :]
    return f"{head}\n\n...[TRUNCATED {len(text) - limit} CHARS]...\n\n{tail}"


_VALID_JSON_ESCAPES = set('"' + "\\\\" + "/bfnrtu")


def _fix_latex_escapes(s: str) -> str:
    """Replace bare LaTeX backslashes (invalid JSON escapes) with double-backslash."""
    result = []
    i = 0
    while i < len(s):
        if s[i] == '\\' and i + 1 < len(s):
            if s[i + 1] in _VALID_JSON_ESCAPES:
                result.append(s[i])
                result.append(s[i + 1])
                i += 2
            else:
                result.append('\\\\')
                i += 1
        else:
            result.append(s[i])
            i += 1
    return ''.join(result)


def _repair_json(text: str) -> str:
    """Multi-stage repair for common LLM JSON issues."""
    # Stage 1: Fix bare LaTeX backslashes inside JSON strings
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    # Stage 2: Remove ASCII control characters (except tab, newline, carriage return)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Stage 3: Truncation recovery — close unclosed brackets/braces
    # If the model hit max_tokens mid-JSON, we try to salvage what we have
    opens = 0
    open_sq = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{': opens += 1
        elif ch == '}': opens -= 1
        elif ch == '[': open_sq += 1
        elif ch == ']': open_sq -= 1
    # If there are unclosed brackets, the output was truncated
    if opens > 0 or open_sq > 0:
        # Strip any trailing partial value (incomplete string, number, etc.)
        text = re.sub(r',\s*"[^"]*$', '', text)      # partial key
        text = re.sub(r',\s*\{[^}]*$', '', text)      # partial object
        text = re.sub(r',\s*$', '', text)               # trailing comma
        text += ']' * open_sq + '}' * opens
        logger.warning(f"Truncation detected — closed {open_sq} arrays and {opens} objects")
    return text


def _parse_response(text: str) -> dict:
    """Extract and parse JSON from the LLM response with multi-stage repair."""
    cleaned = text.strip()

    # Guard: model echoed back the schema definition instead of a manifest
    if '"$defs"' in cleaned[:200] or cleaned.lstrip().startswith('{"$defs"'):
        raise ComponentExtractorError(
            "Model returned the JSON Schema definition instead of a populated manifest. "
            "Re-run with Force Refresh to try again."
        )

    # Step 1: Strip <thinking>...</thinking> block entirely
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', cleaned, flags=re.DOTALL).strip()

    # Step 2: Extract JSON from fenced code blocks
    for pattern in [r"```json\s*(\{.*?\})\s*```", r"```\s*(\{.*?\})\s*```"]:
        m = re.search(pattern, cleaned, re.DOTALL)
        if m:
            target = m.group(1)
            break
    else:
        # Fallback: find outermost { }
        first = cleaned.find('{')
        last = cleaned.rfind('}')
        target = cleaned[first:last + 1] if first != -1 and last > first else cleaned

    # Step 3: Try parsing as-is
    try:
        return json.loads(target)
    except json.JSONDecodeError:
        pass

    # Step 4: Apply repair and retry
    repaired = _repair_json(target)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ComponentExtractorError(
            f"LLM did not return valid JSON: {exc}\nTarget (first 500 chars): {target[:500]}"
        ) from exc


import base64

def _encode_image(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def _describe_images(client: OpenAI, images: list[bytes]) -> str:
    """Stage 1: Use a dedicated VLM to describe architectural figures."""
    if not images:
        return ""
    
    logger.info(f"Describing {len(images)} figures using VLM...")
    prompt = "Describe these ML architecture diagrams in detail. Focus on the flow of tensors, the hierarchy of components (encoder/decoder/etc), and specific layer names or operations mentioned. Use bullet points."
    
    content = [{"type": "text", "text": prompt}]
    for img_bytes in images[:5]:
        b64 = _encode_image(img_bytes)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"}
        })
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-5-nano",
            messages=[{"role": "user", "content": content}],
            max_tokens=2000,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(f"VLM figure analysis failed: {exc}. Proceeding without visual context.")
        return ""


def extract_manifest(
    paper: ArxivPaper,
    parsed: ParsedPaper,
    client: Optional[OpenAI] = None,
    model: str = "minimax/minimax-m2.7",
) -> ComponentManifest:
    if client is None:
        api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
        if not api_key or "your_" in api_key:
            raise ComponentExtractorError("No valid API key found. Set OPENROUTER_API_KEY in .env")
        client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)

    # Stage 0: Identify Focus (Model Name and Goal) from Abstract
    focus_context = ""
    try:
        focus_resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": FOCUS_SYSTEM_PROMPT},
                {"role": "user", "content": FOCUS_USER_TEMPLATE.format(title=paper.title, abstract=paper.abstract)},
            ],
            max_tokens=500,
            temperature=0.1,
            response_format={"type": "json_object"},
        )
        focus_json = _parse_response(focus_resp.choices[0].message.content or "{}")
        proposed_name = focus_json.get("proposed_model_name", "the main model")
        focus_context = f"\n\n## PRIMARY FOCUS: {proposed_name}\nGoal: {focus_json.get('primary_goal')}\nStrategy: {focus_json.get('core_strategy')}"
        logger.info(f"Targeted Focus Identified: {proposed_name}")
    except Exception as e:
        logger.warning(f"Focus identification failed: {e}. Proceeding with general extraction.")

    # Stage 1: Get visual context from figures using GPT-5 Nano (VLM)
    visual_context = _describe_images(client, parsed.figure_images)

    # Stage 2: Final extraction using Minimax 2.7
    # Prepend focus hint BEFORE the body so the model reads it first (higher attention weight)
    focus_prefix = focus_context + "\n\n" if focus_context else ""

    user_message_text = focus_prefix + USER_MESSAGE_TEMPLATE.format(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=", ".join(paper.authors),
        high_context_text=_truncate(parsed.high_context_text, limit=30000),
        equations="\n".join(f"- {eq}" for eq in parsed.equations) or "(none extracted)",
        figure_captions="\n".join(f"- {c}" for c in parsed.figure_captions) or "(none)",
    )

    if visual_context:
        user_message_text += f"\n\n## Visual Analysis of Figures (from VLM)\n{visual_context}"

    response = client.chat.completions.create(
        model=model,  # Minimax 2.7
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message_text},
        ],
        max_tokens=16384,
        temperature=0.1,
        timeout=180,
        response_format={"type": "json_object"},
    )

    raw_text = (response.choices[0].message.content or "").strip()
    if not raw_text:
        raise ComponentExtractorError("LLM response was empty")

    # Pre-parse guard: catch schema echo BEFORE any JSON parsing attempt
    if '"$defs"' in raw_text[:300] or '"$ref"' in raw_text[:300]:
        raise ComponentExtractorError(
            "Model returned the JSON Schema definition instead of a populated manifest. "
            "Enable 'Force Refresh' and try again — the model confused the schema with the output."
        )

    raw_json = _parse_response(raw_text)

    # Fix any duplicate component IDs before graph validation
    raw_json = _deduplicate_ids(raw_json)

    # Post-parse guard
    if "$defs" in raw_json:
        raise ComponentExtractorError(
            "Model returned the JSON Schema definition instead of a populated manifest. "
            "Enable 'Force Refresh' and try again."
        )

    raw_json["paper"] = PaperMetadata(
        arxiv_id=paper.arxiv_id,
        title=paper.title,
        authors=paper.authors,
        abstract=paper.abstract,
        published=paper.published,
        pdf_url=paper.pdf_url,
    ).model_dump()

    if isinstance(raw_json.get("notes"), list):
        raw_json["notes"] = " ".join(str(n) for n in raw_json["notes"])

    for comp in (raw_json.get("components") or []):
        if not isinstance(comp, dict): continue
        if "kind" in comp:
            comp["kind"] = _normalize_kind(comp["kind"])
        if "quote" in comp and isinstance(comp["quote"], str):
            comp["quote"] = {"text": comp["quote"]}
        # Coerce all hyperparameter values to strings (schema requires dict[str, str])
        if "hyperparameters" in comp and isinstance(comp["hyperparameters"], dict):
            comp["hyperparameters"] = {
                k: str(v) for k, v in comp["hyperparameters"].items()
            }
    for inv in (raw_json.get("invariants") or []):
        if not isinstance(inv, dict): continue
        # generate id from name if missing
        if "id" not in inv and "name" in inv:
            inv["id"] = re.sub(r"[^a-z0-9]+", "_", inv["name"].lower()).strip("_")
        if "id" not in inv:
            inv["id"] = "invariant_" + str(raw_json.get("invariants", []).index(inv))
        if "kind" not in inv:
            inv["kind"] = "other"
        else:
            inv["kind"] = _normalize_invariant_kind(inv["kind"])
        if "affected_components" not in inv:
            inv["affected_components"] = []
        if "quote" in inv and isinstance(inv["quote"], str):
            inv["quote"] = {"text": inv["quote"]}
    valid_tcs = []
    for tc in (raw_json.get("tensor_contracts") or []):
        if not isinstance(tc, dict): continue
        if "quote" in tc and isinstance(tc["quote"], str):
            tc["quote"] = {"text": tc["quote"]}
        # coerce common field-name variants
        for wrong, right in [
            ("component", "component_id"), ("id", "component_id"),
            ("input_shape", "input_shapes"), ("output_shape", "output_shapes"),
            ("inputs", "input_shapes"), ("outputs", "output_shapes"),
        ]:
            if wrong in tc and right not in tc:
                tc[right] = tc.pop(wrong)
        # if shapes are still missing, provide empty dicts so Pydantic won't error
        if "input_shapes" not in tc:
            tc["input_shapes"] = {}
        if "output_shapes" not in tc:
            tc["output_shapes"] = {}
        # coerce list-of-strings shapes to dict if needed (LLM sometimes returns ["B","T","d"])
        for key in ("input_shapes", "output_shapes"):
            val = tc[key]
            if isinstance(val, list):
                tc[key] = {"x": [str(s) for s in val]}
            elif isinstance(val, dict):
                for k, v in val.items():
                    if isinstance(v, str):
                        val[k] = [v]
        valid_tcs.append(tc)
    raw_json["tensor_contracts"] = valid_tcs

    # ──────────────────────────────────────────────────────────────────────────
    # Chain 2: Taxonomist (Generate a meaningful legend for this paper)
    # ──────────────────────────────────────────────────────────────────────────
    try:
        # Prepare a minimal JSON of extracted components for the taxonomist
        comps_for_tax = [
            {"name": c.get("name"), "kind": c.get("kind"), "description": c.get("description")}
            for c in (raw_json.get("components") or [])
            if isinstance(c, dict)
        ]
        
        tax_response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": TAXONOMY_SYSTEM_PROMPT},
                {
                    "role": "user", 
                    "content": TAXONOMY_USER_TEMPLATE.format(
                        title=paper.title,
                        components_json=json.dumps(comps_for_tax, indent=2)
                    )
                },
            ],
            max_tokens=2048,
            temperature=0.1,
        )
        
        tax_text = (tax_response.choices[0].message.content or "").strip()
        tax_json = _parse_response(tax_text)
        
        if "taxonomy" in tax_json:
            raw_json["taxonomy"] = tax_json["taxonomy"]
            
            # Use the kind_mapping to update the components so they match the grouped taxonomy
            kind_mapping = tax_json.get("kind_mapping", {})
            for comp in (raw_json.get("components") or []):
                if not isinstance(comp, dict): continue
                original_kind = comp.get("kind")
                if original_kind in kind_mapping:
                    comp["kind"] = kind_mapping[original_kind]
                    
    except Exception as e:
        # Fallback: if taxonomy fails, just continue with empty taxonomy (frontend handles)
        print(f"Taxonomy chain failed: {e}")
        raw_json["taxonomy"] = []
    
    # Post-process: infer missing depends_on connections and remove orphans
    raw_json = _infer_depends_on(raw_json)

    try:
        return ComponentManifest.model_validate(raw_json)
    except ValidationError as exc:
        raise ComponentExtractorError(f"ComponentManifest validation failed: {exc}") from exc
