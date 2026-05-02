from __future__ import annotations
import json

# ─── Stage 0: Focus Identification ───────────────────────────────────────────
FOCUS_SYSTEM_PROMPT = """You are a research paper classifier. 
Your goal is to identify the PRIMARY proposed architecture or model introduced by the authors.
Ignore baselines, comparison models, and general background techniques.
Return a JSON object with:
- "proposed_model_name": The name of the model they built (e.g. "LC-SLab", "Transformer", "BERT").
- "primary_goal": 1 sentence on what it does.
- "core_strategy": 1-2 sentences on the architectural approach (e.g. "A GNN-based aggregator following a pre-trained CNN").
"""

FOCUS_USER_TEMPLATE = """
Title: {title}
Abstract: {abstract}
"""

# ─── Minimal example manifest (what the model should produce) ─────────────────
# We DON'T embed the full Pydantic JSON schema — it confuses models into echoing it back.
# Instead we provide a compact worked example as few-shot guidance.

_EXAMPLE = json.dumps({
  "components": [
    {
      "id": "input_embedding",
      "name": "Input Embedding",
      "kind": "input_embedding",
      "description": "Converts token IDs to dense vectors of dimension d_model.",
      "operations": ["lookup", "scale"],
      "depends_on": [],
      "hyperparameters": {"d_model": "512"},
      "equations": ["x = W_e \\cdot token"],
      "notes": None
    },
    {
      "id": "positional_encoding",
      "name": "Positional Encoding",
      "kind": "positional_encoding",
      "description": "Adds sinusoidal position information to embeddings.",
      "operations": ["sin_cos_encode", "add"],
      "depends_on": ["input_embedding"],
      "hyperparameters": {},
      "equations": ["PE_{pos,2i} = \\sin(pos/10000^{2i/d_{model}})"],
      "notes": None
    },
    {
      "id": "multi_head_attention",
      "name": "Multi-Head Attention",
      "kind": "multi_head_attention",
      "description": "Computes attention over h parallel heads.",
      "operations": ["linear_project_qkv", "scaled_dot_product", "concat", "output_proj"],
      "depends_on": ["positional_encoding"],
      "hyperparameters": {"h": "8", "d_k": "64"},
      "equations": ["Attention(Q,K,V)=softmax(QK^T/\\sqrt{d_k})V"],
      "notes": None
    }
  ],
  "tensor_contracts": [
    {
      "component_id": "multi_head_attention",
      "input_shapes":  {"Q": ["B","T","d_model"], "K": ["B","T","d_model"], "V": ["B","T","d_model"]},
      "output_shapes": {"out": ["B","T","d_model"]},
      "dtype": "float32"
    }
  ],
  "invariants": [
    {
      "id": "residual_add_norm",
      "description": "Each sublayer output is LayerNorm(x + Sublayer(x)).",
      "kind": "residual_connection",
      "affected_components": ["multi_head_attention"]
    }
  ],
  "symbol_table": {
    "B": "batch size",
    "T": "sequence length",
    "d_model": "model hidden dimension"
  },
  "notes": None
}, indent=2)

# ─── Descriptive component kinds (Examples only) ────────────────────────────────
_VALID_KINDS = ", ".join([
  "input_embedding", "positional_encoding", "linear_projection", "attention",
  "convolutional_layer", "recurrent_cell", "pooling", "normalization",
  "activation", "loss", "optimizer", "other"
])

# ─── Valid invariant kinds ─────────────────────────────────────────────────────
_VALID_INV_KINDS = ", ".join([
  "weight_tying", "causal_mask", "residual_connection",
  "init_scheme", "normalization_placement", "scaling", "other"
])

# ─── System prompt ─────────────────────────────────────────────────────────────
EXTRACTION_SYSTEM_PROMPT = f"""You are an expert ML architect. Extract the architecture from the paper and return a JSON manifest.

## Output schema (field names are strict)

```json
{{
  "components": [
    {{
      "id":              "<snake_case string>",
      "name":            "<human-readable string>",
      "kind":            "<a descriptive snake_case string classifying this component (e.g. input_embedding, convolutional_layer, lstm_cell, memory_bank)>",
      "description":     "<string>",
      "operations":      ["<string>", ...],
      "depends_on":      ["<id of upstream component>", ...],
      "hyperparameters": {{"<key>": "<value>"}},
      "equations":       ["<latex string>", ...],
      "notes":           null
    }}
  ],
  "tensor_contracts": [
    {{
      "component_id":  "<id>",
      "input_shapes":  {{"<name>": ["<dim>", ...]}},
      "output_shapes": {{"<name>": ["<dim>", ...]}},
      "dtype":         "float32"
    }}
  ],
  "invariants": [
    {{
      "id":                   "<string>",
      "description":          "<string>",
      "kind":                 "<one of: {_VALID_INV_KINDS}>",
      "affected_components":  ["<id>", ...]
    }}
  ],
  "symbol_table": {{"<symbol>": "<meaning>"}},
  "notes": null
}}
```

## Example of correct output

```json
{_EXAMPLE}
```

## Critical rules for depends_on

- Every component except raw inputs MUST have at least one entry in `depends_on`.
- `depends_on` encodes real data flow: if B receives tensors from A, then `B.depends_on = ["A"]`.
- Encoder-Decoder Cross-Attention MUST list BOTH the decoder previous sublayer (Queries) AND the encoder final output (Keys + Values).
- Do NOT leave components isolated.

## Critical rules for unique IDs

- Every component MUST have a globally unique `id`. NO two components may share an id.
- Use the pattern: `{{block_prefix}}_{{component_type}}_{{instance_number}}`
  - `block_prefix`: derived from the paper's own block structure (e.g. `encoder`, `decoder`, `stage1`, `backbone`, `head`, `branch_a`). Read the paper and use whatever grouping IT defines.
  - `component_type`: the functional role in snake_case (e.g. `self_attention`, `ffn`, `layernorm`, `conv`, `residual`).
  - `instance_number`: append `_1`, `_2`, etc. ONLY when a component type appears more than once within the same block.
  - If a component is unique globally (e.g. a single output head), the prefix and number are optional.
- This convention is architecture-agnostic. YOU derive the prefixes from the paper — do NOT assume Encoder/Decoder structure.

## Critical rules for connectivity

- EVERY component (including auxiliary ones like Positional Encoding and Masking) MUST be connected to the data flow.
- Positional Encoding MUST depend on the preceding Input Embedding.
- Attention mechanisms MUST depend on any Masks or Positional Encodings applied to their inputs.
- Do NOT leave any component "floating" or disconnected unless it is a primary raw input.

## Critical rules for granularity

- ONLY extract primary, high-level architectural blocks (e.g. "Attention", "FFN", "LayerNorm").
- DO NOT extract internal mathematical operations (e.g. "Softmax", "Scaled Dot-Product", "Matrix Multiplication") as separate components. These should be listed in the `operations` array of their parent block instead.
- DO NOT extract auxiliary inputs like "Causal Masking" as separate components unless they are complex, multi-stage modules. If a mask is just a matrix applied during attention, describe it in the Attention component's `operations` or `description`.
- Every node in the graph MUST be a meaningful participant in the data flow. If a node has no dependencies and nothing depends on it, it should NOT exist.

## Critical rules for architectural consolidation

- If the paper describes a repeating stack of layers (e.g., "The encoder is composed of a stack of N = 6 identical layers"), DO NOT extract 6 separate components.
- Extract ONE representative component (e.g., `id: "encoder_block"`) and set its `repetition_count` to the number of layers (e.g., `6`).
- This consolidation applies even if the internal sub-components repeat. Focus on the high-level "Block" as the unit of repetition.
- Only list individual layers if they have UNIQUE connectivity or parameters (which is rare in standard Transformers).

## Critical rules for primary model focus

- ONLY extract the architecture PROPOSED by the authors of this paper.
- DO NOT extract baselines, ablation studies, or comparison models (e.g., if the paper compares their model to ResNet-50 or U-Net, DO NOT extract ResNet-50 or U-Net).
- Identify the proposed model by looking for keywords like "In this work, we introduce...", "Our proposed architecture...", "Figure 1 shows our model...", etc.
- If the paper contains a survey of multiple models, prioritize the one that matches the title and main contribution of the paper.
- The goal is to visualize the authors' unique contribution, not a summary of the entire field.

## Think before you write — sublayer inventory

Before generating the JSON, mentally count how many times each component type appears. For each type that appears more than once, plan unique IDs with instance numbers. This prevents duplicate IDs and ensures every `depends_on` reference resolves to exactly one component.

## ⚠️ FINAL REMINDER — read this before generating your response

1. You are producing a POPULATED manifest of THIS paper's architecture, NOT returning the schema.
2. Your output must start with `{{` — a raw JSON object. No markdown, no prose, no code fences.
3. Every `kind` value MUST be one of: {_VALID_KINDS}
4. Every non-input component MUST have a non-empty `depends_on` list.
5. Cross-attention (if present) MUST depend on TWO parents: query source AND key/value source.
6. All backslashes in LaTeX equations must be doubled: `\\frac`, `\\sqrt`, `\\sum`.
7. Do NOT echo the schema. Do NOT explain anything. Output ONLY the JSON object.
8. Every `id` MUST be globally unique — mentally count repeated types and assign `_1`, `_2` suffixes before writing.
9. If the user message contains a `## PRIMARY FOCUS:` block, you MUST restrict extraction to ONLY that model — ignore all other architectures mentioned.
"""

USER_MESSAGE_TEMPLATE = (
"""Paper: {title} ({arxiv_id})
Authors: {authors}

## Architecture text (methodology sections)
{high_context_text}

## Figure captions
{figure_captions}

## Equations
{equations}

---
REMINDER — before you write your response:
- Output a single raw JSON object only. No markdown fences, no prose, no explanations.
- Every component needs `depends_on` filled in (except raw embedding inputs with no parents).
- Cross-attention MUST depend on BOTH the encoder final output AND the decoder previous sublayer.
- `kind` should be a descriptive snake_case string classifying the component.
- Start your response with {{ immediately.
---

Return the ComponentManifest JSON now.

>>> OUTPUT THE POPULATED ComponentManifest JSON OBJECT NOW. START WITH {{ <<<"""
)

TAXONOMY_SYSTEM_PROMPT = """\
You are an expert ML research engineer and taxonomist. 
Your task is to take a list of extracted ML components from a research paper and group them into a logical, high-level taxonomy (a Legend).

## Input
You will receive a list of components, each with a name, a preliminary "kind", and a description.

## Your Goal
1. Consolidate specific "kinds" into broader, more meaningful categories if there are too many. **CRITICAL: You must aggressively group components to produce NO MORE THAN 4 to 6 categories.** The legend area has limited vertical space. Do not just list every component as its own category.
2. For each unique "kind" used in the final set, provide a human-readable label and a brief description for the user-facing legend.
3. Suggest a color hint for each category (e.g., 'blue', 'green', 'amber', 'rose', 'violet', 'emerald').
4. Provide a mapping from the original component "kind" strings to the new grouped taxonomy "kind".

## Output Format
Produce a JSON object with two keys "taxonomy" and "kind_mapping":
{
  "taxonomy": [
    {
      "kind": "snake_case_id",
      "label": "Human Readable Label",
      "description": "Short explanation of this category",
      "color_hint": "blue"
    }
  ],
  "kind_mapping": {
    "original_kind_1": "grouped_kind",
    "original_kind_2": "grouped_kind"
  }
}
"""

TAXONOMY_USER_TEMPLATE = """Paper Title: {title}
Components extracted:
{components_json}

Now produce the taxonomy (legend) and kind_mapping for these components."""
