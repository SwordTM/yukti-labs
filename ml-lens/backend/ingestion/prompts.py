from __future__ import annotations

import json

from schema.models import ComponentManifest

import copy

_raw_schema = ComponentManifest.model_json_schema()
_SCHEMA_FOR_LLM = copy.deepcopy(_raw_schema)
# Remove the `paper` field from the schema we show the LLM — it is injected server-side
_SCHEMA_FOR_LLM.get("properties", {}).pop("paper", None)
_SCHEMA_FOR_LLM.get("required", [None])  # keep as-is, just don't add
if "paper" in _SCHEMA_FOR_LLM.get("required", []):
    _SCHEMA_FOR_LLM["required"] = [r for r in _SCHEMA_FOR_LLM["required"] if r != "paper"]
_SCHEMA_JSON = json.dumps(_SCHEMA_FOR_LLM, indent=2)

EXTRACTION_SYSTEM_PROMPT = f"""You are an expert ML research engineer extracting a locked architectural contract from a paper. This contract grounds downstream code-generation agents, so precision and structural depth matter.

## Your job

1. **THINK**: First, trace the mathematical data flow of the model in your mind. Identify parallel branches (like Q, K, V projections), residual connections, and normalization placement.
2. **STRUCTURE**: Produce a JSON object that strictly conforms to the JSON Schema below.

## JSON Schema

```json
{_SCHEMA_JSON}
```

## Granularity Rules

- Break down complex layers into their constituent mathematical blocks.
- Example: Don't just extract "Encoder Layer". Extract "Multi-Head Attention", "Add & Norm", "Feed-Forward", and "Add & Norm (Final)".
- Capture sub-components like "Scaled Dot-Product Attention" if they have specific equations or tensor shapes mentioned.

## depends_on — DATA FLOW GRAPH (CRITICAL)

`depends_on` encodes the **data-flow graph**.
- Parallel sources: Components that receive the same input (e.g. Q, K, V projections) should all have the same `depends_on` ids.
- Residual connections: A component like "Add & Norm" should have TWO entries in `depends_on`: the output of the previous layer and the original input that is being added back.
- Every non-root component MUST have at least one entry in `depends_on`.

Example for one Transformer Encoder Block:
```json
[
  {{"id": "input_tokens",         "depends_on": []}},
  {{"id": "input_emb",            "depends_on": ["input_tokens"]}},
  {{"id": "pos_enc",              "depends_on": ["input_tokens"]}},
  {{"id": "emb_sum",              "depends_on": ["input_emb", "pos_enc"]}},
  {{"id": "mha",                  "depends_on": ["emb_sum"]}},
  {{"id": "residual_1",           "depends_on": ["emb_sum", "mha"]}},
  {{"id": "layer_norm_1",         "depends_on": ["residual_1"]}},
  {{"id": "ffn",                  "depends_on": ["layer_norm_1"]}},
  {{"id": "residual_2",           "depends_on": ["layer_norm_1", "ffn"]}},
  {{"id": "layer_norm_2",         "depends_on": ["residual_2"]}}
]
```

## LaTeX and JSON

- Use double backslashes for ALL LaTeX commands (e.g. `\\\\text{{Softmax}}`).
- Ensure equations are valid LaTeX.

## Output Format

You MUST start your response with a `<thinking>` block where you briefly (max 5-10 sentences) outline the model's structure. Then, provide the JSON object inside a ```json``` code block.
"""

USER_MESSAGE_TEMPLATE = """Paper metadata:
- arxiv_id: {arxiv_id}
- title: {title}
- authors: {authors}

Extracted LaTeX equations (deduped, up to 200):
{equations}

Figure captions extracted from PDF:
{figure_captions}

Paper text (PyMuPDF extraction):
---
{text}
---

Produce the ComponentManifest JSON now."""
