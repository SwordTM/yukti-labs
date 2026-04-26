# Yukti

> **Schema-grounded paper understanding for ML engineers.**
> Ingest any arXiv paper, lock a verified architecture contract, and traverse it step-by-step — with tensor shapes, equations, and parameter counts that are faithful to what the paper actually specifies.

---

## The Problem

Every week, hundreds of new ML papers drop proposing novel attention mechanisms, normalisation schemes, and training objectives. For an ML engineer, the workflow is always the same:

1. Read the paper *(~2 hours)*
2. Understand the math *(~2 hours)*
3. Ask Claude or GPT to implement it *(confident, fast, and frequently wrong)*

**The problem is step 3.** Large language models hallucinate implementations by blending architectures from their training data. Ask for Differential Attention and you get vanilla multi-head attention with a lambda variable bolted on. The Q/K split is wrong. The head-wise RMSNorm is missing. The tensor shapes are off by a factor of `h`. You only find out after a cryptic CUDA error two days in.

This is not a prompting problem. It is a **grounding problem.**

Without a verified contract anchoring the LLM to what the paper says, the model fills gaps from memory — and for any paper published in the last six months, that memory is noise.

> *70%+ of ML researchers fail to reproduce published results. The leading cause is not bad code — it is undocumented implementation decisions that deviate silently from the paper.*

---

## The Insight

Before any implementation happens, extract a **locked schema contract** directly from the paper. Every component, every tensor shape, every invariant — tied to the exact paper quote it was extracted from. Then constrain everything downstream to that contract.

The schema is not a prompt. It is a typed, hash-locked, machine-readable specification that the traversal agent cannot deviate from. Violations are caught structurally, not at runtime.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  1. INGESTION                                                    │
│                                                                  │
│  arXiv URL ──► PyMuPDF + LaTeX source ──► Agentic Extraction     │
│                                           (Thinking Phase)       │
│                                                   │              │
│                                                   ▼              │
│                               ComponentManifest (raw JSON)       │
└───────────────────────────────────────────────────┬─────────────┘
                                                    │
┌───────────────────────────────────────────────────▼─────────────┐
│  2. SCHEMA CONTRACT (the key layer)                              │
│                                                                  │
│  ┌──────────────────┐  ┌─────────────────┐  ┌────────────────┐  │
│  │ Component        │  │ Tensor          │  │ Invariants     │  │
│  │ Manifest         │  │ Contracts       │  │                │  │
│  │                  │  │                 │  │ weight tying   │  │
│  │ id, name, kind   │  │ I/O shapes per  │  │ causal masking │  │
│  │ equations        │  │ component with  │  │ residuals      │  │
│  │ depends_on       │  │ symbolic dims   │  │ norm placement │  │
│  │ hyperparameters  │  │ (B, T, d, h…)   │  │                │  │
│  └──────────────────┘  └─────────────────┘  └────────────────┘  │
│                                                                  │
│  ── content-hash locked ──────────────────────────────────────── │
└───────────────────────────────────────────────────┬─────────────┘
                                                    │
┌───────────────────────────────────────────────────▼─────────────┐
│  3. TRAVERSAL AGENT                                              │
│                                                                  │
│  Topological graph walk · deterministic math engine              │
│  Per-step: symbolic shapes → concrete → equations → insight      │
│  Full trace saved for replay                                     │
└───────────────────────────────────────────────────┬─────────────┘
                                                    │
┌───────────────────────────────────────────────────▼─────────────┐
│  4. VISUALIZATION + EXPORT                                       │
│                                                                  │
│  React Flow DAG · KaTeX equations · Agentic Sandbox Chat         │
│  Skill export: locked manifest + trace → portable context bundle │
└─────────────────────────────────────────────────────────────────┘
```

---

## Technical Architecture

### Ingestion Pipeline

The ingestion pipeline runs in three cached stages:

| Stage | Input | Output | Cache key |
|---|---|---|---|
| Metadata | arXiv URL / ID | Paper title, authors, PDF URL | arxiv ID |
| Parsing | PDF bytes | Structured text + LaTeX equations | arxiv ID |
| Extraction | Paper text + equations | Locked `ComponentManifest` | SHA-256 of prompt + text |

**Parsing** uses PyMuPDF for PDF text extraction and fetches the arXiv LaTeX source tarball directly — preserving original equation notation. 

**Extraction** uses an Agentic reasoning flow. The system prompt requires a **Mandatory Thinking Phase** (`<thinking>`) where the LLM traces the mathematical data flow before outputting the `ComponentManifest`. This handles complex structural depths that single-shot prompts miss. A **Topological Tiering** algorithm in the backend then cleans up the dependency graph, correctly identifying parallel sources and residual paths.

### Schema Contract

The `ComponentManifest` is a Pydantic v2 model with a content-hash lock:

```python
class ComponentManifest(BaseModel):
    paper: PaperMetadata
    components: list[Component]          # typed by ComponentKind enum
    tensor_contracts: list[TensorContract]  # input/output shapes per component
    invariants: list[Invariant]          # paper-specific structural rules
    symbol_table: dict[str, str]         # every dimension variable defined
    notes: Optional[str]
    locked: bool
```

`ComponentKind` is a strict enum: `input_embedding`, `positional_encoding`, `multi_head_attention`, `attention`, `feedforward`, `layernorm`, `rmsnorm`, `residual`, `softmax`, `masking`, `linear_projection`, `output_head`, `other`. 

### Traversal Agent

The traversal agent walks the component graph in topological order. For each component:

1. **Math engine** computes deterministically:
   - Parameter count (weights + biases per component kind)
   - FLOPs approximation
   - Intermediate tensor names, symbolic shapes, and LaTeX equations

2. **LLM insight call** produces a one-sentence key insight per component.

### Visualization & Agentic Sandbox

The frontend is a React + Vite app using `@xyflow/react` for the DAG and KaTeX for equation rendering.

- **Layout algorithm:** Longest-path level assignment on the `depends_on` DAG.
- **Agentic Chat**: An integrated Architectural Co-pilot that understands the current manifest. You can ask for modifications (e.g., "duplicate this layer") and the agent will propose a **wiring plan** with inferred `depends_on` connections, which you can apply directly to the Sandbox.

---

## Project Structure

```
yukti/
├── backend/
│   ├── ingestion/
│   │   ├── arxiv_resolver.py      # arXiv ID → metadata + PDF URL
│   │   ├── pdf_parser.py          # PyMuPDF + LaTeX tarball extraction
│   │   ├── component_extractor.py # Agentic extraction + Tiering logic
│   │   ├── prompts.py             # Schema-injected prompts with Thinking phase
│   │   └── pipeline.py            # 3-stage cached orchestrator
│   ├── agent/
│   │   ├── traversal_agent.py     # topological walk, per-component trace
│   │   └── math_engine.py         # deterministic params/FLOPs/shapes
│   ├── schema/
│   │   ├── models.py              # ComponentManifest + Pydantic models
│   │   └── lock.py                # SHA-256 content-hash locking
│   └── main.py                    # FastAPI app, Chat API, Static serving
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── Sandbox.jsx            # React Flow DAG + Model/Code toggle
│       │   ├── ChatPanel.jsx          # Agentic Sandbox Co-pilot
│       │   ├── Header.jsx             # Unified navigation + View toggle
│       │   └── ArchitectureFlow.jsx   # Ingest-driven architecture graph
│       └── utils/
│           └── manifestToFlow.js      # Layout + Experimental highlighting
└── Dockerfile                     # Multi-stage production build
```

---

## Quickstart

### Requirements

- Python 3.12+
- Node.js 20+
- An [OpenRouter](https://openrouter.ai) API key

### Running with Docker (Recommended)

The easiest way to run Yukti is using the unified Docker image:

```bash
docker build -t yukti .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your_key_here yukti
```
Visit `http://localhost:8000` to start exploring.

### Development Setup

**Backend:**
```bash
cd ml-lens/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend:**
```bash
cd ml-lens/frontend
npm install
npm run dev
```

---

## Team

**Saksham Grover + Chamalka Muwangala**

Built at the Florent × Lund AI Society Hackathon — *Build Your Next Startup* — April 18, 2026.

Sponsored by Anthropic · Voyado · Specific (YC F25) · Atech · Librar Labs (YC W26)
