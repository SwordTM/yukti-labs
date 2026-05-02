from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field

# Well-known kinds used for fallback styling, but any string is allowed.
ComponentKind = str

InvariantKind = Literal[
    "weight_tying",
    "causal_mask",
    "residual_connection",
    "init_scheme",
    "normalization_placement",
    "scaling",
    "other",
]


class LegendCategory(BaseModel):
    kind: str = Field(..., description="The unique kind identifier matching component.kind")
    label: str = Field(..., description="Human-readable label for the legend, e.g. 'Core Logic'")
    description: Optional[str] = None
    color_hint: Optional[str] = Field(None, description="Optional color hint (e.g. 'blue', '#FF0000')")


class PaperQuote(BaseModel):
    text: str = Field(..., description="Verbatim excerpt from the paper")
    section: Optional[str] = Field(None, description="Section heading where this appears")


class TensorContract(BaseModel):
    component_id: str = Field(..., description="ID of the component this contract belongs to")
    input_shapes: dict[str, list[str]] = Field(
        ..., description="Symbolic input shapes, e.g. {'x': ['B', 'T', 'd_model']}"
    )
    output_shapes: dict[str, list[str]] = Field(
        ..., description="Symbolic output shapes"
    )
    dtype: Optional[str] = Field(None, description="Expected dtype, e.g. 'float32'")
    quote: Optional[PaperQuote] = None


class Invariant(BaseModel):
    id: str = Field(..., description="Unique snake_case identifier")
    description: str
    kind: InvariantKind
    affected_components: list[str] = Field(..., description="Component IDs this invariant applies to")
    quote: Optional[PaperQuote] = None


class Component(BaseModel):
    id: str = Field(..., description="Unique snake_case identifier, e.g. 'multi_head_attention'")
    name: str = Field(..., description="Human-readable name")
    kind: ComponentKind
    description: str
    operations: list[str] = Field(default_factory=list, description="Ordered list of ops this component performs")
    depends_on: list[str] = Field(default_factory=list, description="IDs of components this depends on")
    hyperparameters: dict[str, str] = Field(default_factory=dict, description="Symbolic hyperparameter names and meanings")
    equations: list[str] = Field(default_factory=list, description="LaTeX equations from the paper")
    repetition_count: Optional[int] = Field(None, description="Number of times this component/block is repeated (e.g. Nx in Transformer)")
    quote: Optional[PaperQuote] = None


class PaperMetadata(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    pdf_url: str


class ComponentManifest(BaseModel):
    paper: PaperMetadata
    components: list[Component]
    tensor_contracts: list[TensorContract] = Field(default_factory=list)
    invariants: list[Invariant] = Field(default_factory=list)
    symbol_table: dict[str, str] = Field(
        default_factory=dict,
        description="Map of symbol -> meaning, e.g. {'d_model': 'model hidden dimension'}"
    )
    notes: Optional[str] = None
    taxonomy: list[LegendCategory] = Field(
        default_factory=list,
        description="Dynamic legend/categories for this specific paper"
    )
    locked: bool = Field(
        default=False, description="Whether this manifest is locked for diff generation"
    )


class StateSnapshot(BaseModel):
    """Captured tensor state at a single layer during forward pass."""

    component_id: str
    input_shape: str
    output_shape: str
    input_sample: list[float]
    output_sample: list[float]
    operation_note: str


class HyperparamDelta(BaseModel):
    """A single hyperparameter change."""

    component_id: str
    param: str
    old_value: int | float
    new_value: int | float


class ComponentDiff(BaseModel):
    """Per-component diff including tensor shape changes and architectural rationale."""

    component_id: str
    changed: bool
    param_deltas: list[HyperparamDelta] = Field(default_factory=list)
    old_shapes: dict
    new_shapes: dict
    rationale: str = ""
    invariants_held: list[str] = Field(default_factory=list)
    invariants_broken: list[str] = Field(default_factory=list)


class SchemaDiff(BaseModel):
    """Top-level diff specification for a modified architecture."""

    paper_id: str
    base_params: dict
    modified_params: dict
    component_diffs: list[ComponentDiff]
    implementation_notes: str
