"""Format SchemaDiff output as Claude Code SKILL.md files."""

from schema.models import SchemaDiff


def schema_diff_to_skill_md(diff: SchemaDiff) -> str:
    """Convert a SchemaDiff to Claude Code SKILL.md format."""

    # Build component reference section
    component_details = []
    for comp in diff.component_diffs:
        details = f"### {comp.component_id.replace('_', ' ').title()}\n\n"

        if comp.changed:
            details += f"**Status**: Changed\n\n"
            details += f"**Shapes**: {comp.old_shapes.get('input', 'N/A')} → {comp.new_shapes.get('input', 'N/A')}\n\n"
            if comp.rationale:
                details += f"**Architectural Impact**:\n{comp.rationale}\n\n"

            if comp.param_deltas:
                details += "**Parameter Changes**:\n"
                for delta in comp.param_deltas:
                    details += f"- `{delta.param}`: {delta.old_value} → {delta.new_value}\n"
                details += "\n"

            if comp.invariants_broken:
                details += f"**Broken Invariants**: {', '.join(comp.invariants_broken)}\n\n"
            else:
                details += "**Broken Invariants**: None\n\n"
        else:
            details += "**Status**: Unchanged\n\n"

        component_details.append(details)

    # Build the SKILL.md file
    skill_md = f"""---
name: model-architecture-update
description: Implementation guide for updating neural network architecture. {diff.paper_id} model hyperparameter modifications from {diff.base_params.get('num_heads', 'N/A')} to {diff.modified_params.get('num_heads', 'N/A')} attention heads. Follow the concrete code changes below to maintain architectural invariants and ensure tensor compatibility.
---

# Architecture Update Guide

## Summary

This guide provides step-by-step instructions for updating the model architecture based on hyperparameter changes.

**Parameter Changes**:
- Base: {dict_to_readable(diff.base_params)}
- Modified: {dict_to_readable(diff.modified_params)}

## Implementation Steps

{diff.implementation_notes}

## Component-by-Component Analysis

{chr(10).join(component_details)}

## Invariant Preservation

When making these changes, ensure:
- All tensor shapes remain compatible with downstream layers
- Residual connections are not broken
- Layer normalization inputs/outputs maintain expected dimensions
- Weight matrices have compatible dimensions for all linear transformations

## Validation Checklist

Before running the modified model:

- [ ] All view() and reshape() calls updated for new tensor dimensions
- [ ] Linear projections updated if affected by head count changes
- [ ] Attention computation correctly handles new d_k value
- [ ] Embedding and output layers unchanged
- [ ] Model compiles without shape mismatch errors
- [ ] Forward pass runs without errors on test batch
- [ ] Verify gradients flow through all layers during backward pass
"""

    return skill_md


def dict_to_readable(params: dict) -> str:
    """Convert parameter dict to readable format."""
    items = [f"{k}={v}" for k, v in params.items()]
    return ", ".join(items)
