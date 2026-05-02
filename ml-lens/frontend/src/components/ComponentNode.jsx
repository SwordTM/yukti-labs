import React from 'react'
import { Handle, Position } from '@xyflow/react'

/**
 * Custom node card:
 * - Kind label at top (small, uppercase, colored)
 * - Component name below (larger, bold)
 * - Left border accent per semantic category
 *
 * Handles any component kind from any paper — known kinds get semantic
 * colors, unknown kinds get a stable hue from a curated palette so the
 * graph is always visually informative, never a sea of white "other" cards.
 */
export default function ComponentNode({ data, selected }) {
  const comp = data.component
  const taxonomy = data.manifest?.taxonomy || []
  const category = taxonomy.find(t => t.kind === comp?.kind)

  const kind = comp?.kind ?? 'other'
  const kindLabel = (category?.label ?? KIND_DISPLAY[kind] ?? kind.replace(/_/g, ' ')).toUpperCase()
  const accent = resolveAccent(kind, category)

  return (
    <div
      className={`comp-node ${selected ? 'comp-node--selected' : ''}`}
      style={{
        borderColor: accent.border,
        background: accent.bg,
        borderLeftColor: accent.accent,
        borderLeftWidth: 3,
      }}
    >
      <Handle type="target" position={Position.Top} className="comp-node-handle" />

      <span className="comp-node-kind" style={{ color: accent.accent }}>
        {kindLabel}
      </span>
      <span className="comp-node-name">
        {comp?.name ?? data.label}
      </span>

      {comp?.repetition_count > 1 && (
        <span className="comp-node-repetition" title={`${comp.repetition_count} stacked layers`}>
          {comp.repetition_count}x
        </span>
      )}

      {comp?.is_experimental && (
        <span className="comp-node-exp-badge">EXP</span>
      )}

      <Handle type="source" position={Position.Bottom} className="comp-node-handle" />
    </div>
  )
}

// ── Semantic palette for well-known kinds ───────────────────────────────────
const SEMANTIC_ACCENT = {
  input_embedding:     { bg: '#EEF3FA', border: '#A8C0DC', accent: '#1E3A5F' },
  positional_encoding: { bg: '#EEF3FA', border: '#A8C0DC', accent: '#1E3A5F' },
  attention:           { bg: '#EDF7ED', border: '#86EFAC', accent: '#16A34A' },
  multi_head_attention:{ bg: '#EDF7ED', border: '#86EFAC', accent: '#16A34A' },
  layernorm:           { bg: '#FEFCE8', border: '#FDE047', accent: '#CA8A04' },
  rmsnorm:             { bg: '#FEFCE8', border: '#FDE047', accent: '#CA8A04' },
  linear_projection:   { bg: '#F8FAFC', border: '#CBD5E1', accent: '#64748B' },
  feedforward:         { bg: '#F8FAFC', border: '#CBD5E1', accent: '#64748B' },
  residual:            { bg: '#EEF2FF', border: '#A5B4FC', accent: '#4F46E5' },
  softmax:             { bg: '#FFF7ED', border: '#FDBA74', accent: '#EA580C' },
  output_head:         { bg: '#FFF7ED', border: '#FDBA74', accent: '#EA580C' },
  masking:             { bg: '#FFF1F2', border: '#FDA4AF', accent: '#E11D48' },
  other:               { bg: '#FFFFFF', border: '#D6E4F0', accent: '#94A3B8' },
}

const COLOR_HINT_MAP = {
  emerald: { bg: '#F0FDF4', border: '#86EFAC', accent: '#15803D' },
  fuchsia: { bg: '#FDF4FF', border: '#E879F9', accent: '#A21CAF' },
  amber:   { bg: '#FFFBEB', border: '#FCD34D', accent: '#B45309' },
  sky:     { bg: '#F0F9FF', border: '#7DD3FC', accent: '#0369A1' },
  red:     { bg: '#FFF5F5', border: '#FC8181', accent: '#C53030' },
  violet:  { bg: '#F5F3FF', border: '#C4B5FD', accent: '#6D28D9' },
  teal:    { bg: '#ECFDF5', border: '#6EE7B7', accent: '#047857' },
  yellow:  { bg: '#FEF3C7', border: '#F59E0B', accent: '#92400E' },
  blue:    { bg: '#EFF6FF', border: '#93C5FD', accent: '#1D4ED8' },
  pink:    { bg: '#FCE7F3', border: '#F9A8D4', accent: '#9D174D' },
  cyan:    { bg: '#ECFEFF', border: '#67E8F9', accent: '#0E7490' },
  lime:    { bg: '#F7FEE7', border: '#A3E635', accent: '#3F6212' },
  rose:    { bg: '#FFF1F2', border: '#FDA4AF', accent: '#E11D48' },
}

// ── Curated palette for novel/unknown kinds — 12 distinct hues ─────────────
const DYNAMIC_PALETTE = Object.values(COLOR_HINT_MAP)

// Simple deterministic hash so the same kind always maps to the same slot.
function hashKind(kind) {
  let h = 0
  for (let i = 0; i < kind.length; i++) {
    h = (h * 31 + kind.charCodeAt(i)) >>> 0
  }
  return h % DYNAMIC_PALETTE.length
}

// Cache to keep colors stable across re-renders
const _cache = {}
export function resolveAccent(kind, category) {
  // 1. Check if LLM provided a color hint in the taxonomy
  if (category?.color_hint && COLOR_HINT_MAP[category.color_hint]) {
    return COLOR_HINT_MAP[category.color_hint]
  }

  // 2. Check if it's a known semantic kind
  if (SEMANTIC_ACCENT[kind]) return SEMANTIC_ACCENT[kind]

  // 3. Deterministic fallback
  if (_cache[kind]) return _cache[kind]
  _cache[kind] = DYNAMIC_PALETTE[hashKind(kind)]
  return _cache[kind]
}

export { COLOR_HINT_MAP }

// ── Human-readable labels for known kinds ───────────────────────────────────
export const KIND_DISPLAY = {
  input_embedding:     'Input Embedding',
  positional_encoding: 'Positional Encoding',
  attention:           'Attention',
  multi_head_attention:'Multi Head Attention',
  layernorm:           'Layer Norm',
  rmsnorm:             'RMS Norm',
  linear_projection:   'Linear Projection',
  feedforward:         'Feed Forward',
  residual:            'Residual',
  softmax:             'Softmax',
  output_head:         'Output Head',
  masking:             'Masking',
  other:               'Other',
}

// ── Legend groups for the bottom-right panel ────────────────────────────────
// Only shows the well-known semantic categories; novel kinds appear as "Other"
export const LEGEND_GROUPS = [
  { label: 'Input / Encoding',  accent: SEMANTIC_ACCENT.input_embedding },
  { label: 'Attention',         accent: SEMANTIC_ACCENT.multi_head_attention },
  { label: 'Norm',              accent: SEMANTIC_ACCENT.layernorm },
  { label: 'Projection / FFN',  accent: SEMANTIC_ACCENT.feedforward },
  { label: 'Residual',          accent: SEMANTIC_ACCENT.residual },
  { label: 'Output / Softmax',  accent: SEMANTIC_ACCENT.output_head },
  { label: 'Masking',           accent: SEMANTIC_ACCENT.masking },
  { label: 'Other / Novel',     accent: SEMANTIC_ACCENT.other },
]
