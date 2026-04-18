import React from 'react'

const KIND_BADGE = {
  input_embedding:     { label: 'Embedding',   bg: '#EEF3FA', color: '#1E3A5F' },
  positional_encoding: { label: 'Positional',  bg: '#EEF3FA', color: '#1E3A5F' },
  multi_head_attention:{ label: 'Attention',   bg: '#EEF3FA', color: '#1E3A5F' },
  attention:           { label: 'Attention',   bg: '#EEF3FA', color: '#1E3A5F' },
  feedforward:         { label: 'FFN',         bg: '#F0FDF4', color: '#166534' },
  layernorm:           { label: 'LayerNorm',   bg: '#F5F5F5', color: '#4B5E78' },
  masking:             { label: 'Masking',     bg: '#FFF7ED', color: '#C2410C' },
  output_head:         { label: 'Output',      bg: '#EEF3FA', color: '#1E3A5F' },
  other:               { label: 'Other',       bg: '#F5F5F5', color: '#4B5E78' },
}

function fmt(n) {
  if (!n && n !== 0) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000)     return `${(n / 1_000).toFixed(0)}K`
  return n.toString()
}

function ShapeTag({ label, shapes }) {
  if (!shapes || shapes.length === 0) return null
  return (
    <span className="traversal-shape-tag">
      <span className="traversal-shape-label">{label}</span>
      <code className="traversal-shape-val">[{shapes.join(', ')}]</code>
    </span>
  )
}

function StepCard({ step, index }) {
  const badge = KIND_BADGE[step.component_kind] ?? KIND_BADGE.other
  return (
    <div className="traversal-step-card">
      <div className="traversal-step-header">
        <span className="traversal-step-num">{index + 1}</span>
        <span className="traversal-step-name">{step.component_name}</span>
        <span className="traversal-kind-badge" style={{ background: badge.bg, color: badge.color }}>
          {badge.label}
        </span>
      </div>

      <div className="traversal-shapes">
        <ShapeTag label="in" shapes={step.input_symbolic} />
        <span className="traversal-arrow">→</span>
        <ShapeTag label="out" shapes={step.output_symbolic} />
      </div>

      <p className="traversal-insight">{step.key_insight}</p>

      <div className="traversal-step-meta">
        {step.parameter_count > 0 && (
          <span className="traversal-meta-chip">{fmt(step.parameter_count)} params</span>
        )}
        {step.flops_approx > 0 && (
          <span className="traversal-meta-chip">{fmt(step.flops_approx)} FLOPs</span>
        )}
      </div>
    </div>
  )
}

export default function TraversalPanel({ result, error, onClose }) {
  return (
    <div className="traversal-panel">
      <div className="traversal-panel-header">
        <div className="traversal-panel-title-row">
          <span className="traversal-panel-title">Traversal Trace</span>
          {result && (
            <span className="traversal-panel-meta">
              {result.total_components} components · {fmt(result.total_parameters)} params
            </span>
          )}
        </div>
        <button className="node-popup-close" onClick={onClose} aria-label="Close">✕</button>
      </div>

      <div className="traversal-panel-body">
        {error && (
          <div className="traversal-error">
            <p>Traversal failed: {error}</p>
          </div>
        )}

        {result && result.steps.map((step, i) => (
          <StepCard key={step.component_id} step={step} index={i} />
        ))}
      </div>
    </div>
  )
}
