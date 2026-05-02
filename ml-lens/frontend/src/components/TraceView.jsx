import React from 'react'
import AsteriskSpinner from './AsteriskSpinner'

// ── Helpers ──────────────────────────────────────────────────────────────────
function fmt(n) {
  if (!n && n !== 0) return '—'
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000)     return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000)         return `${(n / 1_000).toFixed(0)}K`
  return n.toString()
}

function kindColor(kind) {
  const map = {
    input_embedding:     { bg: '#EEF3FA', color: '#1E3A5F', dot: '#1E3A5F' },
    positional_encoding: { bg: '#EEF3FA', color: '#1E3A5F', dot: '#1E3A5F' },
    multi_head_attention:{ bg: '#EDF7ED', color: '#15803D', dot: '#15803D' },
    attention:           { bg: '#EDF7ED', color: '#15803D', dot: '#15803D' },
    feedforward:         { bg: '#F8FAFC', color: '#64748B', dot: '#64748B' },
    linear_projection:   { bg: '#F8FAFC', color: '#64748B', dot: '#64748B' },
    layernorm:           { bg: '#FEFCE8', color: '#CA8A04', dot: '#CA8A04' },
    rmsnorm:             { bg: '#FEFCE8', color: '#CA8A04', dot: '#CA8A04' },
    residual:            { bg: '#EEF2FF', color: '#4F46E5', dot: '#4F46E5' },
    softmax:             { bg: '#FFF7ED', color: '#EA580C', dot: '#EA580C' },
    output_head:         { bg: '#FFF7ED', color: '#EA580C', dot: '#EA580C' },
    masking:             { bg: '#FFF1F2', color: '#E11D48', dot: '#E11D48' },
  }
  return map[kind] ?? { bg: '#F5F5F5', color: '#4B5E78', dot: '#94A3B8' }
}

// ── Sub-components ────────────────────────────────────────────────────────────
function StatPill({ label, value }) {
  return (
    <div className="trace-stat-pill">
      <span className="trace-stat-value">{value}</span>
      <span className="trace-stat-label">{label}</span>
    </div>
  )
}

function StepRow({ step, index, total }) {
  const c = kindColor(step.component_kind)
  const isLast = index === total - 1
  return (
    <div className="trace-step-row">
      {/* Timeline spine */}
      <div className="trace-timeline-col">
        <div className="trace-step-dot" style={{ background: c.dot }} />
        {!isLast && <div className="trace-step-line" style={{ background: c.dot + '33' }} />}
      </div>

      {/* Card */}
      <div className="trace-step-card">
        <div className="trace-step-top">
          <span className="trace-step-index">{index + 1}</span>
          <span className="trace-step-name">{step.component_name}</span>
          <span className="trace-kind-badge" style={{ background: c.bg, color: c.color }}>
            {step.component_kind?.replace(/_/g, ' ') ?? 'other'}
          </span>
        </div>

        {/* Shape flow */}
        {(step.input_symbolic?.length > 0 || step.output_symbolic?.length > 0) && (
          <div className="trace-step-shapes">
            {step.input_symbolic?.length > 0 && (
              <span className="trace-shape-tag">
                <span className="trace-shape-dir">in</span>
                <code>[{step.input_symbolic.join(', ')}]</code>
              </span>
            )}
            {step.input_symbolic?.length > 0 && step.output_symbolic?.length > 0 && (
              <span className="trace-arrow">→</span>
            )}
            {step.output_symbolic?.length > 0 && (
              <span className="trace-shape-tag">
                <span className="trace-shape-dir">out</span>
                <code>[{step.output_symbolic.join(', ')}]</code>
              </span>
            )}
          </div>
        )}

        {step.key_insight && (
          <p className="trace-step-insight">{step.key_insight}</p>
        )}

        <div className="trace-step-chips">
          {step.parameter_count > 0 && (
            <span className="trace-chip">{fmt(step.parameter_count)} params</span>
          )}
          {step.flops_approx > 0 && (
            <span className="trace-chip">{fmt(step.flops_approx)} FLOPs</span>
          )}
        </div>
      </div>
    </div>
  )
}

// ── Main Component ────────────────────────────────────────────────────────────
export default function TraceView({ result, error, loading, onRun, manifest }) {
  const hasResult = result && result.steps?.length > 0

  return (
    <div className="trace-view">
      {/* Header bar */}
      <div className="trace-view-header">
        <div className="trace-view-title-row">
          <span className="trace-view-title">⚡ Traversal Trace</span>
          {hasResult && (
            <div className="trace-stats-row">
              <StatPill label="components" value={result.total_components ?? result.steps.length} />
              {result.total_parameters > 0 && (
                <StatPill label="parameters" value={fmt(result.total_parameters)} />
              )}
            </div>
          )}
        </div>

        <button
          className={`trace-run-btn ${loading ? 'loading' : ''}`}
          onClick={onRun}
          disabled={loading}
        >
          {loading
            ? <><AsteriskSpinner size={13} color="white" /> Running trace…</>
            : hasResult
              ? '↺ Re-run Trace'
              : '▶ Run Trace'
          }
        </button>
      </div>

      {/* Body */}
      <div className="trace-view-body">
        {/* Empty state */}
        {!loading && !hasResult && !error && (
          <div className="trace-empty-state">
            <div className="trace-empty-icon">⚡</div>
            <p className="trace-empty-title">No trace yet</p>
            <p className="trace-empty-sub">
              {manifest
                ? 'Click Run Trace to traverse this architecture and inspect tensor shapes, parameter counts, and data flow.'
                : 'Load a paper first, then run a trace to explore its architecture.'}
            </p>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="trace-loading-state">
            <AsteriskSpinner size={28} color="#1E3A5F" />
            <p>Traversing architecture…</p>
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="trace-error-state">
            <span className="trace-error-icon">⚠</span>
            <p>{error}</p>
          </div>
        )}

        {/* Steps timeline */}
        {hasResult && !loading && (
          <div className="trace-steps-list">
            {result.steps.map((step, i) => (
              <StepRow key={step.component_id ?? i} step={step} index={i} total={result.steps.length} />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
