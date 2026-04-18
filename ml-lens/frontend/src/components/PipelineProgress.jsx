import React, { useState, useEffect } from 'react'
import AsteriskSpinner from './AsteriskSpinner'

const STAGES = [
  {
    id: 'resolve',
    label: 'Resolving paper',
    desc: 'Looking up arXiv metadata, authors, and PDF location',
    estimatedMs: 2000,
  },
  {
    id: 'fetch',
    label: 'Fetching PDF',
    desc: 'Downloading paper from arXiv servers',
    estimatedMs: 4000,
  },
  {
    id: 'parse',
    label: 'Parsing content',
    desc: 'Extracting text and structure with PyMuPDF',
    estimatedMs: 2500,
  },
  {
    id: 'extract',
    label: 'Extracting components',
    desc: 'Identifying model components, equations, and hyperparameters via LLM',
    estimatedMs: 18000,
  },
  {
    id: 'lock',
    label: 'Locking schema',
    desc: 'Validating manifest and computing content hash',
    estimatedMs: 800,
  },
]

// Status: 'pending' | 'active' | 'done'
function buildInitialState() {
  return STAGES.map((s, i) => ({ ...s, status: i === 0 ? 'active' : 'pending' }))
}

export default function PipelineProgress({ done }) {
  const [stages, setStages] = useState(buildInitialState)

  useEffect(() => {
    if (done) {
      setStages((prev) => prev.map((s) => ({ ...s, status: 'done' })))
      return
    }

    // Advance stages based on cumulative estimated durations
    let cumulative = 0
    const timers = STAGES.map((stage, idx) => {
      cumulative += stage.estimatedMs
      return setTimeout(() => {
        setStages((prev) =>
          prev.map((s, i) => {
            if (i < idx + 1) return { ...s, status: 'done' }
            if (i === idx + 1) return { ...s, status: 'active' }
            return s
          })
        )
      }, cumulative)
    })

    return () => timers.forEach(clearTimeout)
  }, [done])

  return (
    <div className="pipeline-progress">
      {stages.map((stage) => (
        <div key={stage.id} className={`pipeline-stage pipeline-stage--${stage.status}`}>
          <div className="pipeline-stage-icon">
            {stage.status === 'done' ? (
              <span className="pipeline-check">✓</span>
            ) : stage.status === 'active' ? (
              <AsteriskSpinner size={16} color="#1E3A5F" />
            ) : (
              <span className="pipeline-dot" />
            )}
          </div>
          <div className="pipeline-stage-body">
            <span className="pipeline-stage-label">{stage.label}</span>
            <span className="pipeline-stage-desc">{stage.desc}</span>
          </div>
        </div>
      ))}
    </div>
  )
}
