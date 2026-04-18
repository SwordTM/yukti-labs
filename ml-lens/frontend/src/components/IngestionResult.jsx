import React from 'react'

export default function IngestionResult({ result, onReset }) {
  if (!result) return null

  const { manifest, cached } = result

  return (
    <div className="ingestion-result">
      <div className="result-header">
        <h3>Ingestion Complete</h3>
        {cached && (
          <span className="badge badge-cached">Cached</span>
        )}
      </div>

      <div className="result-content">
        {manifest.paper_title && (
          <div className="result-item">
            <label>Title</label>
            <p>{manifest.paper_title}</p>
          </div>
        )}

        {manifest.arxiv_id && (
          <div className="result-item">
            <label>arXiv ID</label>
            <p className="monospace">{manifest.arxiv_id}</p>
          </div>
        )}

        {manifest.components && manifest.components.length > 0 && (
          <div className="result-item">
            <label>Components Found</label>
            <div className="components-list">
              {manifest.components.map((comp, idx) => (
                <div key={idx} className="component-item">
                  <p className="component-name">{comp.name || 'Unnamed'}</p>
                  {comp.description && (
                    <p className="component-desc">{comp.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      <button onClick={onReset} className="btn-primary">
        Ingest Another Paper
      </button>
    </div>
  )
}
