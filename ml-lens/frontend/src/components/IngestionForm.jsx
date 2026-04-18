import React, { useState } from 'react'

export default function IngestionForm({ onSubmit, loading }) {
  const [url, setUrl] = useState('')
  const [forceRefresh, setForceRefresh] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!url.trim()) {
      setError('Please enter an arXiv URL or ID')
      return
    }
    setError(null)
    onSubmit(url, forceRefresh)
  }

  return (
    <form onSubmit={handleSubmit} className="ingestion-form">
      <div className="form-group">
        <label htmlFor="arxiv-url">arXiv URL or ID</label>
        <input
          id="arxiv-url"
          type="text"
          placeholder="e.g., 1706.03762 or https://arxiv.org/abs/1706.03762"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={loading}
        />
        {error && <span className="form-error">{error}</span>}
      </div>

      <div className="form-group checkbox-group">
        <label>
          <input
            type="checkbox"
            checked={forceRefresh}
            onChange={(e) => setForceRefresh(e.target.checked)}
            disabled={loading}
          />
          <span>Force refresh (ignore cache)</span>
        </label>
      </div>

      <button 
        type="submit" 
        className="btn-primary"
        disabled={loading}
      >
        {loading ? 'Processing...' : 'Ingest Paper'}
      </button>
    </form>
  )
}
