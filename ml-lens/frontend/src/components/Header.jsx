import React from 'react'
import AsteriskSpinner from './AsteriskSpinner'

export default function Header({ viewMode, setViewMode, onRunTraversal, traversalLoading }) {
  return (
    <header className="header">
      <div className="header-container">
        <h1 className="logo">Yukti</h1>
        
        {/* Integrated View Toggle */}
        <div className="header-view-toggle">
          <button 
            className={`header-toggle-item ${viewMode === 'model' ? 'active' : ''}`}
            onClick={() => setViewMode('model')}
          >
            <span className="toggle-icon">㗊</span> <span className="toggle-label">Model</span>
          </button>
          <button 
            className={`header-toggle-item ${viewMode === 'code' ? 'active' : ''}`}
            onClick={() => setViewMode('code')}
          >
            <span className="toggle-icon">{"</>"}</span> <span className="toggle-label">Code</span>
          </button>
          <button 
            className={`header-toggle-item ${viewMode === 'trace' ? 'active' : ''}`}
            onClick={() => { setViewMode('trace'); onRunTraversal?.() }}
          >
            <span className="toggle-icon">⚡</span> <span className="toggle-label">
              {traversalLoading ? <><AsteriskSpinner size={11} color="currentColor" /> Running</> : 'Trace'}
            </span>
          </button>
        </div>

        <div className="header-actions">
          <button className="btn-primary btn-export-desktop">Export</button>
        </div>
      </div>
    </header>
  )
}
