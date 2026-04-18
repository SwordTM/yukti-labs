import React from 'react'
import AsteriskSpinner from './AsteriskSpinner'

export default function Header({ onRunTraversal, traversalLoading }) {
  return (
    <header className="header">
      <div className="header-container">
        <h1 className="logo">Yukti</h1>
        <div className="header-actions">
          <button
            className="btn-ghost header-traversal-btn"
            onClick={onRunTraversal}
            disabled={traversalLoading}
          >
            {traversalLoading
              ? <><AsteriskSpinner size={13} color="#4B5E78" />Running…</>
              : 'Run Traversal'
            }
          </button>
          <button className="btn-primary">Export</button>
        </div>
      </div>
    </header>
  )
}
