import React, { useState, useCallback } from 'react'
import Header from './components/Header'
import ChatPanel from './components/ChatPanel'
import Sandbox from './components/Sandbox'
import LandingPage from './components/LandingPage'
import { PARAM_DEFAULTS } from './hyperparameters'
import { buildManifest } from './utils/buildManifest'
import './index.css'

const API_BASE = 'http://localhost:8000'

export default function App() {
  const [currentPage, setCurrentPage] = useState('landing')

  // Hyperparams lifted here so Header (Run Traversal) can read them
  const [hyperparams, setHyperparams] = useState(() =>
    Object.fromEntries(
      Object.entries(PARAM_DEFAULTS).map(([id, defaults]) => [id, { ...defaults }])
    )
  )

  const handleParamChange = useCallback((nodeId, key, value) => {
    setHyperparams((prev) => ({
      ...prev,
      [nodeId]: { ...prev[nodeId], [key]: value },
    }))
  }, [])

  const handleParamReset = useCallback((nodeId) => {
    setHyperparams((prev) => ({
      ...prev,
      [nodeId]: { ...PARAM_DEFAULTS[nodeId] },
    }))
  }, [])

  // Traversal state
  const [traversalLoading, setTraversalLoading] = useState(false)
  const [traversalResult, setTraversalResult] = useState(null)
  const [traversalError, setTraversalError]   = useState(null)

  const handleRunTraversal = useCallback(async () => {
    setTraversalLoading(true)
    setTraversalResult(null)
    setTraversalError(null)

    try {
      const manifest = buildManifest(hyperparams)
      const res = await fetch(`${API_BASE}/api/traverse`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(manifest),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Traversal failed' }))
        throw new Error(err.detail || 'Traversal failed')
      }

      setTraversalResult(await res.json())
    } catch (err) {
      setTraversalError(err.message)
    } finally {
      setTraversalLoading(false)
    }
  }, [hyperparams])

  if (currentPage === 'landing') {
    return <LandingPage onEnter={() => setCurrentPage('sandbox')} />
  }

  return (
    <div className="app-shell">
      <Header
        traversalLoading={traversalLoading}
        onRunTraversal={handleRunTraversal}
      />
      <div className="main-split">
        <ChatPanel />
        <Sandbox
          hyperparams={hyperparams}
          onParamChange={handleParamChange}
          onParamReset={handleParamReset}
          traversalResult={traversalResult}
          traversalError={traversalError}
          onCloseTraversal={() => { setTraversalResult(null); setTraversalError(null) }}
        />
      </div>
    </div>
  )
}
