import React, { useState, useCallback } from 'react'
import Header from './components/Header'
import ChatPanel from './components/ChatPanel'
import Sandbox from './components/Sandbox'
import LandingPage from './components/LandingPage'
import { PARAM_DEFAULTS } from './hyperparameters'
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
      // Fetch the real "Attention Is All You Need" manifest from the backend
      const sampleRes = await fetch(`${API_BASE}/api/schema/sample`)
      if (!sampleRes.ok) throw new Error('Could not load paper manifest')
      const { manifest } = await sampleRes.json()

      // Overlay user's live hyperparams into the symbol table
      const p1 = hyperparams['1'], p2 = hyperparams['2']
      const p5 = hyperparams['5'], p6 = hyperparams['6']
      const dK = Math.floor(p2.d_model / p5.num_heads)
      manifest.symbol_table = {
        ...manifest.symbol_table,
        B:       'batch size',
        T:       `sequence length (max ${p1.max_seq_len})`,
        d_model: `model hidden dimension (${p2.d_model})`,
        h:       `number of attention heads (${p5.num_heads})`,
        d_k:     `key/query dimension per head (${dK})`,
        d_ff:    `feed-forward hidden dimension (${p6.d_ff})`,
        V:       `vocabulary size (${p2.vocab_size ?? 37000})`,
      }

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
