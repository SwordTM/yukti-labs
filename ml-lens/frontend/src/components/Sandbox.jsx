import React, { useCallback, useState, useMemo, useEffect } from 'react'
import {
  ReactFlow,
  Controls,
  Background,
  Panel,
  useNodesState,
  useEdgesState,
  addEdge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import NodeInfoPopup from './NodeInfoPopup'
import TraceView from './TraceView'
import ComponentNode, { resolveAccent, LEGEND_GROUPS } from './ComponentNode'
import { manifestToFlow, FLOW_EDGE_TYPES } from '../utils/manifestToFlow.jsx'

const NODE_TYPES = { component: ComponentNode }
const EDGE_TYPES = FLOW_EDGE_TYPES

// ── Fallback hardcoded demo (Attention Is All You Need) ──────────────────────
const nodeBase = { borderRadius: 8, padding: 10, fontFamily: 'Poppins, Arial, sans-serif', fontSize: 13 }
const navyNode   = { ...nodeBase, background: '#EEF3FA', border: '1px solid #1E3A5F',  fontWeight: 600 }
const navyStrong = { ...nodeBase, background: '#EEF3FA', border: '2px solid #1E3A5F',  fontWeight: 700 }
const plainNode  = { ...nodeBase, background: 'white',   border: '1px solid #D6E4F0' }
const decNode    = { ...nodeBase, background: '#FFF7ED', border: '2px solid #F97316',  fontWeight: 700 }

const DEMO_NODES = [
  { id: '1',  type: 'input',  position: { x: 250, y: 0 },   data: { label: '📄 Input Tokens' },        style: navyNode },
  { id: '2',                  position: { x: 100, y: 120 },  data: { label: '🔢 Input Embedding' },      style: plainNode },
  { id: '3',                  position: { x: 400, y: 120 },  data: { label: '📍 Positional Encoding' },  style: plainNode },
  { id: '4',                  position: { x: 250, y: 250 },  data: { label: '🔁 Encoder (×6)' },         style: navyStrong },
  { id: '5',                  position: { x: 100, y: 380 },  data: { label: '🎯 Multi-Head Attention' }, style: plainNode },
  { id: '6',                  position: { x: 400, y: 380 },  data: { label: '⚡ Feed Forward' },          style: plainNode },
  { id: '7',                  position: { x: 250, y: 510 },  data: { label: '🔁 Decoder (×6)' },         style: decNode },
  { id: '8',                  position: { x: 100, y: 640 },  data: { label: '🎯 Masked Attention' },     style: plainNode },
  { id: '9',                  position: { x: 400, y: 640 },  data: { label: '🔗 Cross Attention' },      style: plainNode },
  { id: '10', type: 'output', position: { x: 250, y: 770 },  data: { label: '📊 Linear + Softmax' },     style: navyNode },
]

const DEMO_EDGES = [
  { id: 'e1-2',  source: '1', target: '2',  animated: true, style: { stroke: '#1E3A5F' } },
  { id: 'e1-3',  source: '1', target: '3',  animated: true, style: { stroke: '#1E3A5F' } },
  { id: 'e2-4',  source: '2', target: '4',  style: { stroke: '#7A93B0' } },
  { id: 'e3-4',  source: '3', target: '4',  style: { stroke: '#7A93B0' } },
  { id: 'e4-5',  source: '4', target: '5',  style: { stroke: '#7A93B0' } },
  { id: 'e4-6',  source: '4', target: '6',  style: { stroke: '#7A93B0' } },
  { id: 'e5-7',  source: '5', target: '7',  style: { stroke: '#7A93B0' } },
  { id: 'e6-7',  source: '6', target: '7',  style: { stroke: '#7A93B0' } },
  { id: 'e4-9',  source: '4', target: '9',  animated: true, style: { stroke: '#F97316', strokeDasharray: '5,5' } },
  { id: 'e7-8',  source: '7', target: '8',  style: { stroke: '#7A93B0' } },
  { id: 'e7-9',  source: '7', target: '9',  style: { stroke: '#7A93B0' } },
  { id: 'e8-10', source: '8', target: '10', style: { stroke: '#7A93B0' } },
  { id: 'e9-10', source: '9', target: '10', style: { stroke: '#7A93B0' } },
]
// ─────────────────────────────────────────────────────────────────────────────

export default function Sandbox({
  manifest,
  viewMode,
  hyperparams,
  onParamChange,
  onParamReset,
  traversalResult,
  traversalError,
  traversalLoading,
  onRunTraversal,
  onCloseTraversal,
}) {
  // Derive initial nodes/edges from manifest (or demo fallback)
  const { initialNodes, initialEdges } = useMemo(() => {
    if (manifest?.components?.length) {
      return { initialNodes: manifestToFlow(manifest).nodes, initialEdges: manifestToFlow(manifest).edges }
    }
    return { initialNodes: DEMO_NODES, initialEdges: DEMO_EDGES }
  }, [manifest])

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes)
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges)
  const [selectedNode, setSelectedNode] = useState(null)
  const [legendOpen, setLegendOpen] = useState(true)

  // Re-build the graph whenever the manifest changes (e.g. new paper loaded)
  useEffect(() => {
    if (manifest?.components?.length) {
      const { nodes: n, edges: e } = manifestToFlow(manifest)
      setNodes(n)
      setEdges(e)
    } else {
      setNodes(DEMO_NODES)
      setEdges(DEMO_EDGES)
    }
    setSelectedNode(null)
  }, [manifest])

  const onConnect = useCallback(
    (params) => setEdges((eds) => addEdge(params, eds)),
    [setEdges],
  )

  const onNodeClick = useCallback((_event, node) => setSelectedNode(node), [])
  const onPaneClick = useCallback(() => setSelectedNode(null), [])

  const isManifestMode = !!(manifest?.components?.length)

  // Derive dynamic legend based on taxonomy or actual component kinds
  const legend = useMemo(() => {
    if (!isManifestMode) return LEGEND_GROUPS
    
    // 1. If manifest has an explicit taxonomy, use it
    if (manifest.taxonomy?.length) {
      return manifest.taxonomy.map(t => ({
        label: t.label,
        accent: resolveAccent(t.kind, t)
      }))
    }

    // 2. Fallback: only show categories present in the components
    const usedKinds = Array.from(new Set(manifest.components.map(c => c.kind)))
    return usedKinds.map(kind => ({
      label: kind.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      accent: resolveAccent(kind)
    }))
  }, [manifest, isManifestMode])

  // Derive which edge types are actually present in this manifest
  const edgeLegend = useMemo(() => {
    const EDGE_DEFS = [
      {
        key: 'normal',
        color: '#6366F1',
        label: 'Sequential Data Flow',
        sublabel: 'Tensors moving forward to adjacent components',
        dash: null,
        width: 2,
        animated: true,
      },
      {
        key: 'cross',
        color: '#F59E0B',
        label: 'Cross-Branch Flow',
        sublabel: 'Data flowing between distinct sub-networks (e.g., cross-attention)',
        dash: '6 3',
        width: 2.5,
        animated: false,
      },
      {
        key: 'skip',
        color: '#C4B5FD',
        label: 'Skip Connection',
        sublabel: 'Bypasses intermediate layers to preserve gradient flow',
        dash: '5 3',
        width: 1.5,
        animated: false,
      },
    ]

    if (!isManifestMode) return EDGE_DEFS

    const comps = manifest.components || []
    const compMap = Object.fromEntries(comps.map(c => [c.id, c]))

    const presentKeys = new Set()
    comps.forEach(c => {
      ;(c.depends_on || []).forEach(dep => {
        const src = compMap[dep]
        if (!src) return
        const tName = (c.name || '').toLowerCase()
        const sId   = (src.id || '').toLowerCase()
        const tId   = (c.id || '').toLowerCase()
        const isCross = tName.includes('cross') ||
          (sId.includes('enc') && tId.includes('dec') && c.kind?.includes('attention'))
        if (isCross) { presentKeys.add('cross'); return }
        if (c.kind === 'residual') { presentKeys.add('skip'); return }
        presentKeys.add('normal')
      })
    })

    return EDGE_DEFS.filter(e => presentKeys.has(e.key))
  }, [manifest, isManifestMode])

  // Auto-hide the paper banner after 3.5 s; re-show on hover
  const [bannerVisible, setBannerVisible] = useState(true)
  useEffect(() => {
    if (!isManifestMode) return
    setBannerVisible(true)
    const t = setTimeout(() => setBannerVisible(false), 3500)
    return () => clearTimeout(t)
  }, [manifest, isManifestMode])

  return (
    <div className="sandbox-canvas">
      {viewMode === 'model' ? (
      <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onNodeClick={onNodeClick}
          onPaneClick={onPaneClick}
          nodeTypes={NODE_TYPES}
          edgeTypes={EDGE_TYPES}
          fitView
        >
          {/* SVG filter for glowing animated dot on data flow edges */}
          <svg style={{ position: 'absolute', width: 0, height: 0 }}>
            <defs>
              <filter id="glow-flow" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
          </svg>
          <Controls />
          <Background color="#E5E5E5" gap={16} />

          {/* Paper banner — auto-hides after 3.5 s, re-appears on hover */}
          {isManifestMode && (
            <Panel position="top-center">
              <div
                className="sandbox-top-controls"
                onMouseEnter={() => setBannerVisible(true)}
                onMouseLeave={() => setBannerVisible(false)}
              >
                <div
                  className="sandbox-paper-banner"
                  style={{ opacity: bannerVisible ? 1 : 0, transform: bannerVisible ? 'translateY(0)' : 'translateY(-8px)' }}
                >
                  <span className="sandbox-paper-arxiv">{manifest.paper?.arxiv_id}</span>
                  <span className="sandbox-paper-title">{manifest.paper?.title}</span>
                  <span className="sandbox-paper-count">{manifest.components.length} components</span>
                </div>
              </div>
            </Panel>
          )}

          <Panel position="bottom-right">
            <div className={`sandbox-legend ${!legendOpen ? 'collapsed' : ''}`}>
              <div 
                className="legend-header" 
                onClick={() => setLegendOpen(!legendOpen)}
                style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', cursor: 'pointer' }}
              >
                <span className="legend-title" style={{ margin: 0 }}>Legend</span>
                <button className="legend-toggle-btn" aria-label="Toggle Legend" style={{ background: 'none', border: 'none', fontSize: '10px', color: '#7A93B0', cursor: 'pointer' }}>
                  {legendOpen ? '▼' : '▲'}
                </button>
              </div>

              {legendOpen && (
                <div className="legend-content" style={{ display: 'flex', flexDirection: 'column', gap: '7px', marginTop: '4px' }}>
                  {isManifestMode ? (
                    <>
                      {legend.map((g) => (
                        <div key={g.label} className="legend-row">
                          <span
                            className="legend-node"
                            style={{
                              background: g.accent.bg,
                              border: `1px solid ${g.accent.border}`,
                              borderLeft: `3px solid ${g.accent.accent}`,
                            }}
                          />
                          <span className="legend-label">{g.label}</span>
                        </div>
                      ))}

                      {edgeLegend.length > 0 && (
                        <>
                          <div className="legend-divider" style={{ margin: '8px 0' }} />
                          <span className="legend-title">Edge Types</span>
                          {edgeLegend.map(e => (
                            <div key={e.key} className="legend-row" style={{ alignItems: 'flex-start', gap: 8 }}>
                              <svg width="36" height="14" style={{ flexShrink: 0, marginTop: 3, overflow: 'visible' }}>
                                <defs>
                                  <marker id={`lg-arr-${e.key}`} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                                    <path d="M0,1 L5,3 L0,5 Z" fill={e.color} />
                                  </marker>
                                </defs>
                                <line x1="2" y1="7" x2="30" y2="7"
                                  stroke={e.color}
                                  strokeWidth={e.width}
                                  strokeDasharray={e.dash || undefined}
                                  markerEnd={`url(#lg-arr-${e.key})`}
                                />
                                {e.animated && (
                                  <circle r="2.5" fill={e.color}>
                                    <animateMotion dur="1.2s" repeatCount="indefinite" path="M2,7 L30,7" />
                                  </circle>
                                )}
                              </svg>
                              <div>
                                <div className="legend-label" style={{ fontWeight: 700 }}>{e.label}</div>
                                <div style={{ fontSize: 9, color: '#9CA3AF', marginTop: 1 }}>{e.sublabel}</div>
                              </div>
                            </div>
                          ))}
                        </>
                      )}
                    </>
                  ) : (
                    <>
                      <div className="legend-row"><span className="legend-node navy" /><span className="legend-label">Encoder</span></div>
                      <div className="legend-row"><span className="legend-node orange" /><span className="legend-label">Decoder</span></div>
                      <div className="legend-row"><span className="legend-node plain" /><span className="legend-label">Sub-layer</span></div>
                      <div className="legend-divider" />
                      <div className="legend-row"><span className="legend-edge animated" /><span className="legend-label">Data flow</span></div>
                      <div className="legend-row"><span className="legend-edge dashed" /><span className="legend-label">Cross attention</span></div>
                    </>
                  )}
                </div>
              )}
            </div>
          </Panel>
        </ReactFlow>
      ) : viewMode === 'trace' ? (
        <TraceView
          result={traversalResult}
          error={traversalError}
          loading={traversalLoading}
          onRun={onRunTraversal}
          manifest={manifest}
        />
      ) : (
        <div className="code-view-container">
          <div className="code-view-header">
            <h3>Python Implementation</h3>
            <span className="code-view-badge">Auto-generated from Manifest</span>
          </div>
          <div className="code-view-placeholder">
            <div className="code-mock-line"><span>class</span> TransformerLayer(nn.Module):</div>
            <div className="code-mock-line indent-1"><span>def</span> __init__(self, config):</div>
            <div className="code-mock-line indent-2">super().__init__()</div>
            <div className="code-mock-line indent-2">self.attention = MultiHeadAttention(config)</div>
            <div className="code-mock-line indent-2">self.norm1 = nn.LayerNorm(config.hidden_size)</div>
            <div className="code-mock-line indent-1">...</div>
            <div className="code-empty-state">
              <div className="code-empty-icon">🐍</div>
              <p>Code generation is being prepared for this architecture.</p>
              <button className="btn-primary" disabled>Generate PyTorch Source</button>
            </div>
          </div>
        </div>
      )}

      <NodeInfoPopup
        node={selectedNode}
        params={selectedNode ? hyperparams[selectedNode.id] : null}
        onParamChange={onParamChange}
        onParamReset={onParamReset}
        onClose={() => setSelectedNode(null)}
        isManifestMode={isManifestMode}
      />
    </div>
  )
}
