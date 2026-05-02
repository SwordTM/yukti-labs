import React, { useMemo } from 'react'
import dagre from '@dagrejs/dagre'
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  Handle,
  Position,
  useNodesState,
  useEdgesState,
  MarkerType,
  getBezierPath,
  BaseEdge,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

// ─── Kind metadata ────────────────────────────────────────────────────────────
const KIND_META = {
  input_embedding:      { bg: '#EFF6FF', border: '#3B82F6', text: '#1E40AF', icon: '⌨' },
  positional_encoding:  { bg: '#F0FDF4', border: '#22C55E', text: '#166534', icon: '📍' },
  multi_head_attention: { bg: '#FDF4FF', border: '#A855F7', text: '#6B21A8', icon: '⊛' },
  attention:            { bg: '#FDF4FF', border: '#A855F7', text: '#6B21A8', icon: '⊛' },
  feedforward:          { bg: '#FFF7ED', border: '#F97316', text: '#9A3412', icon: '⚡' },
  layernorm:            { bg: '#FFFBEB', border: '#F59E0B', text: '#92400E', icon: '⊕' },
  rmsnorm:              { bg: '#FFFBEB', border: '#F59E0B', text: '#92400E', icon: '⊕' },
  residual:             { bg: '#F5F3FF', border: '#8B5CF6', text: '#5B21B6', icon: '+' },
  softmax:              { bg: '#FFF1F2', border: '#F43F5E', text: '#9F1239', icon: '∑' },
  masking:              { bg: '#F9FAFB', border: '#D1D5DB', text: '#9CA3AF', icon: '⊘' },
  linear_projection:    { bg: '#F0FDFA', border: '#14B8A6', text: '#134E4A', icon: '→' },
  output_head:          { bg: '#FEF9C3', border: '#CA8A04', text: '#713F12', icon: '🎯' },
  other:                { bg: '#F9FAFB', border: '#9CA3AF', text: '#4B5563', icon: '◇' },
}

const NODE_W = 200
const NODE_H = 70

// ─── Compute topological depth independently of dagre ──────────────────────
function computeDepths(components) {
  const idSet  = new Set(components.map(c => c.id))
  const cache  = {}
  function depth(id, stack = new Set()) {
    if (id in cache)      return cache[id]
    if (stack.has(id))    return 0
    stack.add(id)
    const comp = components.find(c => c.id === id)
    if (!comp?.depends_on?.length) { cache[id] = 0; return 0 }
    const parents = comp.depends_on.filter(p => idSet.has(p))
    cache[id] = parents.length
      ? Math.max(...parents.map(p => depth(p, new Set(stack)))) + 1
      : 0
    return cache[id]
  }
  components.forEach(c => depth(c.id))
  return cache
}

// ─── Dagre layout — only primary (deepest parent) edges fed to dagre ─────────
// Skip connections and residual long-range edges are drawn but don't affect ranking
function buildDagreLayout(components) {
  if (!components.length) return {}

  const depthMap = computeDepths(components)
  const idSet    = new Set(components.map(c => c.id))

  const g = new dagre.graphlib.Graph()
  g.setDefaultEdgeLabel(() => ({}))
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 90, marginx: 50, marginy: 50 })

  components.forEach(c => g.setNode(c.id, { width: NODE_W, height: NODE_H }))

  // For each node, only add ONE edge to dagre — from its deepest (primary) parent.
  // This prevents skip / residual connections from distorting rank placement.
  components.forEach(c => {
    const validParents = (c.depends_on || []).filter(p => idSet.has(p))
    if (!validParents.length) return
    // Primary parent = the one with the highest topological depth
    const primary = validParents.reduce((best, p) =>
      (depthMap[p] ?? 0) >= (depthMap[best] ?? 0) ? p : best
    )
    g.setEdge(primary, c.id)
  })

  dagre.layout(g)

  const pos = {}
  components.forEach(c => {
    const n = g.node(c.id)
    pos[c.id] = n ? { x: n.x - NODE_W / 2, y: n.y - NODE_H / 2 } : { x: 0, y: 0 }
  })
  return pos
}

// ─── Edge classification ──────────────────────────────────────────────────────
function classifyEdge(src, tgt, depthMap) {
  if (!src || !tgt) return 'normal'
  const tName = (tgt.name || '').toLowerCase()
  const sId   = (src.id   || '').toLowerCase()
  const tId   = (tgt.id   || '').toLowerCase()

  // Cross / encoder-decoder attention
  const isCross = tName.includes('cross') ||
    (sId.includes('enc') && tId.includes('dec') && tgt.kind?.includes('attention'))
  if (isCross) return 'cross'

  // Skip connection: depth gap > 1 between parent and child
  const srcDepth = depthMap[src.id] ?? 0
  const tgtDepth = depthMap[tgt.id] ?? 0
  if (tgtDepth - srcDepth > 1) return 'skip'

  // Residual/norm kinds
  if (tgt.kind === 'residual') return 'skip'

  return 'normal'
}

// ─── Component node ───────────────────────────────────────────────────────────
function ComponentNode({ data }) {
  const meta     = KIND_META[data.kind] || KIND_META.other
  const isActive = data.activeStep?.component_id === data.id
  const step     = data.steps?.find(s => s.component_id === data.id)
  const params   = step?.parameter_count
  const isMask   = data.kind === 'masking'

  const paramLabel = params
    ? params >= 1e6 ? `${(params/1e6).toFixed(1)}M`
    : params >= 1e3 ? `${(params/1e3).toFixed(0)}K`
    : `${params}` : null

  return (
    <div style={{
      background:   isActive ? meta.border : meta.bg,
      border:       `2px solid ${meta.border}`,
      borderRadius: 12,
      padding:      '8px 14px',
      width:        NODE_W,
      opacity:      isMask ? 0.65 : 1,
      boxShadow:    isActive
        ? `0 0 0 4px ${meta.border}33, 0 4px 20px ${meta.border}22`
        : '0 1px 4px rgba(0,0,0,0.07)',
      transition:   'all 0.2s ease',
      cursor:       'pointer',
      boxSizing:    'border-box',
    }}>
      <Handle type="target" position={Position.Top}
        style={{ background: meta.border, width: 8, height: 8, border: '2px solid white' }} />

      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        <span style={{ fontSize: 14, flexShrink: 0 }}>{meta.icon}</span>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{
            fontSize: 9, fontWeight: 700, textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: isActive ? 'rgba(255,255,255,0.7)' : meta.text,
            marginBottom: 1,
          }}>
            {data.kind.replace(/_/g, ' ')}
          </div>
          <div style={{
            fontSize: 12, fontWeight: 700,
            color: isActive ? 'white' : '#111827',
            lineHeight: 1.25,
            whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis',
          }}>
            {data.label}
          </div>
        </div>
        {paramLabel && (
          <span style={{
            fontSize: 9, fontWeight: 700, padding: '1px 5px', borderRadius: 8,
            background: isActive ? 'rgba(255,255,255,0.25)' : '#F3F4F6',
            color: isActive ? 'white' : '#6B7280', flexShrink: 0,
          }}>{paramLabel}</span>
        )}
      </div>

      {isActive && step && (
        <div style={{
          marginTop: 5, fontSize: 9, fontFamily: 'monospace',
          background: 'rgba(0,0,0,0.15)', borderRadius: 5, padding: '2px 6px',
          color: 'white', lineHeight: 1.5,
        }}>
          [{step.input_symbolic?.join(', ')}] → [{step.output_symbolic?.join(', ')}]
          {step.key_insight && (
            <div style={{ fontFamily: 'sans-serif', fontStyle: 'italic', marginTop: 2, opacity: 0.85 }}>
              {step.key_insight}
            </div>
          )}
        </div>
      )}

      <Handle type="source" position={Position.Bottom}
        style={{ background: meta.border, width: 8, height: 8, border: '2px solid white' }} />
    </div>
  )
}

// ─── Custom animated data-flow edge ──────────────────────────────────────────
// Renders the real bezier path + a glowing circle that travels along it.
function AnimatedDataFlowEdge({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, markerEnd, style,
}) {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return (
    <g>
      {/* The edge line itself */}
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      {/* Animated packet travelling along the path */}
      <circle r={4} fill="#6366F1" fillOpacity={0.9}
        filter="url(#glow-flow)">
        <animateMotion dur="1.6s" repeatCount="indefinite" calcMode="linear">
          <mpath href={`#${id}-path`} />
        </animateMotion>
      </circle>
      {/* Named path for mpath reference */}
      <path id={`${id}-path`} d={edgePath} fill="none" stroke="none" />
    </g>
  )
}

const nodeTypes = { component: ComponentNode }
const edgeTypes = { dataflow: AnimatedDataFlowEdge }

// ─── Main component ───────────────────────────────────────────────────────────
export default function ArchitectureFlow({ manifest, trace, activeStepIndex, onNodeClick }) {
  const activeStep = trace?.steps?.[activeStepIndex] ?? null
  const compMap    = useMemo(
    () => Object.fromEntries(manifest.components.map(c => [c.id, c])),
    [manifest.components]
  )
  const depthMap   = useMemo(() => computeDepths(manifest.components), [manifest.components])
  const positions  = useMemo(() => buildDagreLayout(manifest.components), [manifest.components])

  const nodes = useMemo(() => manifest.components.map(c => ({
    id:       c.id,
    type:     'component',
    position: positions[c.id] || { x: 0, y: 0 },
    data:     { id: c.id, label: c.name, kind: c.kind, activeStep, steps: trace?.steps ?? [] },
  })), [manifest.components, positions, activeStep, trace])

  const edges = useMemo(() => manifest.components.flatMap(c =>
    (c.depends_on || [])
      .filter(dep => compMap[dep])
      .map(dep => {
        const type     = classifyEdge(compMap[dep], c, depthMap)
        const isCross  = type === 'cross'
        const isSkip   = type === 'skip'
        const isNormal = type === 'normal'

        return {
          id:        `${dep}->${c.id}`,
          source:    dep,
          target:    c.id,
          // Use custom animated edge for main data flow only
          type:      isNormal ? 'dataflow' : isSkip ? 'smoothstep' : 'default',
          animated:  false,   // animateMotion handles it for dataflow
          style: {
            stroke:          isCross ? '#F59E0B' : isSkip ? '#C4B5FD' : '#6366F1',
            strokeWidth:     isCross ? 2.5 : isSkip ? 1.5 : 2,
            strokeDasharray: isSkip ? '6 3' : undefined,
            opacity:         isSkip ? 0.55 : 1,
          },
          markerEnd: {
            type:   MarkerType.ArrowClosed,
            color:  isCross ? '#F59E0B' : isSkip ? '#C4B5FD' : '#6366F1',
            width: 12, height: 12,
          },
          label:      isCross ? 'K, V' : undefined,
          labelStyle: { fill: '#B45309', fontWeight: 700, fontSize: 10 },
          labelBgStyle: { fill: '#FEF3C7', borderRadius: 3 },
          labelBgPadding: [3, 5],
          zIndex:     isSkip ? 0 : 1,
        }
      })
  ), [manifest.components, compMap, depthMap, activeStep])

  const [rfNodes, , onNodesChange] = useNodesState(nodes)
  const [rfEdges, , onEdgesChange] = useEdgesState(edges)

  return (
    <div style={{
      width: '100%', height: 620, borderRadius: 14, overflow: 'hidden',
      border: '1px solid #E5E7EB', background: '#FAFAFA', position: 'relative',
    }}>
      {/* Legend */}
      <div style={{
        position: 'absolute', top: 10, left: 10, zIndex: 10,
        background: 'rgba(255,255,255,0.96)', backdropFilter: 'blur(10px)',
        padding: '8px 14px', borderRadius: 10, border: '1px solid #E5E7EB',
        boxShadow: '0 2px 8px rgba(0,0,0,0.07)',
        display: 'flex', flexDirection: 'column', gap: 7,
      }}>
        <div style={{ fontSize: 9, fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.08em', color: '#9CA3AF', marginBottom: 2 }}>
          Legend
        </div>
        <LegendItem
          color="#6366F1" animated
          label="Main data flow"
          sublabel="Tensors moving forward through the model"
        />
        <LegendItem
          color="#F59E0B" dashed dashArray="6 3" strokeWidth={2.5}
          label="Cross-attention  (K, V)"
          sublabel="Encoder passes Keys & Values to Decoder"
        />
        <LegendItem
          color="#C4B5FD" dashed dashArray="5 3" strokeWidth={1.5}
          label="Residual / skip connection"
          sublabel="Input added back to output — prevents vanishing gradients"
        />
        <div style={{ borderTop: '1px solid #F3F4F6', marginTop: 2, paddingTop: 6, display: 'flex', alignItems: 'flex-start', gap: 7 }}>
          <div style={{
            width: 28, height: 16, borderRadius: 4, flexShrink: 0, marginTop: 1,
            background: '#F9FAFB', border: '1.5px dashed #D1D5DB',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            fontSize: 9, color: '#9CA3AF',
          }}>⊘</div>
          <div>
            <div style={{ fontSize: 10, fontWeight: 600, color: '#6B7280' }}>Masking / auxiliary input</div>
            <div style={{ fontSize: 9, color: '#9CA3AF', marginTop: 1 }}>Controls which tokens can attend to which</div>
          </div>
        </div>
      </div>

      <ReactFlow
        nodes={rfNodes}
        edges={rfEdges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={(_, node) => onNodeClick?.(node.id)}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        fitView
        fitViewOptions={{ padding: 0.25 }}
        minZoom={0.1}
        maxZoom={2}
        proOptions={{ hideAttribution: true }}
      >
        {/* SVG filter for the animated dot glow */}
        <svg style={{ position: 'absolute', width: 0, height: 0 }}>
          <defs>
            <filter id="glow-flow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2" result="blur" />
              <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
            </filter>
          </defs>
        </svg>
        <Background color="#E9EAF0" gap={24} size={1} />
        <Controls showInteractive={false} />
        <MiniMap
          nodeColor={n => KIND_META[n.data?.kind]?.border ?? '#9CA3AF'}
          pannable zoomable
          style={{ borderRadius: 8, border: '1px solid #E5E7EB' }}
        />
      </ReactFlow>
    </div>
  )
}

function LegendItem({ color, label, sublabel, animated, dashed, dashArray = '5 3', strokeWidth = 2 }) {
  const W = 36, H = 14, arrowId = `arr-${color.replace('#','')}`
  return (
    <span style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
      <svg width={W} height={H} style={{ flexShrink: 0, overflow: 'visible', marginTop: 2 }}>
        <defs>
          <marker id={arrowId} markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,1 L5,3 L0,5 Z" fill={color} />
          </marker>
        </defs>
        <line
          x1="2" y1={H / 2} x2={W - 6} y2={H / 2}
          stroke={color}
          strokeWidth={strokeWidth}
          strokeDasharray={dashed ? dashArray : undefined}
          markerEnd={`url(#${arrowId})`}
        />
        {animated && (
          <circle r="2.5" fill={color}>
            <animateMotion dur="1.2s" repeatCount="indefinite" path={`M2,${H/2} L${W-6},${H/2}`} />
          </circle>
        )}
      </svg>
      <div>
        <div style={{ fontSize: 10, fontWeight: 700, color: '#374151', whiteSpace: 'nowrap' }}>
          {label}
        </div>
        {sublabel && (
          <div style={{ fontSize: 9, color: '#9CA3AF', marginTop: 1, lineHeight: 1.3 }}>
            {sublabel}
          </div>
        )}
      </div>
    </span>
  )
}
