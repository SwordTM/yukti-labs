// manifestToFlow — converts a ComponentManifest into a React Flow graph
import React from 'react'
import { getBezierPath, BaseEdge, MarkerType } from '@xyflow/react'

// ── Animated data-flow edge ──────────────────────────────────────────────────
// A circle travels along the real bezier curve of each main data flow edge.
export function AnimatedDataFlowEdge({
  id, sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition, markerEnd, style,
}) {
  const [edgePath] = getBezierPath({ sourceX, sourceY, sourcePosition, targetX, targetY, targetPosition })
  return (
    <g>
      <BaseEdge id={id} path={edgePath} markerEnd={markerEnd} style={style} />
      <path id={`${id}-p`} d={edgePath} fill="none" stroke="none" />
      <circle r={4} fill="#6366F1" fillOpacity={0.85}>
        <animateMotion dur="1.6s" repeatCount="indefinite" calcMode="linear">
          <mpath href={`#${id}-p`} />
        </animateMotion>
      </circle>
    </g>
  )
}

export const FLOW_EDGE_TYPES = { dataflow: AnimatedDataFlowEdge }

// ── Edge classification ───────────────────────────────────────────────────────
function classifyEdge(srcComp, tgtComp, rankMap) {
  if (!srcComp || !tgtComp) return 'normal'
  const tName = (tgtComp.name || '').toLowerCase()
  const sId   = (srcComp.id  || '').toLowerCase()
  const tId   = (tgtComp.id  || '').toLowerCase()
  const isCross = tName.includes('cross') ||
    (sId.includes('enc') && tId.includes('dec') && tgtComp.kind?.includes('attention'))
  if (isCross) return 'cross'
  if (tgtComp.kind === 'residual') return 'skip'
  const srcRank = rankMap[srcComp.id] ?? 0
  const tgtRank = rankMap[tgtComp.id] ?? 0
  if (tgtRank - srcRank > 1) return 'skip'
  return 'normal'
}

// ── Topological Sort ──────────────────────────────────────────────────────────
function topoSort(components) {
  const idSet = new Set(components.map((c) => c.id))
  const inDeg = {}
  const adj = {}
  components.forEach((c) => { inDeg[c.id] = 0; adj[c.id] = [] })
  components.forEach((c) => {
    ;(c.depends_on || []).forEach((dep) => {
      if (idSet.has(dep)) { adj[dep].push(c.id); inDeg[c.id]++ }
    })
  })
  const queue = components.filter((c) => inDeg[c.id] === 0).map((c) => c.id)
  const sorted = []
  while (queue.length) {
    const id = queue.shift()
    sorted.push(id)
    adj[id].forEach((nxt) => { if (--inDeg[nxt] === 0) queue.push(nxt) })
  }
  components.forEach((c) => { if (!sorted.includes(c.id)) sorted.push(c.id) })
  return { sorted, adj }
}

// ── Longest-Path Rank Assignment ─────────────────────────────────────────────
function assignRanks(components) {
  const { sorted, adj } = topoSort(components)
  const compMap = Object.fromEntries(components.map((c) => [c.id, c]))
  const rank = {}
  sorted.forEach((id) => {
    const deps = (compMap[id]?.depends_on || []).filter((d) => rank[d] !== undefined)
    rank[id] = deps.length ? Math.max(...deps.map((d) => rank[d])) + 1 : 0
  })
  return { compMap, rank, adj, sorted }
}

// ── Sub-tree / Column Detection ───────────────────────────────────────────────
function detectColumns(components, compMap, adj, sorted) {
  const roots = components.filter((c) => !(c.depends_on || []).some((d) => compMap[d]))
  if (roots.length <= 1) return [components.map((c) => c.id)]

  // Trace reachability from each root
  const rootReachable = roots.map((root) => {
    const visited = new Set()
    const queue = [root.id]
    while (queue.length) {
      const id = queue.shift()
      if (visited.has(id)) continue
      visited.add(id)
      ;(adj[id] || []).forEach((nxt) => queue.push(nxt))
    }
    return { rootId: root.id, set: visited }
  })

  const columnMap = {}
  const nexusNodes = []

  sorted.forEach((id) => {
    const reachableFrom = rootReachable.filter((r) => r.set.has(id))
    
    if (reachableFrom.length > 1) {
      // Shared across multiple subtrees -> Move to Nexus
      nexusNodes.push(id)
      columnMap[id] = 'NEXUS'
    } else if (reachableFrom.length === 1) {
      columnMap[id] = reachableFrom[0].rootId
    } else {
      columnMap[id] = roots[0].id
    }
  })

  // Group into actual columns: [Left, ..., Nexus, ..., Right]
  const groups = {}
  roots.forEach((r) => { groups[r.id] = [] })
  groups['NEXUS'] = []

  sorted.forEach((id) => {
    const col = columnMap[id]
    groups[col].push(id)
  })

  // Order columns: we want [Subtree1, NEXUS, Subtree2]
  // This assumes the paper extraction lists roots in order (e.g. Enc, then Dec)
  const columns = []
  if (roots.length >= 2) {
    columns.push(groups[roots[0].id])
    if (groups['NEXUS'].length) columns.push(groups['NEXUS'])
    columns.push(groups[roots[1].id])
    // Append any remaining roots (unlikely for standard papers)
    for (let i = 2; i < roots.length; i++) columns.push(groups[roots[i].id])
  } else {
    columns.push(groups[roots[0].id])
    if (groups['NEXUS'].length) columns.push(groups['NEXUS'])
  }

  return columns.filter(c => c.length > 0)
}

// ── Position Assignment ───────────────────────────────────────────────────────
function assignPositions(components) {
  const { compMap, rank, adj, sorted } = assignRanks(components)
  const columns = detectColumns(components, compMap, adj, sorted)
  
  const NODE_W = 210, HORIZ_GAP = 260, COL_GAP = 300, VERT_GAP = 150
  let xOffset = 0
  const columnXOffsets = []

  columns.forEach((colIds) => {
    const rankCounts = {}
    colIds.forEach((id) => { const r = rank[id]; rankCounts[r] = (rankCounts[r] || 0) + 1 })
    const maxNodesInRank = Math.max(...Object.values(rankCounts))
    const colWidth = (maxNodesInRank - 1) * HORIZ_GAP + NODE_W
    
    columnXOffsets.push({ ids: colIds, xStart: xOffset, colWidth })
    xOffset += colWidth + COL_GAP
  })

  const totalWidth = xOffset - COL_GAP
  const globalShift = -totalWidth / 2

  columnXOffsets.forEach(({ ids, xStart, colWidth }) => {
    const rankGroups = {}
    ids.forEach((id) => { 
      const r = rank[id]
      if (!rankGroups[r]) rankGroups[r] = []
      rankGroups[r].push(id) 
    })

    Object.entries(rankGroups).forEach(([r, idsInRank]) => {
      const groupW = (idsInRank.length - 1) * HORIZ_GAP
      const localStart = xStart + (colWidth - groupW) / 2
      idsInRank.forEach((id, i) => {
        compMap[id]._pos = { 
          x: globalShift + localStart + i * HORIZ_GAP, 
          y: parseInt(r) * VERT_GAP 
        }
      })
    })
  })
  return { compMap, rank }
}

// ── Main Export ───────────────────────────────────────────────────────────────
export function manifestToFlow(manifest) {
  if (!manifest?.components?.length) return { nodes: [], edges: [] }

  const { compMap, rank } = assignPositions(manifest.components)
  const idSet = new Set(manifest.components.map((c) => c.id))

  const nodes = manifest.components.map((comp) => ({
    id: comp.id,
    position: comp._pos ?? { x: 0, y: 0 },
    type: 'component',
    data: { label: comp.name, component: comp, manifest },
    className: comp.is_experimental ? 'experimental-node' : '',
  }))

  const edges = []
  manifest.components.forEach((comp) => {
    ;(comp.depends_on || []).forEach((dep) => {
      if (!idSet.has(dep)) return
      const srcComp = compMap[dep]
      const type    = classifyEdge(srcComp, comp, rank)
      const isCross = type === 'cross'
      const isSkip  = type === 'skip'

      edges.push({
        id:        `e-${dep}-${comp.id}`,
        source:    dep,
        target:    comp.id,
        type:      type === 'normal' ? 'dataflow' : 'default',
        animated:  false,
        style: {
          stroke:          isCross ? '#F59E0B' : isSkip ? '#C4B5FD' : '#6366F1',
          strokeWidth:     isCross ? 2.5 : isSkip ? 1.5 : 2,
          strokeDasharray: isSkip || isCross ? '6 3' : undefined,
          opacity:         isSkip ? 0.6 : 1,
        },
        markerEnd: {
          type:   MarkerType.ArrowClosed,
          color:  isCross ? '#F59E0B' : isSkip ? '#C4B5FD' : '#6366F1',
          width: 12, height: 12,
        },
        label:          isCross ? 'K, V' : undefined,
        labelStyle:     { fill: '#B45309', fontWeight: 700, fontSize: 10 },
        labelBgStyle:   { fill: '#FEF3C7', borderRadius: 3 },
        labelBgPadding: [3, 5],
        zIndex:         isSkip ? 0 : 1,
      })
    })
  })

  return { nodes, edges }
}
