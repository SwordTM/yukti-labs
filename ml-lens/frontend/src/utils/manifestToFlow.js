/**
 * Convert a ComponentManifest into a hierarchical React Flow graph.
 * Uses a column-aware layout:
 * 1. Topological sort to determine dependency depth (rank)
 * 2. Detect independent sub-trees (e.g. Encoder vs Decoder) and assign them separate columns
 * 3. Position nodes top-to-bottom within each column, centered horizontally across columns
 */

const KIND_EMOJI = {
  input_embedding:     '🔢',
  positional_encoding: '📍',
  linear_projection:   '➡',
  attention:           '🎯',
  multi_head_attention:'🎯',
  feedforward:         '⚡',
  layernorm:           '📐',
  rmsnorm:             '📐',
  residual:            '🔁',
  softmax:             '📊',
  masking:             '🔒',
  output_head:         '📤',
  other:               '🔷',
}

const KIND_STYLE = {
  input_embedding:     { background: '#EEF3FA', border: '1px solid #1E3A5F',  fontWeight: 600 },
  positional_encoding: { background: '#EEF3FA', border: '1px solid #1E3A5F',  fontWeight: 600 },
  linear_projection:   { background: '#F5F5F5', border: '1px solid #9CA3AF' },
  attention:           { background: '#EDF7ED', border: '1px solid #16A34A',  fontWeight: 600 },
  multi_head_attention:{ background: '#EDF7ED', border: '2px solid #16A34A',  fontWeight: 700 },
  feedforward:         { background: '#F5F5F5', border: '1px solid #9CA3AF' },
  layernorm:           { background: '#FDFCE9', border: '1px solid #CA8A04' },
  rmsnorm:             { background: '#FDFCE9', border: '1px solid #CA8A04' },
  residual:            { background: '#EEF3FA', border: '2px solid #1E3A5F',  fontWeight: 700 },
  softmax:             { background: '#FFF7ED', border: '1px solid #F97316' },
  masking:             { background: '#FFF1F2', border: '1px solid #F43F5E' },
  output_head:         { background: '#EEF3FA', border: '1px solid #1E3A5F',  fontWeight: 600 },
  other:               { background: 'white',   border: '1px solid #D6E4F0' },
}

const BASE_NODE_STYLE = {
  borderRadius: 8,
  padding: 10,
  fontFamily: 'Poppins, Arial, sans-serif',
  fontSize: 13,
  width: 210,
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
// Ensures that nodes visually appear BELOW all their dependencies
function assignRanks(components) {
  const { sorted, adj } = topoSort(components)
  const compMap = Object.fromEntries(components.map((c) => [c.id, c]))
  const rank = {}

  // Forward pass: rank = max(parent ranks) + 1
  sorted.forEach((id) => {
    const deps = (compMap[id]?.depends_on || []).filter((d) => rank[d] !== undefined)
    rank[id] = deps.length ? Math.max(...deps.map((d) => rank[d])) + 1 : 0
  })

  return { compMap, rank, adj, sorted }
}

// ── Sub-tree / Column Detection ───────────────────────────────────────────────
// Finds groups of components that are reachable from each root independently
// so encoder and decoder can be placed in separate horizontal columns
function detectColumns(components, compMap, adj, sorted) {
  const roots = components.filter((c) => !(c.depends_on || []).some((d) => compMap[d]))
  
  if (roots.length <= 1) {
    // Single tree — one column
    return [components.map((c) => c.id)]
  }

  // BFS from each root to find its reachable set
  const reachable = {}
  roots.forEach((root) => {
    const visited = new Set()
    const queue = [root.id]
    while (queue.length) {
      const id = queue.shift()
      if (visited.has(id)) continue
      visited.add(id)
      ;(adj[id] || []).forEach((nxt) => queue.push(nxt))
    }
    reachable[root.id] = visited
  })

  // Assign each component to the column of its closest root
  const column = {}
  sorted.forEach((id) => {
    for (const root of roots) {
      if (reachable[root.id].has(id)) {
        column[id] = root.id
        break
      }
    }
    if (!column[id]) column[id] = roots[0].id
  })

  // Group by column
  const colGroups = {}
  sorted.forEach((id) => {
    const col = column[id]
    if (!colGroups[col]) colGroups[col] = []
    colGroups[col].push(id)
  })

  return Object.values(colGroups)
}

// ── Position Assignment ───────────────────────────────────────────────────────
function assignPositions(components) {
  const { compMap, rank, adj, sorted } = assignRanks(components)
  const columns = detectColumns(components, compMap, adj, sorted)

  const NODE_W = 210
  const HORIZ_GAP = 260  // Gap between nodes in same rank
  const COL_GAP = 300    // Gap between encoder/decoder columns
  const VERT_GAP = 150   // Vertical gap between ranks

  // For each column, figure out the max nodes-per-rank to determine column width
  let xOffset = 0
  const columnXOffsets = []

  columns.forEach((colIds) => {
    // Find max simultaneous nodes at any rank in this column
    const rankCounts = {}
    colIds.forEach((id) => {
      const r = rank[id]
      rankCounts[r] = (rankCounts[r] || 0) + 1
    })
    const maxNodesInRank = Math.max(...Object.values(rankCounts))
    const colWidth = (maxNodesInRank - 1) * HORIZ_GAP + NODE_W

    columnXOffsets.push({ ids: colIds, xStart: xOffset, colWidth })
    xOffset += colWidth + COL_GAP
  })

  // Center the entire graph
  const totalWidth = xOffset - COL_GAP
  const globalShift = -totalWidth / 2

  columnXOffsets.forEach(({ ids, xStart, colWidth }) => {
    // Group by rank within this column
    const rankGroups = {}
    ids.forEach((id) => {
      const r = rank[id]
      if (!rankGroups[r]) rankGroups[r] = []
      rankGroups[r].push(id)
    })

    Object.entries(rankGroups).forEach(([r, idsInRank]) => {
      const groupW = (idsInRank.length - 1) * HORIZ_GAP
      // Center nodes within their column
      const localStart = xStart + (colWidth - groupW) / 2

      idsInRank.forEach((id, i) => {
        compMap[id]._pos = {
          x: globalShift + localStart + i * HORIZ_GAP,
          y: parseInt(r) * VERT_GAP,
        }
      })
    })
  })

  return compMap
}

// ── Main Export ───────────────────────────────────────────────────────────────
export function manifestToFlow(manifest) {
  if (!manifest?.components?.length) return { nodes: [], edges: [] }

  const compMap = assignPositions(manifest.components)
  const idSet = new Set(manifest.components.map((c) => c.id))

  const nodes = manifest.components.map((comp) => ({
    id: comp.id,
    position: comp._pos ?? { x: 0, y: 0 },
    data: {
      label: `${KIND_EMOJI[comp.kind] ?? '🔷'} ${comp.name}`,
      component: comp,
      manifest,
    },
    style: { ...BASE_NODE_STYLE, ...(KIND_STYLE[comp.kind] ?? KIND_STYLE.other) },
    className: comp.is_experimental ? 'experimental-node' : '',
    type: comp.depends_on?.length === 0 ? 'input' : undefined,
  }))

  // Mark leaf nodes (no outgoing edges) as output type
  const hasOutgoing = new Set()
  manifest.components.forEach((c) => {
    ;(c.depends_on || []).filter((d) => idSet.has(d)).forEach((d) => hasOutgoing.add(d))
  })
  nodes.forEach((n) => {
    if (!hasOutgoing.has(n.id) && n.type !== 'input') n.type = 'output'
  })

  const edges = []
  manifest.components.forEach((comp) => {
    ;(comp.depends_on || []).forEach((dep) => {
      if (!idSet.has(dep)) return
      const isImportant = ['attention', 'multi_head_attention', 'residual'].includes(comp.kind)
      edges.push({
        id: `e-${dep}-${comp.id}`,
        source: dep,
        target: comp.id,
        animated: isImportant,
        style: {
          stroke: isImportant ? '#1E3A5F' : '#7A93B0',
          strokeWidth: isImportant ? 2 : 1,
          strokeDasharray: comp.kind === 'masking' ? '5,5' : undefined,
        },
      })
    })
  })

  return { nodes, edges }
}
