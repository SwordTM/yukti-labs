import { PARAM_DEFAULTS } from '../hyperparameters'

// Maps sandbox node IDs to ComponentManifest fields
const NODE_META = {
  '1':  { id: 'input_tokens',        kind: 'input_embedding',     name: 'Input Tokens',          depends_on: [] },
  '2':  { id: 'input_embedding',     kind: 'input_embedding',     name: 'Input Embedding',        depends_on: ['input_tokens'] },
  '3':  { id: 'positional_encoding', kind: 'positional_encoding', name: 'Positional Encoding',    depends_on: ['input_embedding'] },
  '4':  { id: 'encoder',             kind: 'multi_head_attention', name: 'Encoder (×6)',           depends_on: ['input_embedding', 'positional_encoding'] },
  '5':  { id: 'multi_head_attention',kind: 'multi_head_attention', name: 'Multi-Head Attention',   depends_on: ['encoder'] },
  '6':  { id: 'feedforward',         kind: 'feedforward',         name: 'Feed Forward',           depends_on: ['multi_head_attention'] },
  '7':  { id: 'decoder',             kind: 'multi_head_attention', name: 'Decoder (×6)',           depends_on: ['feedforward'] },
  '8':  { id: 'masked_attention',    kind: 'masking',             name: 'Masked Attention',       depends_on: ['decoder'] },
  '9':  { id: 'cross_attention',     kind: 'attention',           name: 'Cross Attention',        depends_on: ['masked_attention', 'encoder'] },
  '10': { id: 'output_head',         kind: 'output_head',         name: 'Linear + Softmax',       depends_on: ['cross_attention'] },
}

// Per-node hyperparameter descriptions (symbolic → meaning with current value)
function nodeHyperparams(nodeId, params) {
  if (!params) return {}
  switch (nodeId) {
    case '1':  return { max_seq_len: `max sequence length = ${params.max_seq_len}` }
    case '2':  return { d_model: `model dimension = ${params.d_model}`, vocab_size: `vocab size = ${params.vocab_size}` }
    case '3':  return { max_seq_len: `max sequence length = ${params.max_seq_len}`, dropout: `dropout = ${params.dropout}` }
    case '4':  return { num_layers: `encoder layers = ${params.num_layers}`, dropout: `dropout = ${params.dropout}` }
    case '5':  return { h: `attention heads = ${params.num_heads}`, d_k: `key dim = ${Math.floor(params.d_model / params.num_heads)}`, dropout: `dropout = ${params.dropout}` }
    case '6':  return { d_ff: `FFN hidden dim = ${params.d_ff}`, dropout: `dropout = ${params.dropout}`, activation: `activation = ${params.activation}` }
    case '7':  return { num_layers: `decoder layers = ${params.num_layers}`, dropout: `dropout = ${params.dropout}` }
    case '8':  return { h: `attention heads = ${params.num_heads}`, d_k: `key dim = ${Math.floor(params.d_model / params.num_heads)}` }
    case '9':  return { h: `attention heads = ${params.num_heads}`, d_k: `key dim = ${Math.floor(params.d_model / params.num_heads)}` }
    case '10': return { vocab_size: `vocab size = ${params.vocab_size}`, temperature: `temperature = ${params.temperature}` }
    default:   return {}
  }
}

export function buildManifest(hyperparams) {
  const p2 = hyperparams['2'] ?? PARAM_DEFAULTS['2']
  const p5 = hyperparams['5'] ?? PARAM_DEFAULTS['5']
  const p6 = hyperparams['6'] ?? PARAM_DEFAULTS['6']
  const p1 = hyperparams['1'] ?? PARAM_DEFAULTS['1']

  const dModel  = p2.d_model
  const nHeads  = p5.num_heads
  const dK      = Math.floor(dModel / nHeads)
  const dFf     = p6.d_ff
  const seqLen  = p1.max_seq_len

  const components = Object.entries(NODE_META).map(([nodeId, meta]) => ({
    id:               meta.id,
    name:             meta.name,
    kind:             meta.kind,
    description:      `${meta.name} component of the Transformer architecture.`,
    operations:       [],
    depends_on:       meta.depends_on,
    hyperparameters:  nodeHyperparams(nodeId, hyperparams[nodeId] ?? PARAM_DEFAULTS[nodeId]),
    equations:        [],
  }))

  return {
    paper: {
      arxiv_id:  '1706.03762',
      title:     'Attention Is All You Need',
      authors:   ['Ashish Vaswani', 'Noam Shazeer', 'Niki Parmar', 'Jakob Uszkoreit',
                  'Llion Jones', 'Aidan N. Gomez', 'Łukasz Kaiser', 'Illia Polosukhin'],
      abstract:  'The dominant sequence transduction models are based on complex recurrent or convolutional neural networks. We propose the Transformer, based solely on attention mechanisms.',
      published: '2017-06-12',
      pdf_url:   'https://arxiv.org/pdf/1706.03762',
    },
    components,
    tensor_contracts: [
      {
        component_id:  'multi_head_attention',
        input_shapes:  { Q: ['B', 'T', 'd_model'], K: ['B', 'T', 'd_model'], V: ['B', 'T', 'd_model'] },
        output_shapes: { out: ['B', 'T', 'd_model'] },
        dtype: 'float32',
      },
      {
        component_id:  'feedforward',
        input_shapes:  { x: ['B', 'T', 'd_model'] },
        output_shapes: { out: ['B', 'T', 'd_model'] },
        dtype: 'float32',
      },
    ],
    invariants: [
      {
        id: 'residual_add_norm',
        description: 'Each sub-layer output is LayerNorm(x + Sublayer(x)). Residual applied before norm.',
        kind: 'residual_connection',
        affected_components: ['multi_head_attention', 'feedforward', 'masked_attention', 'cross_attention'],
      },
      {
        id: 'causal_mask',
        description: 'Decoder self-attention masks future positions to -inf before softmax.',
        kind: 'causal_mask',
        affected_components: ['masked_attention'],
      },
    ],
    symbol_table: {
      B:       'batch size',
      T:       `sequence length (max ${seqLen})`,
      d_model: `model hidden dimension (${dModel})`,
      h:       `number of attention heads (${nHeads})`,
      d_k:     `key/query dimension per head (${dK})`,
      d_ff:    `feed-forward hidden dimension (${dFf})`,
      V:       `vocabulary size (${p2.vocab_size ?? 37000})`,
    },
  }
}
