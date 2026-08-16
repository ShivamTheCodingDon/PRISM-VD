# PRISM-VD Graph Neural Network Algorithms

This document provides an exhaustive, line-by-line detailed explanation of every GNN algorithm utilized inside the PRISM-VD framework (as defined in `gnn_backbones.py`). Each algorithm receives a graph defined by `(node_x, edge_index, edge_type)` and outputs updated node embeddings for vulnerability classification.

---

## 1. RGAT (Relational Graph Attention Network)
**Class Definition:** `RGATWrapper(nn.Module)`
**Purpose:** Natively handles multiple edge types (Relations) while integrating attention mechanisms to focus on highly dangerous structural connections.

### Detailed Breakdown
- `def __init__(self, in_dim, out_dim, num_edge_types, num_heads, num_layers=1, num_bases=None):`
  Initializes the RGAT wrapper.
- `self.convs = nn.ModuleList([...])`
  Creates a list of PyTorch Geometric `RGATConv` layers.
- `RGATConv(..., num_relations=num_edge_types, num_bases=num_bases)`
  Instantiates the convolution layer. `num_relations` tells the layer how many edge types exist (e.g. AST, DFG, CFG). `num_bases` prevents the model from overfitting by applying a basis-decomposition weight matrix, breaking large correlation matrices into shared smaller matrices (crucial when dealing with 40+ USCP edge types).
- `edge_type = edge_type.clamp(0, self.num_edge_types - 1)`
  Safeguards the incoming edge relation index to ensure it doesn't cause out-of-bounds index errors in the internal weight matrices.
- `x = conv(x, edge_index, edge_type)`
  Passes the graph topology and the specific edge types into the convolution to update node representations.
- `x = F.relu(x)`
  Applies the non-linear Rectified Linear Unit activation function between stacked layers.

---

## 2. GATv2 (Dynamic Graph Attention v2)
**Class Definition:** `GATv2Wrapper(nn.Module)`
**Purpose:** Solves the "static attention limitation" of standard GAT where the attention weights depend primarily on the source node. Extremely expressive for capturing vulnerability semantics in complex pointer flows.

### Detailed Breakdown
- `self.edge_embed = nn.Embedding(num_edge_types, in_dim)`
  Since `GATv2Conv` doesn't natively take categorical relations like `RGAT`, we generate a learnable embedding vector for every single edge type.
- `GATv2Conv(..., edge_dim=in_dim, add_self_loops=True)`
  Tells the GATv2 layer to accept continuous edge attributes of size `in_dim`.
- `edge_attr = self.edge_embed(edge_type).to(x.dtype)`
  At runtime, maps the integer ID of the edge (e.g., CFG edge=2) into its learned continuous vector representation.
- `x = conv(x, edge_index, edge_attr=edge_attr)`
  The dynamic attention is computed combining the source node, destination node, AND the new edge attribute vector.

---

## 3. GIN (Graph Isomorphism Network - GINEConv)
**Class Definition:** `GINWrapper(nn.Module)`
**Purpose:** Designed to be as expressive as the Weisfeiler-Lehman (WL) graph isomorphism test. The `GINEConv` variant incorporates edge features correctly by mathematically injecting them into the aggregation phase.

### Detailed Breakdown
- `mlp = nn.Sequential(nn.Linear(...), nn.BatchNorm1d(...), nn.ReLU(), nn.Linear(...))`
  Creates the Multi-Layer Perceptron (MLP) core that GIN uses to map neighborhoods to new feature spaces.
- `self.convs.append(GINEConv(mlp, edge_dim=cur_in_dim, train_eps=True))`
  Instantiates the GINE convolution. `train_eps=True` allows the center node's weight to be a learnable parameter (epsilon), maximizing discriminative power.
- `edge_attr = embed(edge_type_clamped)`
  Retrieves the embedding for the edge.
- `x = conv(x, edge_index, edge_attr=edge_attr)`
  Inside `GINEConv`, this functionally computes `ReLU(x_j + e_ij)`. This means it adds the edge's semantic meaning to the neighboring node's features *before* summing them up.

---

## 4. SAGE (GraphSAGE)
**Class Definition:** `SAGEWrapper(nn.Module)`
**Purpose:** Built for massive graphs (like those in the BigVul dataset) using memory-efficient neighborhood aggregation.

### Detailed Breakdown
- `SAGEConv(..., aggr='mean')`
  Specifies that GraphSAGE should take the mathematical "mean" (average) of all connected neighborhood nodes rather than summing them, preventing gradient explosion in dense code blocks.
- `enrich.index_add_(0, row, edge_emb)`
  *Workaround implementation:* Since SAGEConv natively lacks edge-feature support, this line gathers all outgoing edge embeddings and adds them into a temporary matrix `enrich`.
- `x_enriched = x + enrich`
  Enriches the source node's feature representation with the sum of the types of edges it emits.
- `x = conv(x_enriched, edge_index)`
  Passes the edge-enriched nodes into the SAGE message passing algorithm.

---

## 5. GGCN (Gated Graph Convolutional Network)
**Class Definition:** `GGCNWrapper(nn.Module)`
**Purpose:** Iterative GRU-based state update graph. Perfect for long-range dependency vulnerabilities like Use-After-Free (UAF), where the `malloc` and `free` operations are distanced across many scopes.

### Detailed Breakdown
- `self.input_proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim))`
  Projects input features directly to the recurrent hidden dimension before message passing begins.
- `self.gru_cell = nn.GRUCell(input_size=out_dim, hidden_size=out_dim)`
  A single Gated Recurrent Unit (GRU) shared across all temporal message passing steps (T-steps). It treats graph depth like a temporal time sequence.
- `for _ in range(self.num_steps):`
  Loops `T` times (usually 5). This dictates how many "hops" a message can travel across the Code Graph.
- `agg = self._aggregate(h, edge_index, edge_type)`
  Custom function that manually aggregates relation-specific messages `(a_r(v) = Σ W_r · h_u)` by mapping node matrices through their specific edge weight tensors `W_r`.
- `h = self.gru_cell(agg, h)`
  The GRU combines the aggregated neighborhood message (`agg`) with the node's previous state (`h`) to form the updated node memory state.

---

## 6. Global Attention Pooling
**Class Definition:** `GlobalAttentionPooling(nn.Module)`
**Purpose:** Graph-level readout. Compresses all `N` nodes into a single flat vector.

### Detailed Breakdown
- `gate_input = self.gate_nn(x)`
  Passes all `[num_nodes, in_features]` through an MLP to map each node to a single logit (score).
- `attention_weights = self.softmax(attention_scores)`
  Normalizes these scores so that they sum up to `1.0`. A vulnerability root-cause node (e.g., a buffer overflow pointer) will ideally receive a high weight (e.g., `0.85`), while safe nodes receive low weights (e.g., `0.01`).
- `aggregated = torch.matmul(attention_weights.unsqueeze(0), x)`
  Performs matrix multiplication to multiply every node by its attention weight and sum them all together, emitting the final Graph Embedding used by the classification head.
