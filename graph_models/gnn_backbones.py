import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    RGATConv, GATConv, GATv2Conv, GINConv, GINEConv, SAGEConv, GCNConv
)

# ─────────────────────────────────────────────────────────────────────────────
# Helper wrappers to unify the calling convention across all GNNs.
# All wrappers accept (node_x, edge_index, edge_type) and return node embeddings.
# edge_type is silently ignored by architectures that don't use it.
# ─────────────────────────────────────────────────────────────────────────────

class RGATWrapper(nn.Module):
    """
    RGATConv — Relational Graph Attention Network.
    Natively supports typed edges via num_relations.
    Paper: "Relational Graph Attention Networks" (Busbridge et al., 2019).
    PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGATConv.html
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads, num_layers=1, num_bases=None):
        super().__init__()
        self.num_edge_types = num_edge_types
        self.convs = nn.ModuleList([
            RGATConv(in_dim if i == 0 else out_dim, out_dim, num_relations=num_edge_types,
                     num_bases=num_bases, heads=num_heads, concat=False)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        # Clamp edge_type to valid range for RGAT's internal weight matrix
        edge_type = edge_type.clamp(0, self.num_edge_types - 1)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GATv2Wrapper(nn.Module):
    """
    GATv2Conv — Dynamic Graph Attention v2.
    Fixes the "static attention" limitation of original GAT. More expressive.
    Edge type is encoded as a learnable type embedding added to node features.
    Paper: "How Attentive are Graph Attention Networks?" (Brody et al., ICLR 2022).
    PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GATv2Conv.html
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads, num_layers=1):
        super().__init__()
        # Edge embedding to inject relation type info into attention
        self.edge_embed = nn.Embedding(num_edge_types, in_dim)
        # GATv2Conv with edge_dim enabled
        self.convs = nn.ModuleList([
            GATv2Conv(in_dim if i == 0 else out_dim, out_dim, heads=num_heads, concat=False,
                      edge_dim=in_dim, add_self_loops=True)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        # Clamp edge_type to valid embedding range
        edge_type = edge_type.clamp(0, self.edge_embed.num_embeddings - 1)
        edge_attr = self.edge_embed(edge_type).to(x.dtype)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GATWrapper(nn.Module):
    """
    Classic GATConv — Graph Attention Network.
    Velickovic et al., ICLR 2018.
    Edge type is encoded as a learnable type embedding added to node features.
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads, num_layers=1):
        super().__init__()
        # Edge embedding to inject relation type info
        self.edge_embed = nn.Embedding(num_edge_types, in_dim)
        self.convs = nn.ModuleList([
            GATConv(in_dim if i == 0 else out_dim, out_dim, heads=num_heads, concat=False,
                    edge_dim=in_dim, add_self_loops=True)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        edge_type = edge_type.clamp(0, self.edge_embed.num_embeddings - 1)
        edge_attr = self.edge_embed(edge_type).to(x.dtype)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return x


class GINWrapper(nn.Module):
    """
    GINEConv — Graph Isomorphism Network with Edge Features.
    Extends GIN (Xu et al., ICLR 2019) to correctly incorporate edge features
    using ReLU(x_j + e_ij) in the message aggregation step. This is the
    mathematically correct way to condition messages on specific edge types
    (each edge type strictly influences only its own source→target pair).

    NOTE: Using GINEConv (Hu et al., 2019), NOT GINConv. GINConv has no
    native edge attribute support. The previous index_add_ workaround was
    mathematically incorrect — it mixed all outgoing edge types onto the
    source node before message passing, destroying per-edge conditioning.

    Paper: "Strategies for Pre-training Graph Neural Networks" (Hu et al., 2019).
    PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GINEConv.html
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads=None, num_layers=1):
        super().__init__()
        # Per-layer edge embeddings to correctly match each layer's input dimension
        self.edge_embeds = nn.ModuleList([
            nn.Embedding(num_edge_types, in_dim if i == 0 else out_dim)
            for i in range(num_layers)
        ])
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            cur_in_dim = in_dim if i == 0 else out_dim
            mlp = nn.Sequential(
                nn.Linear(cur_in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            )
            # GINEConv: natively supports edge_attr via ReLU(x_j + e_ij)
            # edge_dim must match the embedding dim for this layer
            self.convs.append(GINEConv(mlp, edge_dim=cur_in_dim, train_eps=True))

    def forward(self, x, edge_index, edge_type):
        for i, (conv, embed) in enumerate(zip(self.convs, self.edge_embeds)):
            edge_type_clamped = edge_type.clamp(0, embed.num_embeddings - 1)
            edge_attr = embed(edge_type_clamped).to(x.dtype)  # (E, layer_in_dim)
            # Correctly pass edge_attr per-edge: GINEConv does ReLU(x_j + e_ij)
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.relu(x)


class SAGEWrapper(nn.Module):
    """
    GraphSAGE — Inductive Representation Learning on Large Graphs.
    Uses mean neighborhood aggregation. Very fast and memory efficient.
    Paper: "Inductive Representation Learning on Large Graphs" (Hamilton et al., NeurIPS 2017).
    PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.SAGEConv.html

    LIMITATION: SAGEConv has no native multi-dim edge attribute support.
    Edge type is injected via an approximate 'node enrichment' workaround:
    all outgoing edge embeddings from a node are summed and added to its
    features before convolution. This is NOT mathematically equivalent to
    per-edge conditioning and will mix edge type signals. Use RGAT, GATv2,
    or GINEConv (GINWrapper) if strict per-edge conditioning is required.
    This backbone is retained as a fast memory-efficient baseline.
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads=None, num_layers=1):
        super().__init__()
        # Per-layer edge embeddings to correctly match each layer's input dimension
        self.edge_embeds = nn.ModuleList([
            nn.Embedding(num_edge_types, in_dim if i == 0 else out_dim)
            for i in range(num_layers)
        ])
        self.convs = nn.ModuleList([
            SAGEConv(in_dim if i == 0 else out_dim, out_dim, aggr='mean')
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        row, col = edge_index
        for i, (conv, embed) in enumerate(zip(self.convs, self.edge_embeds)):
            edge_type_clamped = edge_type.clamp(0, embed.num_embeddings - 1)
            edge_emb = embed(edge_type_clamped).to(x.dtype)
            # Approximate workaround: sum all outgoing edge embeddings onto src node
            enrich = torch.zeros_like(x)
            enrich.index_add_(0, row, edge_emb)
            x_enriched = x + enrich
            x = conv(x_enriched, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.relu(x)


class GCNWrapper(nn.Module):
    """
    GCN — Graph Convolutional Network.
    Classic spectral-based convolution. Simple and fast baseline.
    Paper: "Semi-Supervised Classification with Graph Convolutional Networks"
           (Kipf & Welling, ICLR 2017).
    PyG Docs: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html

    LIMITATION: GCNConv only supports a 1D scalar edge_weight (not multi-dim
    edge attributes). Edge type is injected via an approximate 'node enrichment'
    workaround: all outgoing edge embeddings from a node are summed and added
    to its features before convolution. This is NOT mathematically equivalent
    to per-edge conditioning and will mix edge type signals. Use RGAT, GATv2,
    or GINEConv (GINWrapper) if strict per-edge conditioning is required.
    This backbone is retained as a simple, fast ablation baseline.
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads=None, num_layers=1):
        super().__init__()
        # Per-layer edge embeddings to correctly match each layer's input dimension
        self.edge_embeds = nn.ModuleList([
            nn.Embedding(num_edge_types, in_dim if i == 0 else out_dim)
            for i in range(num_layers)
        ])
        self.convs = nn.ModuleList([
            GCNConv(in_dim if i == 0 else out_dim, out_dim, add_self_loops=True)
            for i in range(num_layers)
        ])

    def forward(self, x, edge_index, edge_type):
        row, col = edge_index
        for i, (conv, embed) in enumerate(zip(self.convs, self.edge_embeds)):
            edge_type_clamped = edge_type.clamp(0, embed.num_embeddings - 1)
            edge_emb = embed(edge_type_clamped).to(x.dtype)
            # Approximate workaround: sum all outgoing edge embeddings onto src node
            enrich = torch.zeros_like(x)
            enrich.index_add_(0, row, edge_emb)
            x_enriched = x + enrich
            x = conv(x_enriched, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
        return F.relu(x)


class GGCNWrapper(nn.Module):
    """
    GGCN — Gated Graph Convolutional Network (Gated GNN / GGNN variant).

    Implements iterative GRU-based message passing over a typed graph.
    At each propagation step t:

      1. Typed message aggregation (per relation r):
             a_r(v) = Σ_{u∈N_r(v)}  W_r · h_u^(t-1)
         where W_r is decomposed via num_bases shared basis matrices to
         prevent overfit on rare edge types (same trick as RGCN/RGAT):
             W_r = Σ_b  coeffs[r, b] · Basis_b

      2. Sum aggregations across all edge types:
             a(v) = Σ_r  a_r(v)

      3. GRU state update (weight-tied across all T steps):
             h_v^(t) = GRU( a(v), h_v^(t-1) )

    After T steps the node hidden states capture T-hop structural context
    with persistent "memory" — ideal for long-range vulnerability patterns
    such as malloc/free distance, taint propagation across many CFG branches,
    or call chains that span multiple scopes.

    API: forward(x, edge_index, edge_type) → (N, out_dim)
    Fully compatible with build_gnn_layer() / UCG_PRISM-VD_VD._process_graph().

    Args:
        in_dim:         Input node feature dimension.
        out_dim:        Hidden / output dimension (GRU hidden size).
        num_edge_types: Number of distinct edge relation types (e.g. 19 from
                        atlas_bridge.EDGE_TYPE_MAP, or 41 for UCG).
        num_heads:      Unused — kept for factory API compatibility.
        num_layers:     Number of GRU message-passing steps T (default: 5).
                        Recommended range: 3–8.
        num_bases:      Number of shared basis matrices for relation-weight
                        decomposition. None → each relation gets its own W_r.
                        Recommended: 4–8 for 19–41 relations.

    Paper: "Gated Graph Sequence Neural Networks" (Li et al., ICLR 2016).
    """
    def __init__(self, in_dim, out_dim, num_edge_types, num_heads=None,
                 num_layers=5, num_bases=None):
        super().__init__()
        self.in_dim         = in_dim
        self.out_dim        = out_dim
        self.num_edge_types = num_edge_types
        self.num_steps      = num_layers    # T propagation steps
        self.num_bases      = num_bases

        # ── Input projection (in_dim → out_dim) ──────────────────────────────
        # Project node features to GRU hidden dimension once, upfront.
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

        # ── Relation weight matrices (with optional basis decomposition) ──────
        # Full parametrisation:  W_r ∈ R^{out_dim × out_dim}  per relation.
        # Basis decomposition:   W_r = Σ_b a_{rb} · B_b  (RGCN-style).
        if num_bases is not None and num_bases < num_edge_types:
            self.bases  = nn.Parameter(torch.empty(num_bases, out_dim, out_dim))
            self.coeffs = nn.Parameter(torch.empty(num_edge_types, num_bases))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.coeffs)
            self._use_bases = True
        else:
            self.rel_weights = nn.Parameter(
                torch.empty(num_edge_types, out_dim, out_dim))
            nn.init.xavier_uniform_(self.rel_weights)
            self._use_bases = False

        # ── GRU cell for state update ─────────────────────────────────────────
        # Single GRU cell shared (weight-tied) across all T steps — matching
        # the original GGNN paper and keeping parameter count low.
        self.gru_cell = nn.GRUCell(input_size=out_dim, hidden_size=out_dim)

        # ── LayerNorm + Dropout on aggregated messages ────────────────────────
        self.msg_norm = nn.LayerNorm(out_dim)
        self.dropout  = nn.Dropout(p=0.1)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _get_rel_weight(self, r: int) -> torch.Tensor:
        """Return the (out_dim, out_dim) transform matrix for relation r."""
        if self._use_bases:
            # W_r = Σ_b coeffs[r, b] * bases[b]   →   (out_dim, out_dim)
            w = (self.coeffs[r].unsqueeze(-1).unsqueeze(-1) * self.bases).sum(0)
        else:
            w = self.rel_weights[r]
        return w  # (out_dim, out_dim)

    def _aggregate(self, h: torch.Tensor,
                   edge_index: torch.Tensor,
                   edge_type: torch.Tensor) -> torch.Tensor:
        """
        One round of typed message aggregation.

        For each relation r:
            messages_r  = h[src_r] @ W_r^T          (E_r, out_dim)
            agg        += scatter_sum(messages_r, dst_r, dim_size=N)

        Returns summed aggregation across all relations: (N, out_dim).
        """
        N   = h.size(0)
        agg = torch.zeros(N, self.out_dim, device=h.device, dtype=h.dtype)
        src, dst = edge_index[0], edge_index[1]
        edge_type_clamped = edge_type.clamp(0, self.num_edge_types - 1)

        for r in range(self.num_edge_types):
            mask = (edge_type_clamped == r)
            if not mask.any():
                continue
            src_r = src[mask]              # source node indices for relation r
            dst_r = dst[mask]              # destination node indices
            h_src = h[src_r]              # (E_r, out_dim)
            W_r   = self._get_rel_weight(r)  # (out_dim, out_dim)
            msgs  = h_src @ W_r.t()       # (E_r, out_dim)
            agg.index_add_(0, dst_r, msgs.to(agg.dtype))

        return agg  # (N, out_dim)

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:          (N, in_dim)  — initial node features
            edge_index: (2, E)       — [src_nodes; dst_nodes]
            edge_type:  (E,)         — integer relation IDs
        Returns:
            h:          (N, out_dim) — node embeddings after T GRU steps
        """
        # Project input to GRU hidden size once
        h = self.input_proj(x)   # (N, out_dim)

        # Iterative GRU message passing — T steps
        for _ in range(self.num_steps):
            agg = self._aggregate(h, edge_index, edge_type)  # (N, out_dim)
            agg = self.msg_norm(agg)
            agg = self.dropout(agg)
            h   = self.gru_cell(agg, h)   # (N, out_dim) — GRU update

        return h  # (N, out_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Registry and Factory
# ─────────────────────────────────────────────────────────────────────────────

GNN_REGISTRY = {
    "rgat":  RGATWrapper,   # Relational GAT — edge-type-native, SOTA for multi-view code graphs
    "gat":   GATWrapper,    # Classic GAT (GATConv) — Velickovic et al., ICLR 2018 (Original baseline)
    "gatv2": GATv2Wrapper,  # Dynamic attention — often outperforms RGAT on small-medium graphs
    "gin":   GINWrapper,    # Most expressive WL-equivalent — good for detecting isomorphic patterns
    "sage":  SAGEWrapper,   # Inductive, memory-efficient — good for large graphs (BigVul)
    "gcn":   GCNWrapper,    # Classic baseline — fast ablation study
    "ggcn":  GGCNWrapper,   # Gated GCN — GRU state update, ideal for long-range vuln patterns
}

SUPPORTED_GNNS = list(GNN_REGISTRY.keys())


def build_gnn_layer(gnn_type: str, in_dim: int, out_dim: int,
                     num_edge_types: int, num_heads: int = 4,
                     num_layers: int = 1, num_bases: int = None) -> nn.Module:
    """
    Factory function to build a GNN layer by name.

    Args:
        gnn_type:       One of 'rgat', 'gat', 'gatv2', 'gin', 'sage', 'gcn', 'ggcn'.
        in_dim:         Input node feature dimension.
        out_dim:        Output node embedding dimension.
        num_edge_types: Number of distinct edge relation types.
        num_heads:      Number of attention heads (used by RGAT and GATv2).
                        Unused by GGCN (GRU-based, no attention heads).
        num_layers:     Number of sequential GNN layers / GRU message-passing
                        steps T (for GGCN, default 5 is recommended).
        num_bases:      Number of basis matrices for RGAT/GGCN relation weight
                        decomposition (regularisation). None = no decomposition.
                        Used by RGAT and GGCN; silently ignored by other GNNs.

    Returns:
        An nn.Module that accepts (node_x, edge_index, edge_type) and
        returns a node embedding tensor of shape (N, out_dim).
    """
    gnn_type = gnn_type.lower().strip()
    if gnn_type not in GNN_REGISTRY:
        raise ValueError(
            f"Unknown GNN type '{gnn_type}'. "
            f"Supported types: {SUPPORTED_GNNS}"
        )
    cls = GNN_REGISTRY[gnn_type]
    if gnn_type in ('rgat', 'ggcn'):
        # Both RGAT and GGCN support num_bases for relation weight decomposition
        return cls(in_dim, out_dim, num_edge_types, num_heads, num_layers, num_bases=num_bases)
    return cls(in_dim, out_dim, num_edge_types, num_heads, num_layers)

class GlobalAttentionPooling(nn.Module):
    """
    Global Attention Pooling mechanism adopted from the original PRISM-VD.
    Computes a weighted sum of all node features based on learned attention scores.
    """
    def __init__(self, in_features, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = in_features // 2
            
        self.gate_nn = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LeakyReLU(negative_slope=0.001),
            nn.Linear(hidden_dim, 1)
        )
        self.softmax = nn.Softmax(dim=0)
        
    def forward(self, x):
        # x shape: [num_nodes, in_features]
        gate_input = self.gate_nn(x)           # [num_nodes, 1]
        attention_scores = gate_input.squeeze(-1) # [num_nodes]
        attention_weights = self.softmax(attention_scores) # [num_nodes]
        
        # Weighted sum: (1, num_nodes) @ (num_nodes, in_features) -> (1, in_features)
        aggregated = torch.matmul(attention_weights.unsqueeze(0), x)
        return aggregated
