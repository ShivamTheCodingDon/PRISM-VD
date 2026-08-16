"""
Multi-View Fusion Strategies for PRISM-VD-Enhanced
===================================================
CLI: --fusion <strategy>

Strategies (ordered from simplest to most powerful):

  concat      — Flat concatenation [baseline, original]
  weighted    — Learnable per-view scalar weights + concat
  gated       — Dynamic Gate Fusion: each view has a sigmoid gate
                (Li et al., 2024 - Multi-View Code Representation)
  attention   — Transformer-style self-attention across views
                (Wang et al., EMNLP 2022 - MVD)
  bilinear    — Bilinear pooling: captures cross-view interactions
                (Kim et al., ICLR 2017 - MLB)
  moe         — Mixture of Experts: sparse gating, top-K expert selection
                (Shazeer et al., ICLR 2017; used in CodeT5+ 2023)
  highway     — Highway-Gated Fusion with residual carry
                (Srivastava et al., NeurIPS 2015; proven for code tasks)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Supported fusion strategies
SUPPORTED_FUSIONS = ["concat", "weighted", "gated", "attention", "bilinear", "moe", "highway"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Flat Concatenation [baseline]
# ─────────────────────────────────────────────────────────────────────────────
class ConcatFusion(nn.Module):
    """
    Original baseline: just concatenate all views.
    Output dim = embed_dim * num_views
    No learnable parameters — fast but treats all views equally.
    """
    def __init__(self, embed_dim, num_views, **kwargs):
        super().__init__()
        self.output_dim = embed_dim * num_views

    def forward(self, views):
        # views: list of [batch, embed_dim] tensors
        return torch.cat(views, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Learnable Weighted Sum + Concat
# ─────────────────────────────────────────────────────────────────────────────
class WeightedFusion(nn.Module):
    """
    Learns a global scalar weight per view (not sample-adaptive).
    Applies softmax so weights sum to 1.
    Output dim = embed_dim * num_views  (weighted views then concatenated)

    Interpretation: If DFG weight → 0.40 and CFG weight → 0.08,
    model found data-flow more informative than control-flow globally.
    """
    def __init__(self, embed_dim, num_views, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.output_dim = embed_dim * num_views
        # Learnable log-weights (init=0 → after softmax = uniform)
        self.log_weights = nn.Parameter(torch.zeros(num_views))

    def forward(self, views):
        # views: list of [batch, embed_dim]
        weights = F.softmax(self.log_weights, dim=0)  # [num_views]
        weighted = [views[i] * weights[i] for i in range(self.num_views)]
        return torch.cat(weighted, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Gate Fusion 
# ─────────────────────────────────────────────────────────────────────────────
class GatedFusion(nn.Module):
    """
    Each view has its own linear gate that computes a per-sample sigmoid mask.
    Gate is input-dependent: model learns WHEN to use each view.

    For vulnerability detection:
    - For a sample with no DFG edges → DFG gate ≈ 0 (suppressed)
    - For a sample with rich memory access patterns → DFG gate ≈ 1 (amplified)

    Output dim = embed_dim * num_views
    """
    def __init__(self, embed_dim, num_views, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.output_dim = embed_dim * num_views
        # One gate per view — input is the view itself
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
            for _ in range(num_views)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_views)])

    def forward(self, views):
        gated = []
        for i, v in enumerate(views):
            gate = self.gates[i](v)              # [batch, embed_dim] in [0,1]
            gated_v = self.norms[i](gate * v)   # element-wise gating + norm
            gated.append(gated_v)
        return torch.cat(gated, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Transformer Self-Attention Fusion 
# ─────────────────────────────────────────────────────────────────────────────
class AttentionFusion(nn.Module):
    """
    Treats each view as a "token" in a Transformer sequence.
    Self-attention lets views attend to each other (e.g., CFG ↔ DFG cross-view info).
    Output: mean-pooled attended views + concat for dimensionality control.

    
    Output dim = embed_dim * num_views
    """
    def __init__(self, embed_dim, num_views, dropout=0.1, num_heads=4, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.output_dim = embed_dim * num_views
        # Positional encoding for each view (learnable view-type embedding)
        self.view_pos_embed = nn.Embedding(num_views, embed_dim)
        # Single Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout, batch_first=True, norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, views):
        # Stack into [batch, num_views, embed_dim]
        x = torch.stack(views, dim=1)
        # Add learnable view-type positional embedding
        positions = torch.arange(self.num_views, device=x.device)
        x = x + self.view_pos_embed(positions).unsqueeze(0)  # broadcast batch
        # Self-attention across views
        x = self.encoder(x)  # [batch, num_views, embed_dim]
        # Output: flatten attended views
        return x.reshape(x.size(0), -1)  # [batch, embed_dim * num_views]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bilinear Pooling 
# ─────────────────────────────────────────────────────────────────────────────
class BilinearFusion(nn.Module):
    """
    Captures pairwise interactions between views using Hadamard product.
    Significantly more expressive than concat for cross-view relationships.
    E.g., "CFG × DFG" captures the interaction between control and data flow.

    Uses compact bilinear pooling (MLB-style) to keep dimensions manageable.
    Output dim = embed_dim * num_views (same as concat for fair comparison)
    Reference: Kim et al., ICLR 2017 - Hadamard Product for Low-rank Bilinear Pooling
    """
    def __init__(self, embed_dim, num_views, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.output_dim = embed_dim * num_views

        # Project all views to a shared space for bilinear interaction
        self.proj = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_views)
        ])
        # Final projection to maintain output dimensionality
        self.out_proj = nn.Linear(embed_dim, embed_dim * num_views)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, views):
        # Project each view
        projected = [F.relu(self.proj[i](views[i])) for i in range(self.num_views)]

        # Compact bilinear: element-wise product of all pairs → accumulate
        # Uses sum over pairs (symmetric Hadamard sum) to avoid O(V^2) explosion
        interaction = torch.zeros_like(projected[0])
        count = 0
        for i in range(self.num_views):
            for j in range(i + 1, self.num_views):
                interaction = interaction + projected[i] * projected[j]  # Hadamard
                count += 1
        if count > 0:
            interaction = interaction / count  # Normalize

        interaction = self.norm(interaction)
        # Expand interaction to match output dim
        expanded = self.out_proj(interaction)  # [batch, embed_dim * num_views]
        return expanded


# ─────────────────────────────────────────────────────────────────────────────
# 6. Mixture of Experts Fusion 
# ─────────────────────────────────────────────────────────────────────────────
class MoEFusion(nn.Module):
    """
    Sparse Mixture of Experts: N expert networks, router selects top-K per sample.
    Only top-K experts are activated — efficient + diverse representations.

    FIX — Dense → Sparse Dispatch:
    ────────────────────────────────
    OLD (Dense):  expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)
                  Ran ALL experts on ALL samples, then discarded unused outputs.
                  Wasted O(num_experts - top_k) FLOPs per sample every forward pass.

    NEW (Sparse): For each expert, a boolean mask selects only the samples routed
                  to it. The expert runs ONLY on that subset; results are scattered
                  back into the full [B, D] output tensor. Experts receiving zero
                  dispatches are skipped entirely.
                  Compute savings: ~(1 - top_k / num_experts) × expert FLOPs.

    Auxiliary Load-Balancing Loss  
    ──────────────────────────────
    self.aux_loss is populated every forward pass. The training loop can optionally
    add it to prevent expert collapse:
        loss = task_loss + 1e-2 * model.fusion.aux_loss

    Output dim = embed_dim * num_views
    """
    def __init__(self, embed_dim, num_views, num_experts=4, top_k=2, dropout=0.1, **kwargs):
        super().__init__()
        self.num_views   = num_views
        self.output_dim  = embed_dim * num_views
        self.num_experts = num_experts
        self.top_k       = min(top_k, num_experts)

        input_dim = embed_dim * num_views
        # Router: learns which experts to activate given the concatenated views
        self.router = nn.Linear(input_dim, num_experts)

        # Expert feed-forward networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(input_dim * 2, input_dim),
                nn.LayerNorm(input_dim)
            ) for _ in range(num_experts)
        ])
        self.residual_norm = nn.LayerNorm(input_dim)

        # Auxiliary load-balancing loss (populated in forward, read by training loop)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, views):
        # ── 1. Concatenate all views for routing ──────────────────────────────
        x = torch.cat(views, dim=-1)          # [B, D]  where D = embed_dim * num_views
        B, D = x.shape

        # ── 2. Router: compute probabilities and select top-K experts ─────────
        router_logits  = self.router(x)                    # [B, num_experts]
        router_probs   = F.softmax(router_logits, dim=-1)  # [B, num_experts]
        routing_weights, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1               # [B, top_k] each
        )
        # Renormalize top-K weights so they sum to 1 per sample
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)

        # ── 3. Auxiliary Load-Balancing Loss (Shazeer et al. §3) ─────────────
        # aux_loss = num_experts × Σ_e( f_e × P_e )
        #   f_e = fraction of batch tokens dispatched to expert e  (no grad)
        #   P_e = mean router probability for expert e             (differentiable)
        with torch.no_grad():
            dispatch_mask = torch.zeros(B, self.num_experts,
                                        device=x.device, dtype=x.dtype)
            dispatch_mask.scatter_(1, selected_experts, 1.0)
            f_e = dispatch_mask.mean(dim=0)                # [num_experts]
        P_e = router_probs.mean(dim=0)                     # [num_experts] — has grad
        self.aux_loss = self.num_experts * (f_e * P_e).sum()

        # ── 4. TRUE SPARSE DISPATCH ───────────────────────────────────────────
        # For each expert, find the samples routed to it via a boolean mask,
        # run the expert ONLY on those n <= B samples, then scatter weighted
        # results back into the full [B, D] output tensor.
        output = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        for e_idx, expert in enumerate(self.experts):
            # Boolean mask: which samples selected this expert?
            expert_mask = (selected_experts == e_idx).any(dim=-1)  # [B] bool
            if not expert_mask.any():
                continue  # Expert unused this batch — zero compute, skip entirely

            # Run expert ONLY on its routed subset
            x_subset   = x[expert_mask]            # [n, D]  n <= B
            expert_out = expert(x_subset)          # [n, D]

            # Retrieve this expert's routing weight for each routed sample
            sel_subset   = selected_experts[expert_mask]  # [n, top_k]
            wt_subset    = routing_weights[expert_mask]   # [n, top_k]
            slot_match   = (sel_subset == e_idx).float()  # [n, top_k] — one-hot slot
            per_sample_w = (wt_subset * slot_match).sum(dim=-1, keepdim=True)  # [n, 1]

            # Weighted accumulation back into full-batch output tensor
            output[expert_mask] = output[expert_mask] + per_sample_w * expert_out

        # ── 5. Residual: combine MoE output with raw concatenated input ───────
        return self.residual_norm(x + output)

# ─────────────────────────────────────────────────────────────────────────────
class HighwayFusion(nn.Module):
    """
    Highway networks with "carry" gate (C) and "transform" gate (T).
    Output = T(x) ⊙ H(x) + (1 - T(x)) ⊙ x

    The carry gate allows gradients to flow directly without attenuation —
    important for deep code graphs where signal can vanish.

    Works well for vulnerability detection because:
    - Simple patterns (trivial buffer overflow): high carry → no transformation needed
    - Complex patterns (logic + data combo): low carry → full transformation applied

    Output dim = embed_dim * num_views
    """
    def __init__(self, embed_dim, num_views, num_layers=2, dropout=0.1, **kwargs):
        super().__init__()
        self.num_views = num_views
        self.output_dim = embed_dim * num_views
        dim = embed_dim * num_views

        # Per-view gating before highway
        self.view_gates = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_views)
        ])

        # Highway layers
        self.highway_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.highway_layers.append(nn.ModuleDict({
                'H': nn.Linear(dim, dim),        # Transform
                'T': nn.Linear(dim, dim),        # Gate (transform gate)
                'norm': nn.LayerNorm(dim),
                'drop': nn.Dropout(dropout),
            }))

        # Init gate biases negative so early training passes most info through
        for layer in self.highway_layers:
            nn.init.constant_(layer['T'].bias, -1.0)

    def forward(self, views):
        # Per-view sigmoid gating first
        gated_views = [
            torch.sigmoid(self.view_gates[i](views[i])) * views[i]
            for i in range(self.num_views)
        ]
        x = torch.cat(gated_views, dim=-1)  # [batch, embed_dim * num_views]

        # Highway layers
        for layer in self.highway_layers:
            h = F.gelu(layer['H'](x))
            t = torch.sigmoid(layer['T'](x))  # Transform gate
            x_new = t * h + (1 - t) * x       # Highway equation
            x = layer['norm'](layer['drop'](x_new))

        return x


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def build_fusion(fusion_type, embed_dim, num_views, dropout=0.1, num_heads=4):
    """
    Factory function — returns the selected fusion module.

    Args:
        fusion_type:  str from SUPPORTED_FUSIONS
        embed_dim:    dimension of each individual view embedding
        num_views:    number of views being fused (e.g., 7)
        dropout:      dropout probability
        num_heads:    for attention-based fusions

    Returns:
        nn.Module with .output_dim attribute
    """
    fusion_type = fusion_type.lower()
    kwargs = dict(embed_dim=embed_dim, num_views=num_views,
                  dropout=dropout, num_heads=num_heads)

    if fusion_type == "concat":
        return ConcatFusion(**kwargs)
    elif fusion_type == "weighted":
        return WeightedFusion(**kwargs)
    elif fusion_type == "gated":
        return GatedFusion(**kwargs)
    elif fusion_type == "attention":
        return AttentionFusion(**kwargs)
    elif fusion_type == "bilinear":
        return BilinearFusion(**kwargs)
    elif fusion_type == "moe":
        return MoEFusion(**kwargs)
    elif fusion_type == "highway":
        return HighwayFusion(**kwargs)
    else:
        raise ValueError(
            f"Unknown fusion type: '{fusion_type}'. "
            f"Choose from: {SUPPORTED_FUSIONS}"
        )