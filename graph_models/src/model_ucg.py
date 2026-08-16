import os
import sys

_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, AutoConfig, AutoTokenizer
from gnn_backbones import build_gnn_layer, SUPPORTED_GNNS, GlobalAttentionPooling
from fusion_strategies import build_fusion, SUPPORTED_FUSIONS


# ── Re-use unchanged helper classes from model_dynamic ───────────────────────

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm    = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, attn_w = self.multihead_attn(
            query, key, value, key_padding_mask=key_padding_mask)
        out = self.norm(query + self.dropout(attn_out))
        return out, attn_w


class MCDropoutClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 dropout_prob=0.3, use_mc_dropout=False):
        super().__init__()
        self.use_mc_dropout = use_mc_dropout
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.ReLU(), nn.Dropout(p=dropout_prob)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_mc_dropout:
            for m in self.modules():
                if isinstance(m, nn.Dropout):
                    m.train()
        return self.net(x)


class FocalWCE_Loss(nn.Module):
    """
    Flexible loss supporting three clean modes:

      'focal_only'  — Standard Focal Loss (Lin et al. 2017).
                      Uses alpha for class-balance, gamma for hard-example mining.
                      pos_weight and label_smoothing are IGNORED to keep
                      the modulating factor pt mathematically correct.

      'wbce_only'   — Weighted BCE with optional label smoothing.
                      Uses pos_weight for class-balance.
                      focal gamma is IGNORED (no focal modulation).

      'combined'    — Legacy mode (focal + pos_weight + label_smoothing all active).
                      ⚠ NOT recommended: pos_weight distorts pt = exp(-bce),
                      making the focal modulating factor incorrect.

    Default mode is 'focal_only' for correct out-of-the-box behavior.
    """

    VALID_MODES = ('focal_only', 'wbce_only', 'combined')

    def __init__(self, pos_weight_val=1.0, alpha=0.25, gamma=2.0,
                 reduction='mean', label_smoothing=0.0, mode='focal_only'):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got '{mode}'")
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.pos_weight_val = pos_weight_val
        self.label_smoothing = label_smoothing
        self.mode = mode

    def forward(self, inputs, targets):
        if self.mode == 'focal_only':
            # ── Correct Focal Loss ────────────────────────────────────
            # Compute pt from UNWEIGHTED BCE so (1-pt)^gamma is correct.
            # Alpha handles class balance instead of pos_weight.
            bce = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none')           # unweighted
            pt    = torch.exp(-bce)                          # true probability
            # Per-class alpha: alpha for positives, (1-alpha) for negatives
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            focal = alpha_t * (1.0 - pt) ** self.gamma * bce
            return focal.mean() if self.reduction == 'mean' else focal.sum()

        elif self.mode == 'wbce_only':
            # ── Weighted BCE (no focal modulation) ────────────────────
            if self.label_smoothing > 0:
                targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            pw = torch.tensor([self.pos_weight_val], device=inputs.device)
            bce = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pw, reduction='none')
            return bce.mean() if self.reduction == 'mean' else bce.sum()

        else:  # 'combined' — legacy (not recommended)
            if self.label_smoothing > 0:
                targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            pw = torch.tensor([self.pos_weight_val], device=inputs.device)
            bce = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=pw, reduction='none')
            pt    = torch.exp(-bce)
            focal = self.alpha * (1.0 - pt) ** self.gamma * bce
            return focal.mean() if self.reduction == 'mean' else focal.sum()



# =============================================================================
# UCG Model — 4-view architecture
# =============================================================================

class UCG_PRISM-VD_VD(nn.Module):
    """
    4-view vulnerability detection model:
        View 1: attended text (CodeBERT + cross-attention)
        View 2: CFG  — Control Flow Graph
        View 3: DFG  — Data Flow Graph
        View 4: UCG  — Unified Causal Graph (merged + deduplicated)
    """

    def __init__(
        self,
        model_name,
        embed_dim      = 128,
        num_edge_types = 41,
        num_heads      = 4,
        hidden_dims    = [512, 128],
        dropout        = 0.3,
        pos_weight_val = 1.2,
        focal_alpha    = 0.25,
        focal_gamma    = 2.0,
        max_edges      = 15000,
        use_mc_dropout = False,
        gnn_type       = 'rgat',
        fusion_type    = 'concat',
        pool_type      = 'mean',
        num_layers     = 1,
        use_custom_vocab = False,
        use_raw_768    = False,
        split_gpus     = False,
        num_bases      = None,
    ):
        super().__init__()

        # ── CodeBERT (text branch — trainable) ───────────────────────────────
        self.config   = AutoConfig.from_pretrained(model_name)
        self.config.use_cache = False
        self.codebert = AutoModel.from_pretrained(model_name, config=self.config)
        
        if hasattr(self.codebert, 'gradient_checkpointing_enable'):
            self.codebert.gradient_checkpointing_enable()

        # ── Frozen CodeBERT (node encoder — static features) ─────────────────
        self.frozen_codebert = AutoModel.from_pretrained(model_name)
        for p in self.frozen_codebert.parameters():
            p.requires_grad_(False)
        self.frozen_codebert.eval()

        if use_custom_vocab:
            import os
            from transformers import RobertaTokenizer
            tok = RobertaTokenizer.from_pretrained(model_name)
            vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "final_c_cpp_vocab.txt")
            if os.path.exists(vocab_path):
                with open(vocab_path, "r", encoding="utf-8") as f:
                    new_tokens = [line.strip() for line in f if line.strip()]
                tok.add_tokens(new_tokens)
                
            self.codebert.resize_token_embeddings(len(tok))
            self.frozen_codebert.resize_token_embeddings(len(tok))
            
            # Unfreeze the embedding layer for the frozen codebert so it can learn the new tokens
            self.frozen_codebert.embeddings.word_embeddings.weight.requires_grad = True

        roberta_dim    = self.config.hidden_size   # 768
        self.embed_dim = embed_dim

        # ── Node Feature Projection Configuration ─────────────────────────────
        self.use_roles = False
        self.role_embed = None
        
        if use_raw_768:
            # Bypass projection: Pass full 768-dim CodeBERT features directly to GNN
            self.embed_dim = roberta_dim
            gnn_input_dim = roberta_dim
            actual_proj_dim = roberta_dim
            self.node_proj = nn.Identity()
        else:
            # Squeeze to embed_dim smoothly
            gnn_input_dim = roberta_dim
            actual_proj_dim = embed_dim
            self.node_proj = nn.Sequential(
                nn.Linear(gnn_input_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, actual_proj_dim),
                nn.LayerNorm(actual_proj_dim),
                nn.GELU()
            )
        
        # Ensure the local embed_dim matches self.embed_dim which could have been updated
        embed_dim = self.embed_dim

        # ── 3 GNN branches (was 6 in the original) ───────────────────────────
        self.gnn_type = gnn_type
        self.split_gpus = split_gpus

        def _make(name):
            layer = build_gnn_layer(gnn_type, actual_proj_dim, self.embed_dim,
                                    num_edge_types, num_heads, num_layers,
                                    num_bases=num_bases)
            layer.__view_name__ = name
            return layer

        self.gnn_cfg = _make('cfg')    # Control Flow
        self.gnn_dfg = _make('dfg')    # Data Flow
        self.gnn_ucg = _make('ucg')    # Unified Causal Graph

        self.pool_type = pool_type
        if pool_type == 'attention':
            self.pool_cfg = GlobalAttentionPooling(embed_dim)
            self.pool_dfg = GlobalAttentionPooling(embed_dim)
            self.pool_ucg = GlobalAttentionPooling(embed_dim)
        else:
            self.pool_cfg = self.pool_dfg = self.pool_ucg = None

        # ── Text projection + cross-attention ─────────────────────────────────
        self.proj_text  = nn.Linear(roberta_dim, embed_dim)
        self.cross_attn = CrossAttention(embed_dim, num_heads, dropout)

        # ── Scale-alignment LayerNorms (replaces broken F.normalize) ──────────
        # Graph embeddings have ~700x smaller magnitude than text.
        # L2-normalize amplifies noise; LayerNorm learns a proper affine scale.
        self.graph_norm_cfg = nn.LayerNorm(embed_dim)
        self.graph_norm_dfg = nn.LayerNorm(embed_dim)
        self.graph_norm_ucg = nn.LayerNorm(embed_dim)
        self.text_norm      = nn.LayerNorm(embed_dim)

        # ── 4-view fusion ──────────────────────────────────────────────────────
        # Views: attended_text + CFG + DFG + UCG = 4
        NUM_VIEWS     = 4
        self.fusion   = build_fusion(fusion_type, embed_dim, NUM_VIEWS,
                                     dropout=dropout, num_heads=num_heads)
        fused_dim     = self.fusion.output_dim
        self.classifier = MCDropoutClassifier(
            fused_dim, hidden_dims, 1, dropout, use_mc_dropout)
        
        # Auxiliary classifier for UCG view only (CFG/DFG don't get aux supervision)
        self.classifier_ucg = MCDropoutClassifier(
            embed_dim, hidden_dims, 1, dropout, use_mc_dropout)
        self.focal_loss = FocalWCE_Loss(
            pos_weight_val=pos_weight_val, alpha=focal_alpha, gamma=focal_gamma)
        self.max_edges  = max_edges

        self.tokenizer  = AutoTokenizer.from_pretrained(model_name)

        import logging
        logging.getLogger(__name__).info(
            f"UCG_PRISM-VD_VD | GNN:[{gnn_type.upper()}] | "
            f"Fusion:[{fusion_type.upper()}] | fused_dim={fused_dim} | "
            f"embed_dim={embed_dim} | gnn_input_dim={gnn_input_dim} | num_edge_types={num_edge_types} | "
            f"heads={num_heads} | views=4 (Text+CFG+DFG+UCG) | layers={num_layers} | "
            f"use_roles={self.use_roles}"
        )

    # ─────────────────────────────────────────────────────────────────────────

    def to(self, *args, **kwargs):
        """Override to() to support Model Parallelism across 2 GPUs."""
        super().to(*args, **kwargs)
        if self.split_gpus and torch.cuda.device_count() >= 2:
            self.dev0 = torch.device('cuda:0')
            self.dev1 = torch.device('cuda:1')
            
            # Text Processing on GPU 0
            self.codebert.to(self.dev0)
            self.frozen_codebert.to(self.dev0)
            self.proj_text.to(self.dev0)
            
            # Graph Processing & Fusion on GPU 1
            self.node_proj.to(self.dev1)
            self.gnn_cfg.to(self.dev1)
            self.gnn_dfg.to(self.dev1)
            self.gnn_ucg.to(self.dev1)
            if hasattr(self, 'pool_cfg') and self.pool_cfg is not None:
                self.pool_cfg.to(self.dev1)
                self.pool_dfg.to(self.dev1)
                self.pool_ucg.to(self.dev1)
            self.cross_attn.to(self.dev1)
            self.fusion.to(self.dev1)
            self.classifier.to(self.dev1)
            self.classifier_ucg.to(self.dev1)
            if self.role_embed is not None:
                self.role_embed.to(self.dev1)
        else:
            self.split_gpus = False
        return self

    def _process_graph(self, gnn_layer, pool_layer, node_x, edge_index, edge_type):
        """Process a single graph view; return (pooled_embed, is_empty).
        
        Uses a residual skip-connection so the GNN only needs to learn
        the structural delta on top of rich CodeBERT node features.
        """
        num_edges = edge_index.size(1)

        if num_edges > self.max_edges:
            edge_index = edge_index[:, :self.max_edges]
            edge_type  = edge_type[:self.max_edges]

        if edge_index.size(1) == 0:
            return torch.zeros((1, self.embed_dim), device=node_x.device), True

        def _fwd(nx, ei, et):
            gnn_out = gnn_layer(nx, ei, et)
            return F.relu(gnn_out + nx)  # Residual: preserve CodeBERT node features

        out = checkpoint(_fwd, node_x, edge_index, edge_type, use_reentrant=False)
        if self.pool_type == 'attention' and pool_layer is not None:
            return pool_layer(out), False
        else:
            return out.mean(dim=0, keepdim=True), False

    def _encode_nodes_dynamic(self, batch_node_labels, device):
        """Encode node labels on-the-fly with CodeBERT (chunked for VRAM)."""
        flat   = [lbl for sample in batch_node_labels for lbl in sample]
        unique = list(set(flat))
        lbl2i  = {l: i for i, l in enumerate(unique)}

        if not unique:
            return [torch.zeros((len(s), 768), device=device)
                    for s in batch_node_labels]

        chunk_size  = 256
        all_embeds  = []
        
        # Force eval mode for frozen branch
        self.frozen_codebert.eval()
        
        # Detect actual device of frozen_codebert (may be on a different GPU)
        frozen_dev = next(self.frozen_codebert.parameters()).device
        
        with torch.no_grad():
            for i in range(0, len(unique), chunk_size):
                chunk  = unique[i: i + chunk_size]
                inp    = self.tokenizer(
                    chunk, return_tensors='pt', padding=True,
                    truncation=True, max_length=64).to(frozen_dev)
                out = self.frozen_codebert(
                    inp['input_ids'],
                    attention_mask=inp['attention_mask'])[0]
                all_embeds.append(out[:, 0, :].to(device))   # CLS token → main device

        unique_embeds = torch.cat(all_embeds, dim=0)

        batch_embeds = []
        for sample_labels in batch_node_labels:
            idx = [lbl2i[l] for l in sample_labels]
            batch_embeds.append(unique_embeds[idx])
        return batch_embeds

    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, input_ids, config_data, return_features=False):
        dev0 = self.dev0 if self.split_gpus else input_ids.device
        dev1 = self.dev1 if self.split_gpus else input_ids.device

        input_ids = input_ids.to(dev0)

        # ── Textual branch ────────────────────────────────────────────────────
        text_out  = self.codebert(input_ids,
                                  attention_mask=input_ids.ne(1))[0]
        text_feat = self.proj_text(text_out)    # [B, seq, embed_dim]
        batch_size = input_ids.size(0)

        pooled = {'cfg': [], 'dfg': [], 'ucg': []}
        padding_masks = []     # [B, 3] — True = graph is empty (ignore)

        # ── Node embeddings ───────────────────────────────────────────────────
        if ('batch_node_features' in config_data
                and config_data['batch_node_features'][0].numel() > 0):
            batch_node_embeds = [
                f.to(dev0)
                for f in config_data['batch_node_features']
            ]
        else:
            batch_node_embeds = self._encode_nodes_dynamic(
                config_data['batch_node_labels'], dev0)

        # ── Per-sample GNN forward pass ───────────────────────────────────────
        for b in range(batch_size):
            sem_x  = batch_node_embeds[b].to(dev1)
            
            if self.use_roles and self.role_embed is not None:
                role_x = self.role_embed(config_data['batch_node_role_ids'][b].to(dev1))
                combined = torch.cat([sem_x, role_x], dim=-1)
            else:
                combined = sem_x

            node_x = self.node_proj(combined)

            cfg_ei = config_data['cfg_edges'][b].to(dev1)
            cfg_et = config_data['cfg_edge_types'][b].to(dev1)
            dfg_ei = config_data['dfg_edges'][b].to(dev1)
            dfg_et = config_data['dfg_edge_types'][b].to(dev1)
            ucg_ei = config_data['ucg_edges'][b].to(dev1)
            ucg_et = config_data['ucg_edge_types'][b].to(dev1)

            f_cfg, s_cfg = self._process_graph(self.gnn_cfg, self.pool_cfg, node_x, cfg_ei, cfg_et)
            f_dfg, s_dfg = self._process_graph(self.gnn_dfg, self.pool_dfg, node_x, dfg_ei, dfg_et)
            f_ucg, s_ucg = self._process_graph(self.gnn_ucg, self.pool_ucg, node_x, ucg_ei, ucg_et)

            pooled['cfg'].append(f_cfg)
            pooled['dfg'].append(f_dfg)
            pooled['ucg'].append(f_ucg)
            padding_masks.append([s_cfg, s_dfg, s_ucg])

        # ── Stack → [B, 1, embed_dim] ─────────────────────────────────────────
        c_feats = torch.stack(pooled['cfg'])    # [B, 1, D]
        d_feats = torch.stack(pooled['dfg'])
        u_feats = torch.stack(pooled['ucg'])

        # graph_memory for cross-attention: [B, 3, D]
        graph_memory = torch.cat([c_feats, d_feats, u_feats], dim=1)

        key_padding_mask = torch.tensor(
            padding_masks, dtype=torch.bool, device=dev1)  # [B, 3]

        # If all 3 graphs empty for a sample, unmask CFG to avoid NaN
        all_empty = key_padding_mask.all(dim=1)
        if all_empty.any():
            key_padding_mask[all_empty, 0] = False

        # ── Cross-attention: text queries graph memory ─────────────────────────
        attended, attn_w = self.cross_attn(
            query            = text_feat[:, :1, :].to(dev1),   # [B, 1, D]
            key              = graph_memory,
            value            = graph_memory,
            key_padding_mask = key_padding_mask,
        )
        attended = attended.squeeze(1)                # [B, D]

        # ── 4-view fusion ──────────────────────────────────────────────────────
        # Scale alignment: graph branches (CFG/DFG/UCG) have ~700x smaller
        # magnitude than text (CodeBERT). Learnable LayerNorm brings all views
        # to comparable scale while preserving gradient flow. This replaces
        # the broken F.normalize which amplified noise to unit scale.
        c_norm = self.graph_norm_cfg(c_feats.squeeze(1))
        d_norm = self.graph_norm_dfg(d_feats.squeeze(1))
        u_norm = self.graph_norm_ucg(u_feats.squeeze(1))
        attended = self.text_norm(attended)
        views = [
            attended,  # View 1: Text (cross-attended + LayerNorm)
            c_norm,    # View 2: CFG  (LayerNorm)
            d_norm,    # View 3: DFG  (LayerNorm)
            u_norm,    # View 4: UCG  (LayerNorm)
        ]
        fused  = self.fusion(views)
        logits = self.classifier(fused)
        
        # Only UCG gets auxiliary supervision
        logits_ucg = self.classifier_ucg(views[3])

        # ── Transfer outputs back to primary device for loss computation ───
        logits = logits.to(dev0)
        logits_ucg = logits_ucg.to(dev0)
        attn_w = attn_w.to(dev0) if attn_w is not None else attn_w

        if return_features:
            views = [v.to(dev0) for v in views]
            fused = fused.to(dev0)
            return logits, logits_ucg, attn_w, views, fused
        return logits, logits_ucg, attn_w
