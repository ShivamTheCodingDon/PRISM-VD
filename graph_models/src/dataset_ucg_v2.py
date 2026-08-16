import os
import sys
import json
import torch
import numpy as np
import logging
import random
from collections import deque

# ── Allow importing from parent model/ directory ─────────────────────────────
_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from dataset_dynamic import DynamicEnhancedCodeGraphDataset  # noqa: E402

logger = logging.getLogger(__name__)

# =============================================================================
# Default guard cap — prevents the ~9,700 guard edges/sample explosion
# =============================================================================
DEFAULT_MAX_GUARDS_PER_PATH = None

# Vulnerability-relevant roles — everything NOT in this set is noise
VULN_ROLES = {"CALL", "DEREF", "ENTRY", "EXIT", "ASSIGN", "GUARD"}


# =============================================================================
# UCGCodeGraphDatasetV2
# Inherits: _index_samples, all slicing methods (_adaptive_bidirectional_dfs_slice,
#           _vpc_ppr_slice, _cta_rwr_slice, _directional_dfs, etc.),
#           _process_edges, _text_to_token_ids, _clean_code, __len__
# Overrides: __getitem__ only
# =============================================================================

class UCGCodeGraphDatasetV2(DynamicEnhancedCodeGraphDataset):
    """
    Dataset that produces 4 views (Text, CFG, DFG, UCG) for training.
    All graph slicing algorithms are inherited from the parent class unchanged.

    v2 fixes:
        - USCP guard edges capped to max_guards_per_path (default 3)
        - Prevents guard signal from overwhelming path/flow signals
    """

    # Supported context padding modes
    CONTEXT_MODES = ('random', 'hop', 'rwr')

    def __init__(self, *args, max_guards_per_path=DEFAULT_MAX_GUARDS_PER_PATH,
                 filter_other=False, max_paths=None,
                 random_context_pad=True, context_ratio=0.3,
                 context_mode='random', **kwargs):
        self.max_guards_per_path = max_guards_per_path
        self.filter_other = filter_other
        self.max_paths = max_paths
        self.random_context_pad = random_context_pad
        self.context_ratio = context_ratio
        self.context_mode = context_mode if context_mode in self.CONTEXT_MODES else 'random'
        super().__init__(*args, **kwargs)
        logger.info(
            f"UCGCodeGraphDatasetV2 | max_guards_per_path={self.max_guards_per_path} | "
            f"filter_other={self.filter_other} | max_paths={self.max_paths} | "
            f"random_context_pad={self.random_context_pad} | context_mode={self.context_mode} | "
            f"context_ratio={self.context_ratio} | samples={len(self)}"
        )

    # ── v3 FIX: Bridge edges around OTHER nodes before removing them ──────
    @staticmethod
    def _bridge_and_filter_edges(edge_list, drop_keys):
        """
        Given a raw edge list and a set of node keys to drop,
        create transitive bridge edges and remove all edges
        that touch a dropped node.

        If  A --[type1]--> OTHER --[type2]--> B,
        we add  A --[type1]--> B  (keeps the source edge type).

        This preserves graph connectivity after removing junk nodes.
        """
        if not drop_keys:
            return edge_list

        # Build adjacency for dropped nodes only
        # incoming[dropped_key] = [(src, edge_type), ...]
        # outgoing[dropped_key] = [(dst, edge_type), ...]
        incoming = {}  # edges INTO a dropped node
        outgoing = {}  # edges OUT OF a dropped node
        clean = []     # edges that don't touch any dropped node

        for e in edge_list:
            src, etype, dst = e[0], e[1], e[2]
            src_drop = src in drop_keys
            dst_drop = dst in drop_keys

            if not src_drop and not dst_drop:
                clean.append(e)
            elif dst_drop and not src_drop:
                incoming.setdefault(dst, []).append((src, etype))
            elif src_drop and not dst_drop:
                outgoing.setdefault(src, []).append((dst, etype))
            # else: both dropped → discard entirely

        # Bridge: for each dropped node, connect all its predecessors
        # to all its successors
        bridged = set()
        for dropped_key in drop_keys:
            preds = incoming.get(dropped_key, [])
            succs = outgoing.get(dropped_key, [])
            for (pred_src, pred_etype) in preds:
                for (succ_dst, _) in succs:
                    bridge_key = (pred_src, pred_etype, succ_dst)
                    if bridge_key not in bridged:
                        bridged.add(bridge_key)
                        clean.append([pred_src, pred_etype, succ_dst])

        return clean

    def __getitem__(self, idx):
        # ── 1. Load raw sample ────────────────────────────────────────────────
        offset = self.offsets[idx]
        with open(self.file_path, 'rb') as f:
            f.seek(offset)
            sample = json.loads(f.readline().decode('utf-8'))

        # ── 2. Tokenise source code ───────────────────────────────────────────
        code = self._clean_code(str(sample.get('code', '')))
        input_ids = torch.tensor(self._text_to_token_ids(code))

        # ── 3. Node metadata ──────────────────────────────────────────────────
        graph_data   = sample.get('graph_data', {})
        nodes_dict   = graph_data.get('nodes', {})
        node_keys    = list(nodes_dict.keys())

        # USCP structural roles (needed early for filtering)
        uscp_roles_dict = graph_data.get('uscp_node_roles', {})

        # ── 3b. Filter OTHER nodes ──────────────────────
        #  Identify nodes to DROP, then bridge their edges before removing them.
        #  This keeps only CALL, DEREF, ENTRY, EXIT, ASSIGN, GUARD nodes.
        if self.filter_other:
            drop_keys = set()
            for k in node_keys:
                role = uscp_roles_dict.get(k, "OTHER")
                if role not in VULN_ROLES:
                    drop_keys.add(k)
            # Keep at least 3 nodes to avoid degenerate graphs
            if len(node_keys) - len(drop_keys) < 3:
                drop_keys = set()  # fallback: keep everything
        else:
            drop_keys = set()

        # Rebuild node_keys without dropped nodes
        if drop_keys:
            node_keys = [k for k in node_keys if k not in drop_keys]
        if not node_keys:
            return None  # skip degenerate sample

        node_key_to_idx = {k: i for i, k in enumerate(node_keys)}

        node_labels = [
            self._clean_code(nodes_dict[k].get('label', '')) for k in node_keys
        ] or [""]

        # NPY cache (optional)
        if self.use_npy and self.npy_dir:
            npy_path = os.path.join(self.npy_dir, f"sample_{idx}.npy")
            node_features = (
                torch.tensor(np.load(npy_path), dtype=torch.float32)
                if os.path.exists(npy_path)
                else torch.zeros((len(node_labels), 768), dtype=torch.float32)
            )
        else:
            node_features = torch.empty(0)

        # Role IDs (after filtering, no OTHER nodes remain)
        if self.use_roles:
            role_ids = [
                self.role_map.get(uscp_roles_dict.get(k, "OTHER"), 0) for k in node_keys
            ] or [0]
        else:
            role_ids = [0] * max(len(node_keys), 1)
        node_role_ids = torch.tensor(role_ids, dtype=torch.long)

        # ── 4. Raw edge lists — bridge around dropped nodes ───────────────────
        cfg_edges_raw = self._bridge_and_filter_edges(
            graph_data.get('cfg_edges', []), drop_keys)
        dfg_edges_raw = self._bridge_and_filter_edges(
            graph_data.get('dfg_edges', []), drop_keys)

        # ── 5. Build UCG edges (causal + semantic + uscp + uscp_guards) ───────
        # path_guards is NOT used — intentionally excluded from UCG.

        raw_ucg: list = []

        # 5a. Causal paths → CAUSAL_SEQ edges (REMOVED: Duplicate of SEMANTIC_SEQ)
        # causal_paths and semantic_paths are identical in the dataset,
        # so we only build SEMANTIC_SEQ edges to prevent duplicate signals.

        # 5b. Semantic paths → SEMANTIC_SEQ edges
        semantic_paths = graph_data.get('semantic_paths', [])
        if self.max_paths is not None:
            semantic_paths = semantic_paths[:self.max_paths]
            
        for path in semantic_paths:
            for i in range(len(path) - 1):
                raw_ucg.append([str(path[i]), "SEMANTIC_SEQ", str(path[i + 1])])

        # 5c. USCP paths → USCP_SEQ edges
        #     USCP guards → USCP_GUARD edges (CAPPED to max_guards_per_path)
        uscp_paths      = graph_data.get('uscp_paths',   [])
        uscp_guards_raw = graph_data.get('uscp_guards',  [])
        
        if self.max_paths is not None:
            uscp_paths = uscp_paths[:self.max_paths]
            uscp_guards_raw = uscp_guards_raw[:self.max_paths]
        for path, guards in zip(uscp_paths, uscp_guards_raw):
            for i in range(len(path) - 1):
                raw_ucg.append([str(path[i]), "USCP_SEQ", str(path[i + 1])])
            if path:
                capped_guards = guards[:self.max_guards_per_path]
                for guard in capped_guards:
                    raw_ucg.append([str(guard), "USCP_GUARD", str(path[0])])

        # 5d. Deduplicate
        seen: set = set()
        ucg_edges_raw: list = []
        for e in raw_ucg:
            key = (e[0], e[1], e[2])
            if key not in seen:
                seen.add(key)
                ucg_edges_raw.append(e)

        # 5e. Bridge UCG edges around dropped OTHER nodes
        ucg_edges_raw = self._bridge_and_filter_edges(ucg_edges_raw, drop_keys)

        label = torch.tensor([sample.get('label', 0)], dtype=torch.float)

        # ── 6. Graph slicing (all methods inherited from parent) ──────────────
        if self.no_slice:
            # We don't return None here anymore to prevent dropping large samples. 
            # It will process all nodes in the graph.
            kept = list(range(len(node_keys)))
        elif self.slice_method in ('cta_rwr', 'dfs,cta_rwr', 'cta_rwr,dfs'):
            kept = self._cta_rwr_slice(
                node_keys, node_labels, cfg_edges_raw, dfg_edges_raw,
                uscp_roles_dict,
                min_nodes=self.min_nodes, max_nodes=self.max_nodes,
                mix_with_dfs='dfs' in self.slice_method,
            )
        elif self.slice_method == 'vpc':
            kept = self._vpc_ppr_slice(
                node_keys, node_labels, cfg_edges_raw, dfg_edges_raw,
                uscp_roles_dict,
                min_nodes=self.min_nodes, max_nodes=self.max_nodes,
            )
        elif self.slice_method in ('dfs_fwd', 'dfs_bwd'):
            kept = self._adaptive_bidirectional_dfs_slice(
                node_keys, node_labels, cfg_edges_raw, dfg_edges_raw,
                uscp_roles_dict,
                min_nodes=self.min_nodes, max_nodes=self.max_nodes,
                direction=self.slice_method,
            )
        else:  # default: bidirectional DFS
            kept = self._adaptive_bidirectional_dfs_slice(
                node_keys, node_labels, cfg_edges_raw, dfg_edges_raw,
                uscp_roles_dict,
                min_nodes=self.min_nodes, max_nodes=self.max_nodes,
                direction='both',
            )

        # ── 6b. Context padding (random / hop / rwr) ───────────────────────────
        #  After priority slicing, pad the graph with context nodes.
        #  This adds surrounding context that distinguishes safe code from unsafe
        #  code. Without this, the slicer creates identical subgraphs for both
        #  classes → overfitting + high recall.
        #
        #  Modes:
        #    random — (original) uniformly sample from remaining nodes
        #    hop    — expand from kept set via 1-hop CFG/DFG neighbors
        #    rwr    — pick next-highest RWR/PPR probability nodes from remaining
        if self.random_context_pad and not self.no_slice:
            kept_set = set(kept)
            remaining = [i for i in range(len(node_keys)) if i not in kept_set]
            pad_count = min(
                int(len(kept) * self.context_ratio),
                len(remaining),
                max(0, self.max_nodes - len(kept)),
            )
            if pad_count > 0:
                if self.context_mode == 'hop':
                    # ── Hop-based: BFS 1-hop neighbors of kept nodes ──────────
                    context_nodes = self._hop_context(
                        kept_set, remaining, pad_count,
                        node_keys, cfg_edges_raw, dfg_edges_raw, ucg_edges_raw)
                elif self.context_mode == 'rwr':
                    # ── RWR-based: next-highest PPR probability nodes ─────────
                    context_nodes = self._rwr_context(
                        kept, kept_set, remaining, pad_count,
                        node_keys, cfg_edges_raw, dfg_edges_raw)
                else:
                    # ── Random (default / original) ───────────────────────────
                    context_nodes = random.sample(remaining, pad_count)
                kept.extend(context_nodes)
                kept.sort()

        # ── 7. Apply node slice to metadata ──────────────────────────────────
        node_keys   = [node_keys[i]   for i in kept]
        node_labels = [node_labels[i] for i in kept]
        if node_features.numel() > 0:
            node_features = node_features[kept]
        node_role_ids   = node_role_ids[kept]
        node_key_to_idx = {k: i for i, k in enumerate(node_keys)}

        # ── 8. Filter edges to kept nodes and convert to tensors ──────────────
        def _flt(elist):
            return [e for e in elist if e[0] in node_key_to_idx and e[2] in node_key_to_idx]

        cfg_edge_index, cfg_edge_type = self._process_edges(
            _flt(cfg_edges_raw), node_key_to_idx)
        dfg_edge_index, dfg_edge_type = self._process_edges(
            _flt(dfg_edges_raw), node_key_to_idx)
        ucg_edge_index, ucg_edge_type = self._process_edges(
            _flt(ucg_edges_raw), node_key_to_idx)

        # ── 9. Return 4-view tuple ────────────────────────────────────────────
        return (
            input_ids,
            node_labels,
            node_features,
            node_role_ids,
            cfg_edge_index, cfg_edge_type,   # View 2: CFG
            dfg_edge_index, dfg_edge_type,   # View 3: DFG
            ucg_edge_index, ucg_edge_type,   # View 4: UCG (merged + dedup, guards CAPPED)
            label,
        )

    # ── Context padding helpers ───────────────────────────────────────────────

    @staticmethod
    def _hop_context(kept_set, remaining, pad_count,
                     node_keys, cfg_edges_raw, dfg_edges_raw, ucg_edges_raw):
        """
        Hop-based context: BFS-expand from the kept set through 1-hop and 2-hop
        CFG / DFG / UCG neighbors.  Nodes that are closer (1-hop) are preferred
        over 2-hop, and within the same hop-level, nodes with more connections
        back to the kept set are ranked higher ("most connected first").

        Returns a list of at most `pad_count` node indices.
        """
        remaining_set = set(remaining)
        if not remaining_set:
            return []

        # Build undirected adjacency on node *indices*
        key2idx = {k: i for i, k in enumerate(node_keys)}
        adj = {i: set() for i in range(len(node_keys))}
        for edges in (cfg_edges_raw, dfg_edges_raw, ucg_edges_raw):
            for e in edges:
                u_key, v_key = e[0], e[2]
                if u_key in key2idx and v_key in key2idx:
                    u, v = key2idx[u_key], key2idx[v_key]
                    adj[u].add(v)
                    adj[v].add(u)

        # BFS from kept_set — collect hop-1 then hop-2 candidates
        hop1_counts = {}   # node_idx → count of kept-set neighbors
        hop2_counts = {}

        for k_idx in kept_set:
            for nb in adj.get(k_idx, []):
                if nb in remaining_set:
                    hop1_counts[nb] = hop1_counts.get(nb, 0) + 1

        # Hop-2: neighbors of hop-1 that are NOT in kept_set and NOT hop-1
        hop1_nodes = set(hop1_counts.keys())
        for h1 in hop1_nodes:
            for nb in adj.get(h1, []):
                if nb in remaining_set and nb not in hop1_nodes:
                    hop2_counts[nb] = hop2_counts.get(nb, 0) + 1

        # Rank: hop-1 first (sorted by connection count desc), then hop-2
        ranked = sorted(hop1_counts.keys(), key=lambda x: hop1_counts[x], reverse=True)
        if len(ranked) < pad_count:
            ranked += sorted(hop2_counts.keys(), key=lambda x: hop2_counts[x], reverse=True)

        # If still not enough, fill with random remaining
        if len(ranked) < pad_count:
            leftovers = [i for i in remaining if i not in set(ranked)]
            random.shuffle(leftovers)
            ranked += leftovers

        return ranked[:pad_count]

    def _rwr_context(self, kept, kept_set, remaining, pad_count,
                     node_keys, cfg_edges_raw, dfg_edges_raw):
        """
        RWR-based context: Run a lightweight Personalized PageRank / Random Walk
        with Restart from the kept (sliced) nodes as seeds.  The remaining nodes
        that receive the highest PPR scores are selected as context — these are
        the nodes most structurally "relevant" to the sliced subgraph.

        Returns a list of at most `pad_count` node indices.
        """
        remaining_set = set(remaining)
        if not remaining_set:
            return []

        N = len(node_keys)
        key2idx = {k: i for i, k in enumerate(node_keys)}

        # Build undirected adjacency list
        adj = [[] for _ in range(N)]
        for edges in (cfg_edges_raw, dfg_edges_raw):
            for e in edges:
                u_key, v_key = e[0], e[2]
                if u_key in key2idx and v_key in key2idx:
                    u, v = key2idx[u_key], key2idx[v_key]
                    adj[u].append(v)
                    adj[v].append(u)

        # PPR with restart on kept nodes as seeds
        alpha = 0.85
        iterations = 8
        scores = np.zeros(N, dtype=np.float32)
        teleport = np.zeros(N, dtype=np.float32)
        if kept:
            for s in kept:
                teleport[s] = 1.0 / len(kept)
        else:
            teleport[:] = 1.0 / N
        scores[:] = teleport

        for _ in range(iterations):
            new_scores = np.zeros(N, dtype=np.float32)
            for u in range(N):
                if adj[u]:
                    spread = scores[u] / len(adj[u])
                    for v in adj[u]:
                        new_scores[v] += alpha * spread
            new_scores += (1.0 - alpha) * teleport
            scores = new_scores

        # Rank remaining nodes by their PPR score (descending)
        scored_remaining = [(idx, scores[idx]) for idx in remaining]
        scored_remaining.sort(key=lambda x: x[1], reverse=True)

        return [idx for idx, _ in scored_remaining[:pad_count]]


# =============================================================================
# Collate function for 4-view UCG batches (unchanged from v1)
# =============================================================================

def custom_collate_ucg(batch):
    """Collate a list of UCGCodeGraphDatasetV2 items into a model-ready batch."""
    batch = [s for s in batch if s is not None]
    if not batch:
        return None

    (
        input_ids, node_labels, node_features, node_role_ids,
        cfg_ei, cfg_et,
        dfg_ei, dfg_et,
        ucg_ei, ucg_et,
        labels,
    ) = zip(*batch)

    config_data = {
        'batch_node_labels':   list(node_labels),
        'batch_node_features': list(node_features),
        'batch_node_role_ids': list(node_role_ids),
        # CFG
        'cfg_edges':           list(cfg_ei),
        'cfg_edge_types':      list(cfg_et),
        # DFG
        'dfg_edges':           list(dfg_ei),
        'dfg_edge_types':      list(dfg_et),
        # UCG (unified causal graph — guards capped)
        'ucg_edges':           list(ucg_ei),
        'ucg_edge_types':      list(ucg_et),
    }

    return torch.stack(input_ids), config_data, torch.stack(labels)
