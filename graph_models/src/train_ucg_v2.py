import argparse
import logging
import os
import sys
import json

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef,
)
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from model_ucg   import UCG_PRISM-VD_VD, FocalWCE_Loss
from dataset_graph_models import UCGCodeGraphDatasetV2, custom_collate_ucg
from dataset_dynamic import EDGE_TYPE_COUNTS
from gnn_backbones   import SUPPORTED_GNNS
from fusion_strategies import SUPPORTED_FUSIONS

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation (identical to train_ucg.py)
# =============================================================================

def evaluate(model, dataloader, device, criterion=None,
             temperature=1.0, precision_guard=0.0,
             use_mcc=False, fixed_threshold=None):
    model.eval()
    all_labels, all_probs = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            input_ids, config_data, labels = batch
            input_ids = input_ids.to(device)
            labels    = labels.to(device)
            logits, *_ = model(input_ids, config_data)
            if criterion:
                total_loss += criterion(logits, labels).item()
            probs = torch.sigmoid(logits / temperature)
            all_labels.extend(labels.cpu().numpy().flatten())
            all_probs.extend(probs.cpu().numpy().flatten())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)

    if fixed_threshold is not None:
        best_thr = fixed_threshold
    else:
        best_score, best_thr = -1.0, 0.5
        for thr in np.arange(0.05, 0.95, 0.01):
            preds = (all_probs > thr).astype(int)
            if precision_score(all_labels, preds, zero_division=0) < precision_guard:
                continue
            score = (matthews_corrcoef(all_labels, preds) if use_mcc
                     else f1_score(all_labels, preds, zero_division=0))
            if score > best_score:
                best_score, best_thr = score, thr

    preds = (all_probs > best_thr).astype(int)
    acc   = accuracy_score(all_labels, preds)
    rec   = recall_score(all_labels, preds, zero_division=0)
    prec  = precision_score(all_labels, preds, zero_division=0)
    f1    = f1_score(all_labels, preds, zero_division=0)
    mcc   = matthews_corrcoef(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    cm      = confusion_matrix(all_labels, preds)
    avg_loss = total_loss / max(len(dataloader), 1)
    return acc, rec, prec, f1, auc, mcc, cm, best_thr, avg_loss


def find_temperature(model, val_loader, device):
    logger.info("Running Temperature Scaling calibration...")
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            inp, cfg, lbl = batch
            inp = inp.to(device)
            lbl = lbl.to(device)
            logits, *_ = model(inp, cfg)
            all_logits.append(logits.cpu())
            all_labels.append(lbl.cpu())
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels).float()
    T    = torch.nn.Parameter(torch.ones(1))
    opt  = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    def _eval():
        opt.zero_grad()
        loss = loss_fn(all_logits / T, all_labels)
        loss.backward()
        return loss
    opt.step(_eval)
    val = max(T.item(), 0.1)
    logger.info(f"Temperature T = {val:.4f}")
    return val

def _tsne_plot_and_test(X, y, view_name, output_dir, dataset_name, epoch):
    """Run t-SNE on features X, plot scatter, and run centroid separability t-test."""
    suffix = '' if epoch is None else f'_epoch_{epoch}'

    # ── Statistical separability t-test ──────────────────────────────────────
    X_vuln     = X[y == 1]
    X_non_vuln = X[y == 0]

    if len(X_vuln) > 0 and len(X_non_vuln) > 0:
        centroid_non_vuln = np.mean(X_non_vuln, axis=0)
        dist_vuln     = np.linalg.norm(X_vuln     - centroid_non_vuln, axis=1)
        dist_non_vuln = np.linalg.norm(X_non_vuln - centroid_non_vuln, axis=1)
        t_stat, p_val = ttest_ind(dist_vuln, dist_non_vuln, equal_var=False)

        centroid_vuln     = np.mean(X_vuln,     axis=0)
        centroid_dist_D   = np.linalg.norm(centroid_vuln - centroid_non_vuln)

        logger.info(
            f"[{view_name}] t-test p-value={p_val:.4e} | "
            f"MeanDist(Vuln)={np.mean(dist_vuln):.4f} | "
            f"MeanDist(NonVuln)={np.mean(dist_non_vuln):.4f} | "
            f"CentroidDist D={centroid_dist_D:.4f}"
        )

        txt_name = f'{dataset_name}_{view_name}{suffix}_separability_test.txt'
        with open(os.path.join(output_dir, txt_name), 'w') as f:
            f.write(f"Feature View     : {view_name}\n")
            f.write(f"t-statistic      : {t_stat}\n")
            f.write(f"p-value          : {p_val}\n")
            f.write(f"Mean dist Vuln   : {np.mean(dist_vuln):.4f}\n")
            f.write(f"Mean dist NonVuln: {np.mean(dist_non_vuln):.4f}\n")
            f.write(f"Centroid Dist D  : {centroid_dist_D:.4f}\n")
            f.write(
                "A small p-value (< 0.05) and large D proves the model "
                "statistically separates the two classes.\n"
            )

    # ── t-SNE projection ──────────────────────────────────────────────────────
    logger.info(f"Computing t-SNE for [{view_name}] (this may take a minute)...")
    n_samples = X.shape[0]
    perplexity = min(30, max(5, n_samples // 10))
    tsne  = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X_2d  = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1],
                c='#4C72B0', label='Non-Vulnerable', alpha=0.9, s=15)
    plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1],
                c='#C44E52', label='Vulnerable',     alpha=0.9, s=15)
    
    title_view = "Combined (Fused)" if view_name == "Fused" else view_name
    plt.title(f't-SNE Projection — {title_view} ({dataset_name})', fontsize=13)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    tsne_name = f'{dataset_name}_{view_name}{suffix}_tsne.png'
    plt.savefig(os.path.join(output_dir, tsne_name), dpi=150)
    plt.close()
    logger.info(f"t-SNE plot saved: {tsne_name}")


def analyze_embeddings_and_plot(model, dataloader, device, output_dir, dataset_name, epoch=None):
    logger.info("Running Feature Separability Analysis (t-SNE & t-test)...")
    model.eval()
    labels_list = []

    # Feature buckets: Text(0), CFG(1), DFG(2), UCG(3), Fused
    VIEW_NAMES = ['Text', 'CFG', 'DFG', 'UCG', 'Fused']
    all_feats  = {n: [] for n in VIEW_NAMES}

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            inp, cfg, lbl = batch
            inp, lbl = inp.to(device), lbl.to(device)
            out = model(inp, cfg, return_features=True)
            # Model returns: logits, logits_ucg, attn_w, views, fused
            views, fused = out[-2], out[-1]
            labels_list.extend(lbl.cpu().numpy().flatten())
            for i, name in enumerate(['Text', 'CFG', 'DFG', 'UCG']):
                all_feats[name].append(views[i].detach().cpu().numpy())
            all_feats['Fused'].append(fused.detach().cpu().numpy())

    if not any(all_feats[n] for n in VIEW_NAMES):
        logger.warning("No features collected — skipping t-SNE.")
        return

    y = np.array(labels_list)

    for name in VIEW_NAMES:
        X = np.concatenate(all_feats[name], axis=0)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        _tsne_plot_and_test(X, y, name, output_dir, dataset_name, epoch)


# =============================================================================
# Main training loop
# =============================================================================

def run_training(
    dataset_name, train_file, val_file, test_file,
    use_npy=False, npy_dir=None,
    batch_size=1, grad_accum=64, epochs=10,
    lr=5e-5, lr_scratch=5e-4,
    freeze_codebert=False, weight_decay=0.01, patience=5,
    max_seq_len=512,
    pos_weight=1.2, focal_alpha=0.25, focal_gamma=2.0,
    output_dir='results_graph_models',
    min_nodes=100, max_nodes=2000, max_edges=6000,
    resume_path=None, fexpn=False, slice_method='dfs', no_slice=False, ignore_empty_cfg=False,
    mc_dropout=False, dropout_prob=0.3, fixed_threshold=None,
    gnn_type='rgat', fusion_type='concat', pool_type='mean',
    model_name='microsoft/codebert-base',
    edge_num=35, num_layers=1,
    use_custom_vocab=False,
    use_raw_768=False,
    embed_dim=256,
    random_context_pad=True,
    context_ratio=0.3,
    context_mode='random',
    label_smoothing=0.1,
    aux_loss_weight=0.0,
    plot_tsne=False,
    split_gpus=False,
    num_bases=None,
    max_guards=None,
    loss_mode='focal_only',
    save_weights_dir=None,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(
        f"UCG v2 4-view | Dataset:{dataset_name} | "
        f"GNN:[{gnn_type.upper()}] | Fusion:[{fusion_type.upper()}] | "
        f"edge_num={edge_num} | use_raw_768={use_raw_768} | "
        f"max_edges={max_edges} | layers={num_layers}"
    )
    os.makedirs(output_dir, exist_ok=True)
    if save_weights_dir:
        os.makedirs(save_weights_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if use_custom_vocab:
        vocab_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "final_c_cpp_vocab.txt")
        if os.path.exists(vocab_path):
            with open(vocab_path, "r", encoding="utf-8") as f:
                new_tokens = [line.strip() for line in f if line.strip()]
            tokenizer.add_tokens(new_tokens)
            logger.info(f"Loaded {len(new_tokens)} custom tokens into tokenizer.")

    def _npy(split):
        if not npy_dir:
            return None
        for p in [
            os.path.join(npy_dir, f'{dataset_name.lower()}_{split}'),
            os.path.join(npy_dir, split),
        ]:
            if os.path.exists(p):
                return p
        return None

    common = dict(block_size=max_seq_len, use_npy=use_npy,
                  min_nodes=min_nodes, max_nodes=max_nodes,
                  fexpn=fexpn, slice_method=slice_method, no_slice=no_slice, ignore_empty_cfg=ignore_empty_cfg,
                  edge_num=edge_num,
                  random_context_pad=random_context_pad,
                  context_ratio=context_ratio,
                  context_mode=context_mode,
                  max_guards_per_path=max_guards)
    try:
        train_ds = UCGCodeGraphDatasetV2(tokenizer, train_file, npy_dir=_npy('train'), **common)
        val_ds   = UCGCodeGraphDatasetV2(tokenizer, val_file,   npy_dir=_npy('val'),   **common)
        test_ds  = UCGCodeGraphDatasetV2(tokenizer, test_file,  npy_dir=_npy('test'),  **common)
    except FileNotFoundError as e:
        logger.error(f"Data missing: {e}")
        return {}

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=custom_collate_ucg)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              collate_fn=custom_collate_ucg)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              collate_fn=custom_collate_ucg)

    num_edge_types = EDGE_TYPE_COUNTS.get(edge_num, 35)
    model = UCG_PRISM-VD_VD(
        model_name=model_name, embed_dim=embed_dim, num_edge_types=num_edge_types,
        dropout=dropout_prob, pos_weight_val=pos_weight,
        focal_alpha=focal_alpha, focal_gamma=focal_gamma,
        max_edges=max_edges, use_mc_dropout=mc_dropout,
        gnn_type=gnn_type, fusion_type=fusion_type, pool_type=pool_type,
        num_layers=num_layers,
        use_custom_vocab=use_custom_vocab,
        use_raw_768=use_raw_768,
        split_gpus=split_gpus,
        num_bases=num_bases,
    ).to(device)

    if resume_path and os.path.exists(resume_path):
        logger.info(f"Resuming from {resume_path}")
        model.load_state_dict(torch.load(resume_path, map_location=device))

    if freeze_codebert:
        logger.info("CodeBERT frozen.")
        for p in model.codebert.parameters():
            p.requires_grad = False

    no_decay = ['bias', 'LayerNorm.weight']
    cb_params, sc_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (cb_params if 'codebert' in n else sc_params).append((n, p))

    opt_groups = [
        {'params': [p for n, p in cb_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr},
        {'params': [p for n, p in cb_params if     any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': lr},
        {'params': [p for n, p in sc_params if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay, 'lr': lr_scratch},
        {'params': [p for n, p in sc_params if     any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': lr_scratch},
    ]
    optimizer = torch.optim.AdamW(opt_groups, lr=lr)
    criterion = FocalWCE_Loss(pos_weight_val=pos_weight,
                              alpha=focal_alpha, gamma=focal_gamma,
                              label_smoothing=label_smoothing,
                              mode=loss_mode)
    logger.info(f"Loss mode: {loss_mode} | pos_weight={pos_weight} | "
                f"focal_alpha={focal_alpha} | focal_gamma={focal_gamma} | "
                f"label_smoothing={label_smoothing}")

    steps_per_epoch = len(train_loader) // grad_accum + (
        1 if len(train_loader) % grad_accum else 0)
    total_steps  = steps_per_epoch * epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda', enabled=use_amp)

    history = {k: [] for k in [
        'train_loss',
        'val_f1', 'val_acc', 'val_mcc', 'val_loss',
        'test_f1', 'test_acc', 'test_mcc', 'test_loss',
    ]}
    best_val_f1 = -1.0
    best_val_auc = -1.0
    best_val_loss = float('inf')
    no_improve = 0

    csv_path = os.path.join(output_dir, f'{dataset_name}_metrics_history.csv')
    with open(csv_path, 'w') as f:
        f.write("epoch,train_loss,"
                "val_loss,val_acc,val_recall,val_precision,val_f1,val_auc,val_mcc,val_thr,"
                "test_loss,test_acc,test_recall,test_precision,test_f1,test_auc,test_mcc,test_thr\n")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f"Epoch {epoch+1}")
        oom_skips = 0
        for step, batch in pbar:
            if batch is None:
                continue
            inp, cfg, lbl = batch
            inp = inp.to(device)
            lbl = lbl.to(device)
            try:
                with torch.amp.autocast(device.type, enabled=use_amp):
                    logits_fused, logits_ucg, _, views, _ = model(inp, cfg, return_features=True)
                    loss_fused = criterion(logits_fused, lbl)
                    
                    if aux_loss_weight > 0.0:
                        # alpha / (1-alpha) scheme: UCG-only auxiliary supervision
                        alpha = aux_loss_weight  # e.g. 0.7 means 70% fused, 30% UCG aux
                        loss_ucg = criterion(logits_ucg, lbl)
                        loss = (alpha * loss_fused + (1.0 - alpha) * loss_ucg) / grad_accum
                    else:
                        loss = loss_fused / grad_accum
                        
                scaler.scale(loss).backward()
            except torch.cuda.OutOfMemoryError:
                oom_skips += 1
                logger.warning(f"OOM at step {step} — skipping sample (total skips: {oom_skips})")
                # Clean up to recover VRAM
                if 'loss' in dir():
                    del loss
                if 'logits' in dir():
                    del logits
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                continue
            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skip_lr_sched = (scale > scaler.get_scale())
                if not skip_lr_sched:
                    scheduler.step()
                optimizer.zero_grad()
            total_loss += loss.item() * grad_accum
            pbar.set_postfix({'loss': total_loss / (step + 1), 'oom': oom_skips})

        avg_loss = total_loss / max(len(train_loader), 1)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        v_acc, v_rec, v_prec, v_f1, v_auc, v_mcc, _, v_thr, v_loss = evaluate(
            model, val_loader, device, criterion=criterion,
            use_mcc=True, precision_guard=0.35, fixed_threshold=fixed_threshold)
        t_acc, t_rec, t_prec, t_f1, t_auc, t_mcc, _, t_thr, t_loss = evaluate(
            model, test_loader, device, criterion=criterion,
            use_mcc=True, fixed_threshold=v_thr)

        for k, v in [('train_loss', avg_loss), ('val_loss', v_loss),
                     ('val_acc', v_acc), ('val_f1', v_f1), ('val_mcc', v_mcc),
                     ('test_loss', t_loss), ('test_acc', t_acc),
                     ('test_f1', t_f1), ('test_mcc', t_mcc)]:
            history[k].append(v)

        with open(csv_path, 'a') as f:
            f.write(f"{epoch+1},{avg_loss:.4f},"
                    f"{v_loss:.4f},{v_acc:.4f},{v_rec:.4f},{v_prec:.4f},"
                    f"{v_f1:.4f},{v_auc:.4f},{v_mcc:.4f},{v_thr:.2f},"
                    f"{t_loss:.4f},{t_acc:.4f},{t_rec:.4f},{t_prec:.4f},"
                    f"{t_f1:.4f},{t_auc:.4f},{t_mcc:.4f},{t_thr:.2f}\n")

        with open(os.path.join(output_dir, f'epoch_{epoch+1}_metrics.json'), 'w') as f:
            json.dump({'epoch': epoch+1, 'train_loss': avg_loss,
                       'val':  {'loss':v_loss,'acc':v_acc,'rec':v_rec,'prec':v_prec,
                                'f1':v_f1,'auc':v_auc,'mcc':v_mcc,'thr':v_thr},
                       'test': {'loss':t_loss,'acc':t_acc,'rec':t_rec,'prec':t_prec,
                                'f1':t_f1,'auc':t_auc,'mcc':t_mcc,'thr':t_thr}}, f, indent=4)

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            if save_weights_dir:
                torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_best_f1.pt'))
                logger.info(f"*** Best Val F1: {best_val_f1:.4f} — Weights saved to {save_weights_dir}/model_best_f1.pt ***")
            else:
                logger.info(f"*** Best Val F1: {best_val_f1:.4f} — (Weights not saved) ***")

        if v_auc > best_val_auc:
            best_val_auc = v_auc
            if save_weights_dir:
                torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_best_auc.pt'))
                logger.info(f"*** Best Val AUC: {best_val_auc:.4f} — Weights saved to {save_weights_dir}/model_best_auc.pt ***")
            else:
                logger.info(f"*** Best Val AUC: {best_val_auc:.4f} — (Weights not saved) ***")

        if v_loss < best_val_loss:
            best_val_loss = v_loss
            no_improve = 0
            logger.info(f"*** Best Val Loss: {best_val_loss:.4f} ***")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break

        # torch.save(model.state_dict(), os.path.join(output_dir, 'model_last.pt'))
        logger.info(f"Epoch {epoch+1} → Val F1:{v_f1:.4f} Test F1:{t_f1:.4f} "
                    f"VLoss:{v_loss:.4f} Thr:{v_thr:.2f}")

        if plot_tsne:
            analyze_embeddings_and_plot(model, test_loader, device, output_dir, dataset_name, epoch=epoch+1)

    if save_weights_dir:
        torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_last.pt'))
        logger.info(f"Saved model_last.pt to {save_weights_dir}")

    # ── Temperature scaling + final eval ─────────────────────────────────────
    # best_pt = os.path.join(output_dir, 'model_best.pt')
    # last_pt = os.path.join(output_dir, 'model_last.pt')
    # if os.path.exists(best_pt):
    #     model.load_state_dict(torch.load(best_pt, map_location=device))
    #     logger.info("Loaded model_best.pt for final evaluation.")
    # elif os.path.exists(last_pt):
    #     model.load_state_dict(torch.load(last_pt, map_location=device))
    #     logger.info("Loaded model_last.pt for final evaluation.")
    logger.info("Evaluating with current weights (not loading from disk).")
    T = find_temperature(model, val_loader, device)

    acc, rec, prec, f1, auc, mcc, cm, final_thr, test_loss = evaluate(
        model, test_loader, device, criterion=criterion,
        temperature=T, precision_guard=0.0, use_mcc=True,
        fixed_threshold=fixed_threshold)

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history['train_loss'], label='Train', color='red', linestyle='--')
    axes[0].plot(history['val_loss'],   label='Val',   color='blue')
    axes[0].plot(history['test_loss'],  label='Test',  color='green')
    axes[0].set_title('Loss Curves (UCG v2)')
    axes[0].legend()
    axes[1].plot(history['val_f1'],   label='Val F1',  color='blue')
    axes[1].plot(history['test_f1'],  label='Test F1', color='green')
    axes[1].plot(history['val_mcc'],  label='Val MCC', color='blue',  alpha=0.3)
    axes[1].plot(history['test_mcc'], label='Test MCC',color='green', alpha=0.3)
    axes[1].set_title('Metrics (UCG v2 — guards capped)')
    axes[1].legend()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2])
    axes[2].set_title('Confusion Matrix')
    axes[2].set_xlabel('Predicted')
    axes[2].set_ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_graph_models_metrics.png'))

    metrics = {
        'Dataset': dataset_name, 'Model': 'UCG-v2-4view',
        'Accuracy': acc, 'Recall': rec, 'Precision': prec,
        'F1 Score': f1, 'MCC': mcc, 'AUC-ROC': auc,
        'Test_Loss': test_loss, 'Best_Threshold': final_thr,
    }
    with open(os.path.join(output_dir, f'{dataset_name}_final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    if plot_tsne:
        analyze_embeddings_and_plot(model, test_loader, device, output_dir, dataset_name)
        
    return metrics


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='UCG v2 (Guard-Capped) — 4-view training script')

    parser.add_argument('--dataset',      type=str, default=None)
    parser.add_argument('--train_data',   type=str, required=True)
    parser.add_argument('--val_data',     type=str, required=True)
    parser.add_argument('--test_data',    type=str, required=True)
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch_size',   type=int,   default=1)
    parser.add_argument('--grad_accum',   type=int,   default=64)
    parser.add_argument('--max_seq_len',  type=int,   default=512)
    parser.add_argument('--model_name',   type=str,
                        default='microsoft/codebert-base')
    parser.add_argument('--output_dir',   type=str,   default='results_graph_models')
    parser.add_argument('--lr',           type=float, default=5e-5)
    parser.add_argument('--lr_scratch',   type=float, default=5e-4)
    parser.add_argument('--freeze_codebert', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience',     type=int,   default=3)
    parser.add_argument('--embed_dim',    type=int,   default=256)
    parser.add_argument('--mem',          action='store_true')
    parser.add_argument('--npy_dir',      type=str,   default=None)
    parser.add_argument('--min_nodes',    type=int,   default=100)
    parser.add_argument('--max_nodes',    type=int,   default=2000)
    parser.add_argument('--max_edges',    type=int,   default=6000,
                        help='Hard cap on edges per GNN view (default 6000, was 15000 in v1)')
    parser.add_argument('--mc_dropout',   action='store_true')
    parser.add_argument('--dropout_prob', type=float, default=0.5)
    parser.add_argument('--pos_weight',   type=float, default=1.0)
    parser.add_argument('--focal_alpha',  type=float, default=1.0)
    parser.add_argument('--focal_gamma',  type=float, default=2.0)
    parser.add_argument('--resume_path',  type=str,   default=None)
    parser.add_argument('--fexpn',        action='store_true')
    parser.add_argument('--gnn',          type=str,   default='rgat',
                        choices=SUPPORTED_GNNS)
    parser.add_argument('--num_bases',    type=int,   default=None)
    parser.add_argument('--fusion',       type=str,   default='concat',
                        choices=SUPPORTED_FUSIONS)
    parser.add_argument('--pooling',      type=str,   default='attention',
                        choices=['mean', 'attention'])
    parser.add_argument('--slice_method', type=str,   default='cta_rwr',
                        choices=['dfs','dfs_fwd','dfs_bwd','vpc',
                                 'cta_rwr','dfs,cta_rwr','cta_rwr,dfs'])
    parser.add_argument('--fixed_threshold', type=float, default=None)
    parser.add_argument('--no_slice',     action='store_true')
    parser.add_argument('--ignore_empty_cfg', action='store_true')

    # ── Anti-Overfitting Controls ─────────────────────────────────────────
    parser.add_argument("--edge_num", type=int, default=11, choices=[11],
                        help="Number of semantic edge types to use (hardcoded to 11 for optimal anti-overfitting + function boundary awareness)")
    parser.add_argument("--use_raw_768", action="store_true",
                        help="Bypass the 128-dim bottleneck and feed the raw 768-dim CodeBERT features directly into the GNN (requires more VRAM, matches PRISM-VD).")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--use_custom_vocab", action="store_true")
    parser.add_argument("--plot_tsne", action="store_true",
                        help="Generate a t-SNE plot and run a statistical separability test at the end of training.")

    # ── v4: Anti-overfitting — Random context padding + label smoothing ───
    parser.add_argument("--no_random_pad", action="store_true",
                        help="Disable random context node padding after priority slicing.")
    parser.add_argument("--context_ratio", type=float, default=0.3,
                        help="Ratio of random context nodes to add after slicing (default 0.3 = 30%%).")
    parser.add_argument("--context_mode", type=str, default='random',
                        choices=['random', 'hop', 'rwr'],
                        help="Context padding strategy: 'random' (uniform sampling, original), "
                             "'hop' (1/2-hop CFG/DFG neighbors, most-connected-first), "
                             "'rwr' (next-highest PPR/RWR probability nodes). Default: random.")
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor to prevent overconfident predictions (default 0.1).")
    parser.add_argument("--loss_mode", type=str, default='focal_only',
                        choices=['focal_only', 'wbce_only', 'combined'],
                        help="Loss function mode: "
                             "'focal_only' = correct focal loss (alpha for balance, no pos_weight); "
                             "'wbce_only' = weighted BCE (pos_weight for balance, no focal); "
                             "'combined' = legacy broken mode (not recommended). Default: focal_only")
    parser.add_argument("--aux_loss_weight", type=float, default=0.0,
                        help="Alpha for loss weighting: alpha*fused + (1-alpha)*UCG_aux. "
                             "Set 0.0 to disable aux loss. E.g. 0.7 = 70%% fused + 30%% UCG aux.")
    parser.add_argument('--split_gpus', action='store_true', help="Split model across 2 GPUs to double VRAM")
    parser.add_argument('--max_guards',  type=int, default=None,
                        help="Max USCP guard edges per path (default: None = unlimited). "
                             "Set to 3-5 to reduce graph noise and speed up training.")
    parser.add_argument("--save_weights_dir", type=str, default=None,
                        help="Directory to save model_best_f1.pt, model_best_auc.pt and model_last.pt")

    parser.set_defaults(no_slice=False)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'train_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    metrics = run_training(
        dataset_name   = args.dataset,
        train_file     = args.train_data,
        val_file       = args.val_data,
        test_file      = args.test_data,
        use_npy        = args.mem,
        npy_dir        = args.npy_dir,
        batch_size     = args.batch_size,
        grad_accum     = args.grad_accum,
        epochs         = args.epochs,
        lr             = args.lr,
        lr_scratch     = args.lr_scratch,
        freeze_codebert= args.freeze_codebert,
        weight_decay   = args.weight_decay,
        patience       = args.patience,
        max_seq_len    = args.max_seq_len,
        pos_weight     = args.pos_weight,
        focal_alpha    = args.focal_alpha,
        focal_gamma    = args.focal_gamma,
        output_dir     = args.output_dir,
        min_nodes      = args.min_nodes,
        max_nodes      = args.max_nodes,
        max_edges      = args.max_edges,
        resume_path    = args.resume_path,
        fexpn          = args.fexpn,
        slice_method   = args.slice_method,
        no_slice       = args.no_slice,
        ignore_empty_cfg= args.ignore_empty_cfg,
        mc_dropout     = args.mc_dropout,
        dropout_prob   = args.dropout_prob,
        fixed_threshold= args.fixed_threshold,
        gnn_type       = args.gnn,
        fusion_type    = args.fusion,
        pool_type      = args.pooling,
        model_name     = args.model_name,
        edge_num       = args.edge_num,
        num_layers     = args.num_layers,
        use_custom_vocab = args.use_custom_vocab,
        use_raw_768    = args.use_raw_768,
        embed_dim      = args.embed_dim,
        random_context_pad = not args.no_random_pad,
        context_ratio  = args.context_ratio,
        context_mode   = args.context_mode,
        label_smoothing= args.label_smoothing,
        loss_mode      = args.loss_mode,
        aux_loss_weight= args.aux_loss_weight,
        plot_tsne      = args.plot_tsne,
        split_gpus     = args.split_gpus,
        num_bases      = args.num_bases,
        max_guards     = args.max_guards,
        save_weights_dir= args.save_weights_dir,
    )

    print('\n' + '=' * 70)
    print('FINAL RESULTS — UCG v2 (guard-capped) 4-view model')
    print('=' * 70)
    print(pd.DataFrame([metrics]).to_markdown(index=False))
    print('=' * 70 + '\n')
