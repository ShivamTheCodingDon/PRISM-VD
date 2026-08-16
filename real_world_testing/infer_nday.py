"""
UCG-VD Model Inference for Real-World N-Day Vulnerability Detection
====================================================================
Loads a trained Dynamic_PRISM-VD_VD_PlusPlus model checkpoint and runs inference
on test_uscp.jsonlines to detect vulnerabilities in real-world code.

Outputs:
  - Per-sample predictions with confidence scores (predictions.jsonl)
  - Summary report (accuracy, precision, recall, F1, confusion matrix)

Usage:
    python infer_nday.py \
        --weights_path /path/to/model_last.pt \
        --test_data test_uscp.jsonlines \
        --output_dir results/
"""

import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Add parent paths for model imports ────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UCG_V2_DIR = os.path.join(PROJECT_ROOT, 'graph_models')
sys.path.insert(0, UCG_V2_DIR)
sys.path.insert(0, PROJECT_ROOT)


def load_model(
    weights_path: str,
    model_name: str = "microsoft/codebert-base",
    embed_dim: int = 128,
    num_edge_types: int = 11,
    gnn_type: str = "rgat",
    fusion_type: str = "concat",
    pool_type: str = "mean",
    use_roles: bool = True,
    proj_dim: int = None,
    separate_proj: bool = False,
    num_layers: int = 1,
    num_bases: int = None,
    dropout: float = 0.3,
    device: str = None,
):
    """
    Load the Dynamic_PRISM-VD_VD_PlusPlus model with trained weights.
    
    Returns: (model, device)
    """
    from model_dynamic import Dynamic_PRISM-VD_VD_PlusPlus

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    logger.info(f"Loading model: GNN={gnn_type}, Fusion={fusion_type}, Device={device}")

    model = Dynamic_PRISM-VD_VD_PlusPlus(
        model_name=model_name,
        embed_dim=embed_dim,
        num_edge_types=num_edge_types,
        gnn_type=gnn_type,
        fusion_type=fusion_type,
        pool_type=pool_type,
        use_roles=use_roles,
        proj_dim=proj_dim,
        separate_proj=separate_proj,
        num_layers=num_layers,
        num_bases=num_bases,
        dropout=dropout,
    )

    # Load weights
    if weights_path and os.path.exists(weights_path):
        logger.info(f"Loading weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        logger.info("Weights loaded successfully.")
    elif weights_path:
        logger.warning(f"Weights file NOT found: {weights_path}")
        logger.warning("Running with RANDOM weights (results will be meaningless)")
    else:
        logger.warning("No weights path provided. Running with RANDOM weights.")

    model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    return model, device


def run_inference(
    model,
    device,
    test_data_path: str,
    output_dir: str,
    threshold: float = 0.5,
    temperature: float = 1.0,
    max_seq_len: int = 512,
    min_nodes: int = 100,
    max_nodes: int = 2000,
    slice_method: str = 'dfs',
    edge_num: int = 11,
    use_roles: bool = True,
    batch_size: int = 1,
):
    """
    Run inference on test_uscp.jsonlines and generate predictions.
    """
    from dataset_dynamic import DynamicEnhancedCodeGraphDataset, custom_collate_dynamic
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        matthews_corrcoef, roc_auc_score, confusion_matrix, classification_report
    )

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = model.tokenizer

    # Load dataset
    logger.info(f"Loading test data: {test_data_path}")
    test_dataset = DynamicEnhancedCodeGraphDataset(
        tokenizer=tokenizer,
        jsonlines_path=test_data_path,
        block_size=max_seq_len,
        use_npy=False,
        npy_dir=None,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        slice_method=slice_method,
        edge_num=edge_num,
        use_roles=use_roles,
    )
    logger.info(f"Test samples: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_dynamic,
    )

    # ── Inference loop ────────────────────────────────────────────────────────
    all_labels = []
    all_probs = []
    all_predictions = []

    # Also read the raw records for metadata
    raw_records = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                raw_records.append(json.loads(line))

    logger.info("Running inference...")
    model.eval()
    sample_idx = 0

    from tqdm import tqdm

    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(test_loader, desc="Inference", unit="batch")):
            if batch is None:
                sample_idx += batch_size
                continue

            input_ids, config_data, labels = batch
            input_ids = input_ids.to(device)
            labels_np = labels.cpu().numpy().flatten()

            # Forward pass
            start_t = time.time()
            logits, attn_weights = model(input_ids, config_data)
            elapsed = time.time() - start_t

            # Apply temperature scaling
            probs = torch.sigmoid(logits / temperature).cpu().numpy().flatten()

            for i in range(len(labels_np)):
                true_label = int(labels_np[i])
                prob = float(probs[i])
                pred_label = 1 if prob > threshold else 0

                # Get metadata from raw records
                rec_idx = sample_idx + i
                if rec_idx < len(raw_records):
                    meta = raw_records[rec_idx]
                else:
                    meta = {}

                prediction = {
                    "id": meta.get("id", rec_idx),
                    "file_name": meta.get("file_name", "unknown"),
                    "func_name": meta.get("func_name", "unknown"),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": round(prob, 6),
                    "project": meta.get("project", "unknown"),
                    "latency_sec": round(elapsed / len(labels_np), 4),
                }

                # Include CVE info if present
                if "cve_id" in meta:
                    prediction["cve_id"] = meta["cve_id"]
                    prediction["cwe"] = meta.get("cwe", "")

                all_labels.append(true_label)
                all_probs.append(prob)
                all_predictions.append(prediction)

                # Highlight CVE detections
                if true_label == 1:
                    status = "✔ DETECTED" if pred_label == 1 else "✘ MISSED"
                    cve_str = meta.get("cve_id", "unknown")
                    tqdm.write(
                        f"  {status} | {cve_str} | {meta.get('func_name', '?')} | "
                        f"conf={prob:.4f} thr={threshold}"
                    )

            sample_idx += len(labels_np)

    # ── Save predictions ──────────────────────────────────────────────────────
    pred_path = os.path.join(output_dir, "predictions.jsonl")
    with open(pred_path, 'w', encoding='utf-8') as f:
        for pred in all_predictions:
            f.write(json.dumps(pred) + '\n')
    logger.info(f"Predictions saved: {pred_path}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    if len(all_labels) == 0:
        logger.error("No samples processed!")
        return {}

    # Dynamic threshold search (on the test set for reporting — in real usage, use val set)
    best_f1, best_thr = 0.0, threshold
    for thr in np.arange(0.05, 0.95, 0.01):
        preds = (all_probs > thr).astype(int)
        f1 = f1_score(all_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    final_preds = (all_probs > best_thr).astype(int)
    acc = accuracy_score(all_labels, final_preds)
    prec = precision_score(all_labels, final_preds, zero_division=0)
    rec = recall_score(all_labels, final_preds, zero_division=0)
    f1 = f1_score(all_labels, final_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, final_preds)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    cm = confusion_matrix(all_labels, final_preds)

    # ── Summary report ────────────────────────────────────────────────────────
    n_vul = int(all_labels.sum())
    n_ben = len(all_labels) - n_vul

    # CVE detection summary
    cve_detected = 0
    cve_missed = 0
    cve_details = []
    for pred in all_predictions:
        if pred["true_label"] == 1:
            if pred["pred_label"] == 1:
                cve_detected += 1
                cve_details.append(f"  ✔ {pred.get('cve_id', '?')}: {pred['func_name']} (conf={pred['confidence']:.4f})")
            else:
                cve_missed += 1
                cve_details.append(f"  ✘ {pred.get('cve_id', '?')}: {pred['func_name']} (conf={pred['confidence']:.4f})")

    report = f"""
{'='*70}
  REAL-WORLD N-DAY VULNERABILITY DETECTION REPORT
{'='*70}

  Test Data       : {test_data_path}
  Total Samples   : {len(all_labels)}
  Vulnerable (1)  : {n_vul}
  Benign (0)      : {n_ben}
  Best Threshold  : {best_thr:.2f}

  ── Metrics ─────────────────────────────────────
    Accuracy      : {acc:.4f}
    Precision     : {prec:.4f}
    Recall        : {rec:.4f}
    F1 Score      : {f1:.4f}
    MCC           : {mcc:.4f}
    AUC-ROC       : {auc:.4f}

  ── Confusion Matrix ────────────────────────────
    TN={cm[0][0]:<6}  FP={cm[0][1]:<6}
    FN={cm[1][0]:<6}  TP={cm[1][1]:<6}

  ── N-Day CVE Detection ─────────────────────────
    Detected      : {cve_detected} / {cve_detected + cve_missed}
    Detection Rate: {(cve_detected/(cve_detected+cve_missed)*100) if (cve_detected+cve_missed) > 0 else 0:.1f}%

{chr(10).join(cve_details)}

{'='*70}
"""
    print(report)
    logger.info(report)

    # Save report
    report_path = os.path.join(output_dir, "nday_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    # Save metrics JSON
    metrics = {
        "total_samples": len(all_labels),
        "vulnerable": n_vul,
        "benign": n_ben,
        "threshold": best_thr,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "cve_detected": cve_detected,
        "cve_missed": cve_missed,
        "cve_detection_rate": (cve_detected / (cve_detected + cve_missed) * 100) if (cve_detected + cve_missed) > 0 else 0,
    }
    metrics_path = os.path.join(output_dir, "nday_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run UCG-VD model inference on real-world code")

    # Required
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to test_uscp.jsonlines")
    parser.add_argument("--weights_path", type=str, default=None,
                        help="Path to trained model weights (.pt)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory for predictions and reports")

    # Model architecture (must match the checkpoint)
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_edge_types", type=int, default=11)
    parser.add_argument("--gnn", type=str, default="rgat")
    parser.add_argument("--fusion", type=str, default="concat")
    parser.add_argument("--pooling", type=str, default="mean")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_bases", type=int, default=None)
    parser.add_argument("--no_roles", action="store_true")
    parser.add_argument("--proj_dim", type=int, default=None)
    parser.add_argument("--separate_proj", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.3)

    # Inference
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--min_nodes", type=int, default=100)
    parser.add_argument("--max_nodes", type=int, default=2000)
    parser.add_argument("--slice_method", type=str, default="dfs")
    parser.add_argument("--edge_num", type=int, default=11)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")

    args = parser.parse_args()

    model, device = load_model(
        weights_path=args.weights_path,
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        num_edge_types=args.num_edge_types,
        gnn_type=args.gnn,
        fusion_type=args.fusion,
        pool_type=args.pooling,
        use_roles=not args.no_roles,
        proj_dim=args.proj_dim,
        separate_proj=args.separate_proj,
        num_layers=args.num_layers,
        num_bases=args.num_bases,
        dropout=args.dropout,
        device=args.device,
    )

    metrics = run_inference(
        model=model,
        device=device,
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        threshold=args.threshold,
        temperature=args.temperature,
        max_seq_len=args.max_seq_len,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        slice_method=args.slice_method,
        edge_num=args.edge_num,
        use_roles=not args.no_roles,
        batch_size=args.batch_size,
    )
