"""
llm_baselines – Evaluation & Metrics  (sklearn-based)
==================================================
Loads `results/predictions.jsonl` and computes publication-grade binary
classification metrics using scikit-learn for comparison against
graph-based PRISM-VD models.

Metrics reported
----------------
  sklearn.metrics.classification_report  – per-class + macro/weighted avg
  accuracy_score                         – overall accuracy
  precision_score                        – binary precision (label=1)
  recall_score                           – binary recall / TPR (label=1)
  f1_score                               – binary F1 (label=1)
  matthews_corrcoef                      – MCC
  roc_auc_score                          – AUC-ROC (via confidence scores)
  confusion_matrix                       – [[TN, FP], [FN, TP]]
  ConfusionMatrixDisplay                 – ASCII confusion matrix

  FPR / FNR derived from confusion_matrix values
  Latency stats from prediction records (not sklearn)

Usage
-----
  python evaluate.py
  python evaluate.py --predictions results/predictions.jsonl \\
                     --save        results/metrics_summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import METRICS_FILE, PREDICTIONS_FILE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_NAMES = ["Safe (0)", "Vulnerable (1)"]


# ─── Data loader ──────────────────────────────────────────────────────────────

def load_predictions(path: Path) -> list[dict]:
    """Load every JSON-line record from *path*."""
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── sklearn metric computation ───────────────────────────────────────────────

def compute_metrics(records: list[dict]) -> dict:
    """
    Compute all metrics using scikit-learn.

    Returns a JSON-serialisable dict suitable for publication tables.
    """
    if not records:
        raise ValueError("No predictions found — run inference first.")

    y_true  = np.array([r["true_label"] for r in records], dtype=int)
    y_pred  = np.array([r["pred_label"] for r in records], dtype=int)
    # Normalised confidence score → probability estimate for AUC-ROC
    y_score = np.array([r["confidence"] for r in records], dtype=float) / 100.0

    # ── sklearn core metrics ───────────────────────────────────────────────────
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec  = recall_score(y_true, y_pred,    pos_label=1, zero_division=0)
    f1   = f1_score(y_true, y_pred,        pos_label=1, zero_division=0)
    mcc  = matthews_corrcoef(y_true, y_pred)

    # sklearn classification_report as a dict (for JSON export)
    cls_report_dict = classification_report(
        y_true, y_pred,
        target_names=TARGET_NAMES,
        output_dict=True,
        zero_division=0,
    )
    # Human-readable string version (printed to console)
    cls_report_str = classification_report(
        y_true, y_pred,
        target_names=TARGET_NAMES,
        zero_division=0,
    )

    # ── sklearn confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    # FPR / FNR derived from confusion matrix values
    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr_val = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # ── sklearn AUC-ROC ───────────────────────────────────────────────────────
    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except ValueError:
        auc_roc = None   # only one class present (tiny test runs)

    # ── Latency (from record metadata) ────────────────────────────────────────
    latencies = np.array([r.get("latency_sec", 0.0) for r in records], dtype=float)
    avg_lat   = float(latencies.mean())
    total_lat = float(latencies.sum())

    # ── Parse quality ─────────────────────────────────────────────────────────
    parse_fail = sum(1 for r in records if not r.get("parse_ok", True))
    parse_fail_pct = parse_fail / len(records) * 100

    # ── Token Usage ───────────────────────────────────────────────────────────
    total_prompt_tokens = sum(r.get("prompt_tokens", 0) for r in records)
    total_completion_tokens = sum(r.get("completion_tokens", 0) for r in records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)

    # ── Label distribution ─────────────────────────────────────────────────────
    n_vuln      = int(y_true.sum())
    n_safe      = int(len(y_true) - n_vuln)
    n_pred_vuln = int(y_pred.sum())
    n_pred_safe = int(len(y_pred) - n_pred_vuln)

    return {
        # ── dataset ──────────────────────────────────────────────────────────
        "n_samples":            len(records),
        "n_vulnerable_true":    n_vuln,
        "n_safe_true":          n_safe,
        "n_vulnerable_pred":    n_pred_vuln,
        "n_safe_pred":          n_pred_safe,

        # ── sklearn binary metrics ────────────────────────────────────────────
        "accuracy":             round(acc,  4),
        "precision":            round(prec, 4),
        "recall":               round(rec,  4),
        "f1_score":             round(f1,   4),
        "mcc":                  round(mcc,  4),
        "auc_roc":              round(auc_roc, 4) if auc_roc is not None else None,

        # ── sklearn confusion matrix ──────────────────────────────────────────
        "confusion_matrix": {
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
            "array": cm.tolist(),
        },

        # ── derived rates ─────────────────────────────────────────────────────
        "fpr":                  round(fpr_val, 4),
        "fnr":                  round(fnr_val, 4),

        # ── sklearn classification_report (full) ──────────────────────────────
        "classification_report": cls_report_dict,
        "_classification_report_str": cls_report_str,   # for console display

        # ── quality & timing ──────────────────────────────────────────────────
        "parse_fail_count":     parse_fail,
        "parse_fail_pct":       round(parse_fail_pct, 2),
        "avg_latency_sec":      round(avg_lat,   3),
        "total_latency_sec":    round(total_lat, 1),
        
        # ── token usage ───────────────────────────────────────────────────────
        "total_prompt_tokens":      total_prompt_tokens,
        "total_completion_tokens":  total_completion_tokens,
        "total_tokens_used":        total_tokens,
    }


# ─── Console printer ──────────────────────────────────────────────────────────

def print_metrics(m: dict) -> None:
    SEP  = "─" * 56
    SEP2 = "═" * 56

    print(f"\n{SEP2}")
    print("   llm_baselines  ·  GLM-5.2  ·  Vulnerability Detection")
    print(f"   Dataset : Devign   |   Samples : {m['n_samples']}")
    print(SEP2)

    # ── Distribution ──────────────────────────────────────────────────────────
    print(f"\n  LABEL DISTRIBUTION")
    print(f"  True  –  Vulnerable: {m['n_vulnerable_true']:>5}  "
          f"Safe: {m['n_safe_true']:>5}")
    print(f"  Pred  –  Vulnerable: {m['n_vulnerable_pred']:>5}  "
          f"Safe: {m['n_safe_pred']:>5}")

    # ── sklearn classification_report ─────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SKLEARN CLASSIFICATION REPORT")
    print(SEP)
    for line in m["_classification_report_str"].splitlines():
        print(f"  {line}")

    # ── Publication metrics table ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PUBLICATION METRICS  (binary, positive = vulnerable)")
    print(SEP)
    print(f"  {'Accuracy':<24} {m['accuracy']*100:>7.2f} %")
    print(f"  {'Precision':<24} {m['precision']*100:>7.2f} %")
    print(f"  {'Recall  (TPR)':<24} {m['recall']*100:>7.2f} %")
    print(f"  {'F1-Score':<24} {m['f1_score']*100:>7.2f} %")
    print(f"  {'MCC':<24} {m['mcc']:>8.4f}")
    auc_str = f"{m['auc_roc']:.4f}" if m["auc_roc"] is not None else "     N/A"
    print(f"  {'AUC-ROC':<24} {auc_str:>8}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = m["confusion_matrix"]
    print(f"\n{SEP}")
    print("  CONFUSION MATRIX  (sklearn)")
    print(SEP)
    print(f"  {'':16}  Pred Safe   Pred Vuln")
    print(f"  {'True Safe':16}  {cm['TN']:>9}   {cm['FP']:>9}")
    print(f"  {'True Vuln':16}  {cm['FN']:>9}   {cm['TP']:>9}")

    # ── Secondary ─────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SECONDARY / DIAGNOSTICS")
    print(SEP)
    print(f"  {'FPR (False Pos. Rate)':<26} {m['fpr']*100:>6.2f} %")
    print(f"  {'FNR (False Neg. Rate)':<26} {m['fnr']*100:>6.2f} %")
    print(f"  {'Parse Failures':<26} {m['parse_fail_count']:>6}  ({m['parse_fail_pct']:.1f}%)")
    print(f"  {'Avg latency / sample':<26} {m['avg_latency_sec']:>6.3f} s")
    print(f"  {'Total inference time':<26} {m['total_latency_sec']:>6.0f} s")
    print(f"\n{SEP2}\n")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate llm_baselines predictions with sklearn metrics."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=PREDICTIONS_FILE,
        help=f"Path to predictions.jsonl  (default: {PREDICTIONS_FILE})",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=METRICS_FILE,
        help=f"Path to write metrics JSON  (default: {METRICS_FILE})",
    )
    args = parser.parse_args()

    if not args.predictions.exists():
        logger.error("Predictions file not found: %s", args.predictions)
        logger.error("Run  python run_devign.py  first.")
        sys.exit(1)

    records = load_predictions(args.predictions)
    logger.info("Loaded %d prediction records.", len(records))

    metrics = compute_metrics(records)
    print_metrics(metrics)

    # Serialise — drop the str version of classification_report (redundant)
    export = {k: v for k, v in metrics.items() if not k.startswith("_")}
    with open(args.save, "w", encoding="utf-8") as fh:
        json.dump(export, fh, indent=2)
    logger.info("Metrics saved → %s", args.save)


if __name__ == "__main__":
    main()
