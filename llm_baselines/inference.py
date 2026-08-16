"""
llm_baselines – Inference Engine
============================
Streams through `test_uscp.jsonlines` (Devign), sends each C/C++ function
to GLM-5.2 via NVIDIA Build Platform, and writes per-sample predictions to
`results/predictions.jsonl`.

Features
--------
* Resume:   already-processed IDs are skipped (--resume flag or auto-detect)
* Dry-run:  prints prompts without calling the API (--dry-run)
* Limit:    process only the first N samples (--limit N)
* Progress: tqdm bar with live accuracy estimate
* Logging:  INFO to console + DEBUG to results/inference.log
"""

from __future__ import annotations

import json
import logging
import sys
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

# ── internal modules ──────────────────────────────────────────────────────────
from config import (
    CHECKPOINT_FILE,
    DATASET_PATH,
    MAX_CODE_CHARS,
    MAX_WORKERS,
    PREDICTIONS_FILE,
    RESULTS_DIR,
)
from llm_client import NvidiaLLMClient
from prompts import build_messages

# ── logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = RESULTS_DIR / "inference.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─── Dataset loading ──────────────────────────────────────────────────────────

def iter_dataset(path: Path) -> Iterator[dict]:
    """Yield parsed records from a .jsonlines / .jsonl file."""
    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %d – JSON parse error: %s", line_no, exc)


# ─── Threading Locks ─────────────────────────────────────────────────────────
_file_lock = threading.Lock()

# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def load_checkpoint(path: Path) -> set[str]:
    """Return set of sample IDs that have already been processed."""
    if not path.exists():
        return set()
    with open(path, "r", encoding="utf-8") as fh:
        return {line.strip() for line in fh if line.strip()}


def save_checkpoint(path: Path, sample_id: str) -> None:
    """Append a single sample ID to the checkpoint file."""
    with _file_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(sample_id + "\n")


# ─── Prediction writer ────────────────────────────────────────────────────────

def append_prediction(path: Path, record: dict) -> None:
    """Append a single prediction record as a JSON line."""
    with _file_lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")


# ─── Main inference loop ──────────────────────────────────────────────────────

def run_inference(
    limit: int | None = None,
    resume: bool = True,
    dry_run: bool = False,
    dataset_path: Path | None = None,
    output_file: Path | None = None,
    checkpoint_file: Path | None = None,
) -> None:
    """
    Main inference loop.

    Parameters
    ----------
    limit           : Stop after processing this many samples (None = all).
    resume          : Skip samples whose IDs appear in the checkpoint file.
    dry_run         : Build prompts and print them, but do NOT call the API.
    dataset_path    : Override default DATASET_PATH from config.
    output_file     : Override default PREDICTIONS_FILE from config.
    checkpoint_file : Override default CHECKPOINT_FILE from config.
    """
    dataset_path    = dataset_path    or DATASET_PATH
    output_file     = output_file     or PREDICTIONS_FILE
    checkpoint_file = checkpoint_file or CHECKPOINT_FILE

    # ── Validate dataset ──────────────────────────────────────────────────────
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    done_ids: set[str] = set()
    if resume:
        done_ids = load_checkpoint(checkpoint_file)
        if done_ids:
            logger.info("Resuming – skipping %d already-processed samples.", len(done_ids))

    # ── Init client (only if not dry-run) ────────────────────────────────────
    client: NvidiaLLMClient | None = None
    if not dry_run:
        client = NvidiaLLMClient()

    # ── Counters for live metrics ─────────────────────────────────────────────
    total_processed = 0
    total_correct   = 0
    total_skipped   = 0
    t_start         = time.time()

    logger.info(
        "Starting inference | dataset=%s | limit=%s | resume=%s | dry_run=%s",
        dataset_path.name, limit, resume, dry_run,
    )

    samples = list(iter_dataset(dataset_path))
    if limit:
        samples = samples[:limit]

    # ── Processor function for threading ─────────────────────────────────────
    def process_record(record: dict) -> dict | None:
        sample_id  = str(record.get("id", record.get("file_name", "unknown")))
        
        if resume and sample_id in done_ids:
            return {"status": "skipped", "id": sample_id}
            
        code       = record.get("code", record.get("func", ""))
        true_label = int(record.get("label", record.get("target", 0)))
        
        if dry_run:
            msgs = build_messages(code, max_chars=MAX_CODE_CHARS)
            return {
                "status": "dry_run", 
                "id": sample_id, 
                "true": true_label, 
                "sys": msgs[0]['content'][:300], 
                "usr": msgs[1]['content'][:500]
            }
            
        # Prevent API rate limiting (HTTP 429) — Limit is 40 RPM
        # Sleeping 2.5s guarantees max 24 requests per minute.
        time.sleep(2.5)
        
        # Live inference
        prediction = client.predict(code=code, sample_id=sample_id)
        
        pred_label = prediction["vulnerable"]
        is_correct = int(pred_label == true_label)
        
        if not prediction["parse_ok"] and "failed" in prediction["reason"].lower():
            tqdm.write(f"⚠ Sample {sample_id:<4} | Inference failed (likely rate limit/token size). Skipping save so it can be retried.")
            return {"status": "error", "id": sample_id}

        out = {
            "id":            sample_id,
            "file_name":     record.get("file_name", sample_id),
            "true_label":    true_label,
            "pred_label":    pred_label,
            "correct":       bool(is_correct),
            "confidence":    prediction["confidence"],
            "cwe":           prediction.get("cwe", "N/A"),
            "reason":        prediction.get("reason", ""),
            "parse_ok":      prediction["parse_ok"],
            "latency_sec":   prediction["latency_sec"],
            "raw_response":  prediction.get("raw_response", ""),
            "prompt_tokens": prediction.get("prompt_tokens", 0),
            "completion_tokens": prediction.get("completion_tokens", 0),
            "total_tokens":  prediction.get("total_tokens", 0),
        }
        return {"status": "success", "id": sample_id, "out": out, "correct": is_correct}

    # ── Execute with ThreadPool ──────────────────────────────────────────────
    pbar = tqdm(total=len(samples), desc="Inference", unit="sample", dynamic_ncols=True)

    # Pre-create files to avoid concurrent file-creation race conditions
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.touch(exist_ok=True)
    checkpoint_file.touch(exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_record, r): r for r in samples}
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if not res:
                pbar.update(1)
                continue
                
            if res["status"] == "skipped":
                total_skipped += 1
            elif res["status"] == "dry_run":
                total_processed += 1
                print(f"\n{'='*60}")
                print(f"[DRY-RUN] Sample ID: {res['id']}  |  True label: {res['true']}")
                print(f"SYSTEM:\n{res['sys']}…")
                print(f"USER:\n{res['usr']}…")
            elif res["status"] == "success":
                total_processed += 1
                total_correct += res["correct"]
                append_prediction(output_file, res["out"])
                save_checkpoint(checkpoint_file, res["id"])
                
                # Show processing in output so user sees what it's doing
                out_data = res["out"]
                tqdm.write(
                    f"✔ Sample {out_data['id']:<4} | "
                    f"Pred: {out_data['pred_label']} (True: {out_data['true_label']}) | "
                    f"CWE: {out_data['cwe']:<7} | "
                    f"Reason: {out_data['reason'][:70]}..."
                )
                
            # Update progress bar
            running_acc = total_correct / total_processed * 100 if total_processed > 0 else 0
            elapsed     = time.time() - t_start
            rate        = total_processed / elapsed if elapsed > 0 else 0
            pbar.set_postfix(
                acc=f"{running_acc:.1f}%",
                correct=total_correct,
                rate=f"{rate:.2f}/s",
                refresh=True,
            )
            pbar.update(1)
            
    pbar.close()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    logger.info(
        "Done! processed=%d skipped=%d correct=%d acc=%.2f%% elapsed=%.1fs",
        total_processed,
        total_skipped,
        total_correct,
        (total_correct / total_processed * 100) if total_processed else 0,
        elapsed,
    )
    if not dry_run:
        logger.info("Predictions saved to: %s", output_file)
        logger.info("Run  python evaluate.py  for full publication metrics.")
