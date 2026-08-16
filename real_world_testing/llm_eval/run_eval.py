"""
llm_baselines – CLI Entry Point
===========================
Run vulnerability detection on the Devign test split using NVIDIA GLM-5.2.

Usage examples
--------------
# Dry-run: inspect prompts on 3 samples (no API calls)
python run_devign.py --limit 3 --dry-run

# Quick sanity check: 10 real samples
python run_devign.py --limit 10

# Full run (2796 samples) with automatic resume
python run_devign.py --resume

# Re-run specific output dir
python run_devign.py --output-dir results/run2 --resume

# Evaluate results
python evaluate.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# ─── Parse args BEFORE importing heavy deps (faster --help) ──────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_devign",
        description=(
            "llm_baselines – NVIDIA GLM-5.2 vulnerability detection\n"
            "on the Devign test_uscp.jsonlines dataset.\n\n"
            "Results are written to results/predictions.jsonl\n"
            "then evaluated with:  python evaluate.py"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process only the first N samples (default: all 2796).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help=(
            "Skip samples already in the checkpoint file (default: True). "
            "Use --no-resume to start fresh."
        ),
    )
    p.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore checkpoint and re-process all samples.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print prompts without making any API calls.",
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override the default dataset path from config.py.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Override the default results/ directory.",
    )
    p.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Console log verbosity (default: INFO).",
    )
    return p


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_parser().parse_args()

    # Configure logging BEFORE importing inference (which sets up its own handler)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    # Silence noisy HTTP and OpenAI retry logs so they don't look like errors
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # ── Override paths if requested ───────────────────────────────────────────
    from config import CHECKPOINT_FILE, PREDICTIONS_FILE, RESULTS_DIR

    output_dir      = args.output_dir or RESULTS_DIR
    output_dir      = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_out = output_dir / "predictions.jsonl"
    checkpoint_out  = output_dir / "checkpoint.txt"

    # ── Print banner ──────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║          llm_baselines  ·  Azure OpenAI               ║")
    print("║   Devign Vulnerability Detection Baseline        ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print(f"  Dataset  : {args.dataset or 'Devign test_uscp.jsonlines (default)'}")
    print(f"  Limit    : {args.limit or 'ALL 2796 samples'}")
    print(f"  Resume   : {args.resume}")
    print(f"  Dry-run  : {args.dry_run}")
    print(f"  Output   : {output_dir}")
    print()

    if args.dry_run:
        print("  ⚠  DRY-RUN MODE — no API calls will be made.\n")

    # ── Full run hint ─────────────────────────────────────────────────────────
    if not args.dry_run and args.limit is None:
        print(
            "  ℹ  Running inference on ALL 2796 samples.\n"
            "     This may take a while depending on API rate limits.\n"
        )

    # ── Run ───────────────────────────────────────────────────────────────────
    from inference import run_inference

    run_inference(
        limit           = args.limit,
        resume          = args.resume,
        dry_run         = args.dry_run,
        dataset_path    = args.dataset,
        output_file     = predictions_out,
        checkpoint_file = checkpoint_out,
    )

    # ── Post-run hint ─────────────────────────────────────────────────────────
    if not args.dry_run:
        print()
        print("  ✅  Inference complete!")
        print(f"  📄  Predictions : {predictions_out}")
        print()
        print("  Next step — evaluate:")
        print(f"      python evaluate.py --predictions {predictions_out}")
        print()


if __name__ == "__main__":
    main()
