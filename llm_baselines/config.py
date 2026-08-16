"""
llm_baselines Configuration
======================
Central config for NVIDIA Build Platform GLM-5.2 vulnerability detection
on the Devign dataset.  Edit this file before running inference.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ─── Paths ───────────────────────────────────────────────────────────────────

BASE_DIR       = Path(__file__).parent.resolve()
DATA_DIR       = BASE_DIR / "data"
RESULTS_DIR    = BASE_DIR / "results"

# Load .env file
load_dotenv(BASE_DIR / ".env")

# Devign test split (absolute path)

DATASET_PATH   = Path(
    "/home/user1/AIVul(Don't Delete It)/PRISM-VD"
    "/PRISM-VD-Enhanced/data/processed/Reveal/test_uscp.jsonlines"
)
# DATASET_PATH   = BASE_DIR / "test_uscp.jsonlines"

# Where per-sample predictions are saved (JSONL)
PREDICTIONS_FILE = RESULTS_DIR / "predictions.jsonl"

# Checkpoint: stores set of already-processed sample IDs (for resume)
CHECKPOINT_FILE  = RESULTS_DIR / "checkpoint.txt"

# Final metrics JSON
METRICS_FILE     = RESULTS_DIR / "metrics_summary.json"

# ─── Azure API ──────────────────────────────────────────────────────────────

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "")

# ─── Model ───────────────────────────────────────────────────────────────────

# For now use any one LLM (deployment name), can change whenever needed
MODEL_NAME   = os.getenv("MODEL_NAME", "Codestral-Agent")
TEMPERATURE  = 0.7       
TOP_P        = 0.95
MAX_TOKENS   = 4096      
SEED         = 42

# ─── Inference ───────────────────────────────────────────────────────────────

# Max C code characters sent to the model (trim very long functions)
MAX_CODE_CHARS   = 4096

# Retry / back-off settings
MAX_RETRIES      = 5          # Per-sample retries on API errors
RETRY_BASE_DELAY = 2.0        # seconds — doubled each retry (exponential back-off)
API_TIMEOUT      = 180        # seconds — GLM-5.2 can be slow on large C functions

# Concurrency
# If you have 30 CPUs and want to run in parallel, increase MAX_WORKERS.
# WARNING: Setting this too high may hit NVIDIA API rate limits (HTTP 429).
MAX_WORKERS      = 5

# ─── Ensure output directories exist ─────────────────────────────────────────

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
