# llm_baselines – NVIDIA GLM-5.2 Vulnerability Detection Baseline

> **Purpose**: Establish a pure-LLM baseline for C/C++ vulnerability detection on the **Devign** benchmark, suitable for publication comparison against graph-based models (PRISM-VD, VulnHGT, MVGD, DiverseVul, etc.).

---

## Directory Structure

```
llm_baselines/
├── config.py          # API key, model params, file paths
├── prompts.py         # Security auditor system + user prompt templates
├── llm_client.py      # ChatNVIDIA wrapper with retry & JSON extraction
├── inference.py       # Main inference loop (resume, checkpoint, tqdm)
├── evaluate.py        # Publication-grade metrics computation
├── run_devign.py      # CLI entry point
├── requirements.txt   # Python dependencies
└── results/           # Created automatically
    ├── predictions.jsonl   # Per-sample predictions
    ├── checkpoint.txt      # Processed sample IDs (for resume)
    ├── inference.log       # Full inference log
    └── metrics_summary.json
```

---

## Setup

```bash
# Activate the project conda environment
conda activate vulai

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Dry-run (no API calls, inspect prompts)
```bash
conda activate vulai
cd llm_baselines
python run_devign.py --limit 3 --dry-run
```

### 2. Sanity check (10 real samples)
```bash
python run_devign.py --limit 10
```

### 3. Full run (2 796 samples, with auto-resume)
```bash
python run_devign.py --resume
```
> The script will ask for confirmation before running all samples.  
> If interrupted, re-run with `--resume` to skip already-processed samples.

### 4. Evaluate
```bash
python evaluate.py
```

---

## Output Format

`results/predictions.jsonl` — one JSON object per line:

```json
{
  "id":           "81",
  "file_name":    "81",
  "true_label":   1,
  "pred_label":   1,
  "correct":      true,
  "confidence":   88,
  "cwe":          "CWE-119",
  "reason":       "Buffer overflow via unchecked RAM_size parameter.",
  "parse_ok":     true,
  "latency_sec":  1.243,
  "raw_response": "{\"vulnerable\": 1, ...}"
}
```

---

## Metrics (publication table)

| Metric       | Description                              |
|--------------|------------------------------------------|
| Accuracy     | (TP+TN) / All                            |
| Precision    | TP / (TP+FP)                             |
| Recall       | TP / (TP+FN)  — sensitivity              |
| F1-Score     | Harmonic mean of Precision & Recall      |
| MCC          | Matthews Correlation Coefficient         |
| AUC-ROC      | Area under ROC (using confidence scores) |
| FPR          | False Positive Rate                      |
| FNR          | False Negative Rate                      |

---

## Model Configuration

| Parameter    | Value              |
|--------------|--------------------|
| Model        | z-ai/glm-5.2       |
| Temperature  | 0.1                |
| Top-P        | 0.9                |
| Max Tokens   | 512                |
| Seed         | 42                 |
| Platform     | NVIDIA Build / NIM |

> **API key**: Set `NVIDIA_API_KEY` environment variable or edit `config.py`.

---

## Prompt Design

The system uses a two-turn prompt:

1. **System prompt** — primes GLM-5.2 as a *senior security analyst* with 15+ years auditing C/C++ code, covering:
   - Memory safety (buffer overflow, UAF, double-free, null deref)
   - Integer issues (overflow, truncation, off-by-one)
   - Format string vulnerabilities
   - Resource leaks, race conditions
   - Injection & cryptographic weaknesses

2. **User prompt** — structured to elicit a single machine-parseable JSON:
   ```json
   {"vulnerable": 0, "confidence": 75, "cwe": "N/A", "reason": "..."}
   ```

---

## Citation / Attribution

If using these results in a publication, please cite:

```
llm_baselines Baseline – GLM-5.2 via NVIDIA Build Platform
Dataset: Devign (Zhou et al., 2019)
Test split: test_uscp.jsonlines (2 796 samples)
```
