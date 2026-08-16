#!/bin/bash

# Datasets to evaluate
DATASETS=("qemu" "xen" "ffmpeg" "openssl" "libav")

# Define the models to evaluate
MODELS=("Codestral-Agent" "gpt-4.1-2" "DeepSeek-V3.2")

for MODEL in "${MODELS[@]}"; do
    export MODEL_NAME="$MODEL"
    BASE_OUT_DIR="results_$MODEL"

    echo "========================================================"
    echo "🤖 Running inference for model: $MODEL_NAME"
    echo "========================================================"

    for DS in "${DATASETS[@]}"; do
        DS_FILE="../output/${DS}_cve_db_auto.jsonlines"
        OUT_DIR="${BASE_OUT_DIR}/${DS}"
        
        echo "--------------------------------------------------------"
        echo "📂 Processing dataset: $DS"
        echo "📁 Saving results to: $OUT_DIR"
        echo "--------------------------------------------------------"
        
        if [ ! -f "$DS_FILE" ]; then
            echo "⚠️ ERROR: Dataset file $DS_FILE not found! Skipping..."
            continue
        fi

        # 1. Run Inference
        python run_eval.py --dataset "$DS_FILE" --output-dir "$OUT_DIR" --resume
        
        # 2. Run Evaluation
        if [ -f "$OUT_DIR/predictions.jsonl" ]; then
            echo "📊 Evaluating metrics for $MODEL_NAME on $DS..."
            python evaluate.py --predictions "$OUT_DIR/predictions.jsonl" --save "$OUT_DIR/metrics_summary.json"
        else
            echo "⚠️ WARNING: No predictions found for $MODEL_NAME on $DS. Skipping evaluation."
        fi
    done
done

echo "✅ All datasets processed across all models!"
