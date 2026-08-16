#!/bin/bash

# Define the models you want to evaluate here. 
# These names must match your Azure deployment names.
MODELS=("Codestral-Agent" "gpt-4.1-2" "DeepSeek-V3.2")

for MODEL in "${MODELS[@]}"; do
    # Override the MODEL_NAME from config.py
    export MODEL_NAME="$MODEL"
    OUT_DIR="results_$MODEL"
    
    echo "========================================================"
    echo "🤖 Running inference for model: $MODEL_NAME"
    echo "📁 Saving results to: $OUT_DIR"
    echo "========================================================"
    
    # 1. Run Inference
    python run_devign.py --output-dir "$OUT_DIR"
    
    # 2. Run Evaluation
    if [ -f "$OUT_DIR/predictions.jsonl" ]; then
        echo "📊 Evaluating metrics for $MODEL_NAME..."
        python evaluate.py --predictions "$OUT_DIR/predictions.jsonl" --save "$OUT_DIR/metrics_summary.json"
    else
        echo "⚠️ WARNING: No predictions found for $MODEL_NAME. Skipping evaluation."
    fi
done

echo "✅ All models processed!"
