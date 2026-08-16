#!/bin/bash
# Process all generated JSONL datasets: Generate Graphs -> Run Inference

# Define your best trained model weights here!
WEIGHTS="/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/model_best.pt" 
BASE_OUT="/media/user1/One Touch1/00 Data/realdatavul"

cd "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced"

# Wait for ATLAS initialization before starting the batch job
python3 data_processing/wait_for_atlas.py
if [ $? -ne 0 ]; then
    echo "ATLAS failed to initialize. Exiting."
    exit 1
fi

# Loop through all generated JSONLines datasets
for json_file in real_world_testing/output/*_cve_db_auto.jsonlines; do
    
    # Extract project name
    filename=$(basename "$json_file")
    project="${filename%_cve_db_auto.jsonlines}"
    
    echo "======================================================"
    echo "  PROCESSING REPOSITORY: $project"
    echo "======================================================"
    
    # Output paths
    graph_out_dir="$BASE_OUT/${project}_graphs"
    results_dir="$BASE_OUT/${project}_results"
    
    mkdir -p "$graph_out_dir"
    
    # STEP 1: Generate USCP Graphs using the official script with retries
    echo "[1/2] Generating ATLAS Graphs for $project..."
    
    # Remove old failure log if we want to retry all
    rm -f "$graph_out_dir/failed_uscp.txt"
    
    conda run -n vulai python data_processing/prepare_data_uscp.py \
        --input "$json_file" \
        --format jsonlines \
        --output_dir "$graph_out_dir" \
        --text_col code \
        --label_col label \
        --lang c \
        --fallback_to_cpp \
        --retry_all \
        --no_wait \
        --skip_empty False \
        --max_workers 25
        
    # Real-world functions not mapped to train/val/test go to unmapped_uscp.jsonlines
    graph_file="$graph_out_dir/unmapped_uscp.jsonlines"
    
    # Check if graphs were generated successfully
    if [ ! -s "$graph_file" ]; then
        echo "Error: Graph generation failed or produced 0 graphs for $project. Skipping inference."
        continue
    fi
    
    # STEP 2: Run UCG-VD Inference
    echo "[2/2] Running Inference for $project..."
    conda run -n vulai python real_world_testing/infer_nday.py \
        --test_data "$graph_file" \
        --weights_path "$WEIGHTS" \
        --output_dir "$results_dir"
        
    echo "✔ Finished $project! Report saved to: $results_dir/nday_report.txt"
    echo ""
    
done

echo "🎉 ALL REPOSITORIES PROCESSED!"
