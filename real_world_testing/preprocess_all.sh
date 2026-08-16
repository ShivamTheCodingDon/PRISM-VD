#!/bin/bash
set -e

# Change to the data processing directory where atlas_adapter.py and prepare_data_uscp.py reside
cd "/home/user1/AIVul(Don't Delete It)/PRISM-VD/PRISM-VD-Enhanced/data_processing"

DATASETS=(
    "libav"
    "qemu"
    "xen"
    "openssl"
    "ffmpeg"
)

for DS in "${DATASETS[@]}"; do
    echo "=========================================================="
    echo "Processing $DS..."
    echo "=========================================================="
    
    # Define output directory for this dataset's splits
    OUT_DIR="../real_world_testing/output/${DS}_processed"
    mkdir -p "$OUT_DIR"
    
    # Run the UCG/USCP preparation script
    # We use --fallback_to_cpp to ensure C++ features are handled, and max_workers for speed
    python prepare_data_uscp.py \
        --input "../real_world_testing/output/${DS}_cve_db_auto.jsonlines" \
        --format jsonlines \
        --output_dir "$OUT_DIR" \
        --lang c \
        --fallback_to_cpp \
        --max_workers 16
        
    echo "Combining splits into single eval file..."
    # The script generates train_uscp, valid_uscp, test_uscp, unmapped_uscp because no split map was provided
    # We just combine them all into a single file for evaluation
    cat "$OUT_DIR"/train_uscp.jsonlines \
        "$OUT_DIR"/valid_uscp.jsonlines \
        "$OUT_DIR"/test_uscp.jsonlines \
        "$OUT_DIR"/unmapped_uscp.jsonlines 2>/dev/null > "../real_world_testing/output/${DS}_eval_ready.jsonlines" || true
        
    echo "Finished $DS"
done

echo "All real-world datasets have been preprocessed!"
