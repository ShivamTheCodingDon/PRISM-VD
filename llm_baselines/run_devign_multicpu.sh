#!/bin/bash
# run_devign_multicpu.sh
# 
# Automatically run the llm_baselines vulnerability detection using multi-threading.
# By default, it will use 15 concurrent workers (as defined in config.py),
# taking advantage of your 30 CPUs to process requests in parallel.
#
# IMPORTANT: Before running this script, ensure you have activated your conda environment.
# Example: conda activate vulai

# Exit on error
set -e

echo "======================================================"
echo " Starting Multi-CPU llm_baselines Inference Pipeline"
echo "======================================================"

# Navigate to the llm_baselines directory if not already there
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

# Ensure we're running the python from the active environment
PYTHON_CMD="python3"

echo "Checking if langchain-nvidia-ai-endpoints is installed..."
$PYTHON_CMD -c "import langchain_nvidia_ai_endpoints" || {
    echo "Required packages are missing. Please run: pip install -r requirements.txt"
    exit 1
}

echo "Starting parallel inference..."
# We run without limit and resume by default
$PYTHON_CMD run_devign.py --resume

echo "Inference complete. Running evaluation..."
$PYTHON_CMD evaluate.py

echo "======================================================"
echo " Pipeline Complete."
echo "======================================================"
