"""
Zero-Day Code Snippet Scanner
==============================
Takes a raw C/C++ file (e.g., copied from a website) and scans it for 0-day vulnerabilities
using the trained UCG-VD model.

Usage:
    conda run -n vulai python scan_snippet.py test_code.c
"""

import argparse
import json
import logging
import os
import sys
import time
import torch

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Add necessary paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
UCG_V2_DIR = os.path.join(PROJECT_ROOT, 'graph_models')
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data_processing'))
sys.path.insert(0, UCG_V2_DIR)

# Import necessary modules
from extract_functions import extract_functions_treesitter
from atlas_adapter import parse_code_to_graph_data_uscp
from dataset_dynamic import custom_collate_dynamic
from model_dynamic import Dynamic_PRISM-VD_VD_PlusPlus

# Default weights path
DEFAULT_WEIGHTS = os.path.join(UCG_V2_DIR, 'model_best.pt')

def scan_snippet(file_path: str, weights_path: str, threshold: float = 0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Scanning file: {file_path}")
    if not os.path.exists(file_path):
        logger.error("File not found!")
        return

    # STEP 1: Extract functions
    logger.info("Extracting functions from code...")
    lang = 'cpp' if any(file_path.endswith(e) for e in ('.cpp', '.cc', '.cxx')) else 'c'
    functions = extract_functions_treesitter(file_path, lang=lang)
    
    if not functions:
        logger.warning("No valid C/C++ functions found in the file.")
        return
        
    logger.info(f"Found {len(functions)} function(s).")
    
    # STEP 2: Load Model
    logger.info("Loading UCG-VD Model...")
    model = Dynamic_PRISM-VD_VD_PlusPlus()
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        logger.error(f"Weights not found at {weights_path}")
        return
        
    model.to(device)
    model.eval()

    # STEP 3: Process and Infer each function
    print("\n" + "="*60)
    print(" SCAN RESULTS")
    print("="*60)
    
    for idx, func_obj in enumerate(functions):
        func_name = func_obj.get('func_name', f'unknown_{idx}')
        code = func_obj.get('code', '')
        
        # Generate ATLAS Graph
        try:
            graph_data = parse_code_to_graph_data_uscp(code, lang=lang)
        except Exception as e:
            print(f"[{func_name}] ⚠️ Parsing Error: {str(e)[:50]}")
            continue
            
        if not graph_data.get('uscp_paths'):
            print(f"[{func_name}] ⚠️ Skipped: No USCP paths generated (too simple?)")
            continue
            
        # Create a mock dataset item for the collate function
        item = {
            'graph_data': graph_data,
            'label': 0
        }
        
        # Use the custom collate function to build the batch
        batch = custom_collate_dynamic([item])
        if not batch:
            print(f"[{func_name}] ⚠️ Data Loader Error")
            continue
            
        input_ids, config_data, _ = batch
        input_ids = input_ids.to(device)
        
        # Inference
        with torch.no_grad():
            logits, _ = model(input_ids, config_data)
            prob = torch.sigmoid(logits).item()
            
        # Report
        if prob >= threshold:
            print(f"[{func_name}] 🚨 VULNERABLE! (Confidence: {prob*100:.1f}%)")
        else:
            print(f"[{func_name}] ✅ Safe (Confidence: {(1-prob)*100:.1f}%)")
            
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scan arbitrary C/C++ snippets for 0-days")
    parser.add_argument("file", type=str, help="Path to the C/C++ file to scan")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.5, help="Vulnerability confidence threshold")
    args = parser.parse_args()
    
    scan_snippet(args.file, args.weights, args.threshold)
