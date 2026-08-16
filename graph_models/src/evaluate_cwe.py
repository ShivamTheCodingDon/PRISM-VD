import argparse
import logging
import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train_graph_models import evaluate
from model_ucg import UCG_PRISM-VD_VD
from dataset_graph_models import UCGCodeGraphDatasetV2, custom_collate_ucg
from dataset_dynamic import EDGE_TYPE_COUNTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TARGET_CWES = [
    "CWE-787", "CWE-416", "CWE-78", "CWE-20", "CWE-125",
    "CWE-476", "CWE-190", "CWE-94", "CWE-119", "CWE-400"
]

def ensure_full_cwe_data(target_cwes, bigvul_dir):
    """
    Generates the full dataset (combining all splits) for the target CWEs if they don't already exist.
    Reads from the main BigVul jsonlines to extract lines for these specific CWEs.
    """
    full_dir = os.path.join(bigvul_dir, "CWE_classes_full_data")
    os.makedirs(full_dir, exist_ok=True)
    
    missing_cwes = [cwe for cwe in target_cwes if not os.path.exists(os.path.join(full_dir, f"full_uscp_{cwe}.jsonlines"))]
    if not missing_cwes:
        return full_dir
        
    logger.info(f"Generating full dataset for missing CWEs: {missing_cwes}...")
    cwe_to_ids_path = os.path.join(bigvul_dir, "cwe_to_ids_split.json")
    
    if not os.path.exists(cwe_to_ids_path):
        logger.error(f"Cannot find {cwe_to_ids_path} to generate full datasets!")
        sys.exit(1)
        
    with open(cwe_to_ids_path, 'r') as f:
        cwe_to_ids = json.load(f)
        
    # Build a lookup dictionary for fast matching: ID -> CWE
    id_to_missing_cwe = {}
    for cwe in missing_cwes:
        splits = cwe_to_ids.get(cwe, {})
        for s, ids in splits.items():
            for i in ids:
                id_to_missing_cwe[str(i)] = cwe
                
    # Open file handles for writing
    handles = {cwe: open(os.path.join(full_dir, f"full_uscp_{cwe}.jsonlines"), 'w') for cwe in missing_cwes}
    
    # Do one pass through the big files
    for split_name in ['train_uscp', 'valid_uscp', 'test_uscp', 'unmapped_uscp']:
        split_file = os.path.join(bigvul_dir, f"{split_name}.jsonlines")
        if not os.path.exists(split_file): 
            continue
            
        logger.info(f"Extracting records from {split_file}...")
        with open(split_file, 'r') as in_f:
            for line in in_f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    cwe_of_line = id_to_missing_cwe.get(str(data.get('id')))
                    if cwe_of_line:
                        handles[cwe_of_line].write(line)
                except Exception:
                    pass
                    
    for h in handles.values():
        h.close()
        
    return full_dir

def main():
    parser = argparse.ArgumentParser(description="Evaluate specific CWE classes using a trained model.")
    parser.add_argument('--resume_path', type=str, required=True, help="Path to the trained .pt model file")
    
    # ── NEW MODE ARGUMENT ──
    parser.add_argument('--mode', type=str, choices=['test', 'full'], default='test',
                        help="Evaluate on 'test' split only, or 'full' dataset for the CWE")
                        
    parser.add_argument('--bigvul_dir', type=str, 
                        default="/media/user1/One Touch1/00 Data/PRISM-VD/data/processed/BigVul",
                        help="Base directory containing the datasets and mappings")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5, help="Classification threshold")
    
    # Model architecture arguments
    parser.add_argument('--gnn', type=str, default='rgat')
    parser.add_argument('--slice_method', type=str, default='cta_rwr')
    parser.add_argument('--no_fexpn', action='store_true', help="Disable fexpn")
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--fusion', type=str, default='gated')
    parser.add_argument('--pooling', type=str, default='attention')
    parser.add_argument('--edge_num', type=int, default=11)
    parser.add_argument('--context_ratio', type=float, default=0.5)
    parser.add_argument('--max_guards', type=int, default=5)
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"EVALUATION MODE: {args.mode.upper()}")

    # Determine paths based on mode
    if args.mode == 'full':
        # Generate full datasets if they don't exist
        cwe_dir = ensure_full_cwe_data(TARGET_CWES, args.bigvul_dir)
        file_prefix = "full_uscp_"
    else:
        # Default test split
        cwe_dir = os.path.join(args.bigvul_dir, "CWE_classes_test_data")
        file_prefix = "test_uscp_"

    # 1. Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    num_edge_types = EDGE_TYPE_COUNTS.get(args.edge_num, 35)
    
    logger.info("Initializing UCG_PRISM-VD_VD model...")
    model = UCG_PRISM-VD_VD(
        model_name='microsoft/codebert-base', embed_dim=256, num_edge_types=num_edge_types,
        gnn_type=args.gnn, fusion_type=args.fusion, pool_type=args.pooling,
        num_layers=args.num_layers, max_edges=6000
    ).to(device)

    # 2. Load weights
    logger.info(f"Loading weights from {args.resume_path}...")
    try:
        model.load_state_dict(torch.load(args.resume_path, map_location=device))
    except Exception as e:
        logger.error(f"Failed to load weights: {e}")
        sys.exit(1)
        
    model.eval()
    results = []

    # 3. Evaluate each CWE dataset
    for cwe in TARGET_CWES:
        target_file = os.path.join(cwe_dir, f"{file_prefix}{cwe}.jsonlines")
        
        if not os.path.exists(target_file):
            logger.warning(f"Target file not found for {cwe}: {target_file}")
            continue
            
        logger.info(f"==========================================")
        logger.info(f"Evaluating {cwe}...")
        
        try:
            # Create dataset and loader
            test_ds = UCGCodeGraphDatasetV2(
                tokenizer, target_file, npy_dir=None,
                slice_method=args.slice_method, 
                fexpn=not args.no_fexpn,
                edge_num=args.edge_num,
                context_ratio=args.context_ratio,
                max_guards_per_path=args.max_guards
            )
            
            if len(test_ds) == 0:
                logger.warning(f"No valid graph samples in {cwe} dataset. Skipping.")
                continue

            test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=custom_collate_ucg)

            # Evaluate (train_graph_models.py evaluate returns 9 items)
            acc, rec, prec, f1, auc, mcc, cm, thr, t_loss = evaluate(
                model, test_loader, device, criterion=None,
                precision_guard=0.0, use_mcc=True, fixed_threshold=args.threshold
            )
            
            # Store and display results
            results.append({
                "CWE": cwe,
                "Mode": args.mode.upper(),
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "AUC": auc,
                "Samples": len(test_ds)
            })
            
            logger.info(f"[{cwe}] F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Acc: {acc:.4f} (Samples: {len(test_ds)})")
            
        except Exception as e:
            logger.error(f"Error evaluating {cwe}: {e}")

    # 4. Print final table and save to CSV
    if results:
        df = pd.DataFrame(results)
        print("\n" + "="*80)
        print(f"FINAL CWE EVALUATION RESULTS — {args.mode.upper()} DATASET")
        print("="*80)
        print(df.to_markdown(index=False))
        print("="*80)
        
        out_csv = f"cwe_evaluation_results_{args.mode}.csv"
        df.to_csv(out_csv, index=False)
        print(f"\nSaved results to {os.path.abspath(out_csv)}\n")
    else:
        logger.error("No results were generated. Ensure files exist and are valid.")

if __name__ == '__main__':
    main()
