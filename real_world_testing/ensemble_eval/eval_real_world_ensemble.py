import os
import sys
import json
import glob
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add paths to load model and dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))) # PRISM-VD
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../graph_models/src"))) # ucg

from model_ucg import UCG_PRISM-VD_VD
from dataset_graph_models import UCGCodeGraphDatasetV2, custom_collate_ucg
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

def evaluate_real_world_ensemble():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    model_configs = {
        "Reveal": {
            "path": "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_reveal_ucg_rgat_cta_rwr_maxg7",
            "slice_method": "cta_rwr",
            "max_guards_per_path": 7,
            "context_mode": "random",
            "context_ratio": 0.5,
            "num_bases": None
        },
        "Devign": {
            "path": "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_devign_ucg_rgat_vpc_smoth0.2",
            "slice_method": "vpc",
            "max_guards_per_path": 5,
            "context_mode": "random",
            "context_ratio": 0.5,
            "num_bases": None
        },
        "BigVul": {
            "path": "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_bigvul_ucg_rgat_rwr_wbce_pw3",
            "slice_method": "rwr", # Fallback to rwr based on the folder name
            "max_guards_per_path": 5,
            "context_mode": "hop",
            "context_ratio": 0.5,
            "num_bases": 4
        }
    }
    
    models = {}
    
    for name, config in model_configs.items():
        print(f"Loading {name} model from: {config['path']}")
        model = UCG_PRISM-VD_VD(
            model_name="microsoft/codebert-base",
            embed_dim=256,
            gnn_type="rgat",
            num_layers=1,
            dropout=0.3,
            fusion_type="gated",
            pool_type="attention",
            num_edge_types=11,
            num_bases=config["num_bases"]
        )
        model.load_state_dict(torch.load(os.path.join(config['path'], "model_best_f1.pt"), map_location='cpu'))
        # model.to(device) # Keep on CPU until needed
        model.eval()
        models[name] = model
        
    real_world_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../output"))
    datasets = glob.glob(os.path.join(real_world_dir, "*_eval_ready.jsonlines"))
    
    all_metrics = {}
    
    for dataset_path in datasets:
        dataset_name = os.path.basename(dataset_path).replace("_eval_ready.jsonlines", "")
        print(f"\n=========================================")
        print(f"Evaluating dataset: {dataset_name}")
        print(f"=========================================")
        
        # Read the samples info to map CVE IDs and labels
        base_name = dataset_name.replace("_eval_ready", "")
        cve_map = {}
        cve_db_path = os.path.join(real_world_dir, f"{base_name}_cve_db_auto.jsonlines")
        if os.path.exists(cve_db_path):
            with open(cve_db_path, 'r') as f:
                for line in f:
                    if not line.strip(): continue
                    orig_record = json.loads(line)
                    if "id" in orig_record and "cve_id" in orig_record:
                        cve_map[str(orig_record["id"])] = orig_record["cve_id"]

        samples_info = []
        with open(dataset_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                record = json.loads(line)
                rec_id = str(record.get("id", ""))
                samples_info.append({
                    "id": rec_id,
                    "cve_id": cve_map.get(rec_id, "UNKNOWN_CVE"),
                    "true_label": int(record.get("label", 0))
                })
        
        if not samples_info:
            print(f"No samples found in {dataset_path}")
            continue
            
        # Dictionary to hold predictions for each model. Key: model_name, Value: dict mapping sample idx -> pred
        model_predictions = {name: {} for name in model_configs.keys()}
        
        for name, config in model_configs.items():
            print(f"Running inference for model: {name} (slice: {config['slice_method']}, max_guards: {config['max_guards_per_path']})")
            dataset = UCGCodeGraphDatasetV2(
                tokenizer,
                dataset_path,
                block_size=512,
                slice_method=config["slice_method"],
                ignore_empty_cfg=True,
                fexpn=True,
                context_mode=config["context_mode"],
                context_ratio=config["context_ratio"],
                max_guards_per_path=config["max_guards_per_path"],
                edge_num=11
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=4,
                shuffle=False,
                collate_fn=custom_collate_ucg,
                num_workers=4
            )
            
            idx = 0
            model = models[name]
            model.to(device)
            with torch.no_grad():
                for batch in tqdm(dataloader, desc=f"{name} Eval"):
                    if batch is None:
                        continue
                    input_ids, config_data, labels = batch
                    input_ids = input_ids.to(device)
                    
                    logits, *_ = model(input_ids, config_data)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    preds = (probs > 0.5).astype(int)
                    
                    for pred in preds:
                        if idx < len(samples_info):
                            model_predictions[name][idx] = pred
                        idx += 1
                        
            model.to('cpu')
            torch.cuda.empty_cache()
                        
        print(f"Aggregating ensemble predictions...")
        
        dataset_correct = 0
        dataset_total = len(samples_info)
        
        cve_stats = defaultdict(lambda: {"correct": 0, "total": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0, "actual_vul": 0, "pred_vul": 0})
        dataset_tp, dataset_fp, dataset_tn, dataset_fn = 0, 0, 0, 0
        dataset_actual_vul, dataset_pred_vul = 0, 0
        
        for idx in range(len(samples_info)):
            sample = samples_info[idx]
            true_label = sample["true_label"]
            cve = sample["cve_id"]
            
            votes = 0
            for name in model_configs.keys():
                votes += model_predictions[name].get(idx, 0)
                
            pred_label = 1 if votes >= 2 else 0
            is_correct = (pred_label == true_label)
            
            if true_label == 1:
                dataset_actual_vul += 1
                cve_stats[cve]["actual_vul"] += 1
            if pred_label == 1:
                dataset_pred_vul += 1
                cve_stats[cve]["pred_vul"] += 1
                
            if pred_label == 1 and true_label == 1:
                dataset_tp += 1
                cve_stats[cve]["tp"] += 1
            elif pred_label == 1 and true_label == 0:
                dataset_fp += 1
                cve_stats[cve]["fp"] += 1
            elif pred_label == 0 and true_label == 0:
                dataset_tn += 1
                cve_stats[cve]["tn"] += 1
            elif pred_label == 0 and true_label == 1:
                dataset_fn += 1
                cve_stats[cve]["fn"] += 1
            
            if is_correct:
                dataset_correct += 1
                cve_stats[cve]["correct"] += 1
                
            cve_stats[cve]["total"] += 1

        accuracy = dataset_correct / dataset_total if dataset_total > 0 else 0
        precision = dataset_tp / (dataset_tp + dataset_fp) if (dataset_tp + dataset_fp) > 0 else 0
        recall = dataset_tp / (dataset_tp + dataset_fn) if (dataset_tp + dataset_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{dataset_name.upper()} Dataset Metrics (ENSEMBLE):")
        print(f"  Accuracy : {accuracy:.4f} ({dataset_correct}/{dataset_total})")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall   : {recall:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  Actual Vulns: {dataset_actual_vul} | Predicted Vulns: {dataset_pred_vul}")
        print(f"  TP: {dataset_tp} | FP: {dataset_fp} | TN: {dataset_tn} | FN: {dataset_fn}")
        
        sorted_cves = sorted(cve_stats.items(), key=lambda x: x[1]["total"], reverse=True)
        cve_metrics = {}
        for cve, stat in sorted_cves:
            cve_acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
            c_prec = stat["tp"] / (stat["tp"] + stat["fp"]) if (stat["tp"] + stat["fp"]) > 0 else 0
            c_rec = stat["tp"] / (stat["tp"] + stat["fn"]) if (stat["tp"] + stat["fn"]) > 0 else 0
            c_f1 = 2 * c_prec * c_rec / (c_prec + c_rec) if (c_prec + c_rec) > 0 else 0
            
            cve_metrics[cve] = {
                "correct": stat["correct"],
                "total": stat["total"],
                "accuracy": cve_acc,
                "precision": c_prec,
                "recall": c_rec,
                "f1": c_f1,
                "actual_vul": stat["actual_vul"],
                "pred_vul": stat["pred_vul"],
                "tp": stat["tp"],
                "fp": stat["fp"],
                "tn": stat["tn"],
                "fn": stat["fn"]
            }
            if stat["total"] > 1:
                print(f"  {cve}: Acc {cve_acc:.4f} ({stat['correct']}/{stat['total']}) | Actual Vul: {stat['actual_vul']} | Pred Vul: {stat['pred_vul']} | TP: {stat['tp']} FP: {stat['fp']} TN: {stat['tn']} FN: {stat['fn']}")
                
        all_metrics[dataset_name] = {
            "total_correct": dataset_correct,
            "total_samples": dataset_total,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "actual_vul": dataset_actual_vul,
            "pred_vul": dataset_pred_vul,
            "tp": dataset_tp,
            "fp": dataset_fp,
            "tn": dataset_tn,
            "fn": dataset_fn,
            "cve_breakdown": cve_metrics
        }
        
    output_json = os.path.join(os.path.dirname(__file__), "ensemble_metrics_summary.json")
    with open(output_json, "w") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nSaved detailed ensemble metrics to {output_json}")

if __name__ == "__main__":
    evaluate_real_world_ensemble()
