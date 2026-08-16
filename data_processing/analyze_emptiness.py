import json
import os
import glob
from collections import defaultdict

def analyze_emptiness(dataset_dir, dataset_name, file_suffix):
    print(f"\\n{'='*60}")
    print(f"Analyzing {dataset_name} - {file_suffix.upper()} at {dataset_dir}")
    print(f"{'='*60}")
    
    files = glob.glob(os.path.join(dataset_dir, f'*_{file_suffix}.jsonlines'))
    if not files:
        print(f"No _{file_suffix}.jsonlines found in {dataset_dir}")
        return
        
    counts = defaultdict(lambda: {
        'empty_benign': 0,
        'empty_vuln': 0,
        'present_benign': 0,
        'present_vuln': 0,
    })
    
    total_benign = 0
    total_vuln = 0
    
    for file_path in files:
        print(f"Reading {os.path.basename(file_path)}...")
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    label = int(data.get('label', 0))
                    
                    if label == 1:
                        total_vuln += 1
                        is_vuln = True
                    else:
                        total_benign += 1
                        is_vuln = False
                        
                    graph_data = data.get('graph_data', {})
                    
                    for key, val in graph_data.items():
                        if key == 'nodes':
                            continue  # usually skipping nodes, but can be removed if nodes emptiness is desired
                        
                        if not val: # empty list or None
                            if is_vuln:
                                counts[key]['empty_vuln'] += 1
                            else:
                                counts[key]['empty_benign'] += 1
                        else:
                            if is_vuln:
                                counts[key]['present_vuln'] += 1
                            else:
                                counts[key]['present_benign'] += 1
                except json.JSONDecodeError:
                    continue
                    
    print(f"\\nOverall Totals:\\n  Vulnerable: {total_vuln}\\n  Benign: {total_benign}\\n  Total: {total_vuln + total_benign}")
    
    print("\\nDetailed analysis per graph type:\\n")
    for key, c in counts.items():
        empty_b = c['empty_benign']
        empty_v = c['empty_vuln']
        pres_b = c['present_benign']
        pres_v = c['present_vuln']
        
        total_b = empty_b + pres_b
        total_v = empty_v + pres_v
        
        ratio_b = (empty_b / total_b * 100) if total_b > 0 else 0.0
        ratio_v = (empty_v / total_v * 100) if total_v > 0 else 0.0
        
        empty_total = empty_b + empty_v
        total_all = total_b + total_v
        ratio_total = (empty_total / total_all * 100) if total_all > 0 else 0.0
        
        print(f"--- {key.upper()} ---")
        print(f"  Empty   | Benign: {empty_b} | Vuln: {empty_v} | Total: {empty_total}")
        print(f"  Present | Benign: {pres_b} | Vuln: {pres_v} | Total: {total_all - empty_total}")
        print(f"  Empty Ratios | Benign: {ratio_b:.2f}% | Vuln: {ratio_v:.2f}% | Overall: {ratio_total:.2f}%")
        print()


def main():
    datasets = {
        'BigVul': '/home/azure/PRISM-VD/PRISM-VD-Enhanced/data/processed/BigVul',
        'Reveal': '/home/azure/PRISM-VD/PRISM-VD-Enhanced/data/processed/Reveal',
        'Devign': '/home/azure/PRISM-VD/PRISM-VD-Enhanced/data/processed/Devign'
    }
    
    for name, path in datasets.items():
        analyze_emptiness(path, name, 'causal')
        analyze_emptiness(path, name, 'uscp')

if __name__ == '__main__':
    main()
