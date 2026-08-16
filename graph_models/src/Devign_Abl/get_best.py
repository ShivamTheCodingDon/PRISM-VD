import os
import csv

base_dir = '/home/user1/AIVul(Don\'t Delete It)/PRISM-VD/PRISM-VD-Enhanced/graph_models/src/Devign_Abl'
results = []
for folder in sorted(os.listdir(base_dir)):
    fpath = os.path.join(base_dir, folder, 'Devign_metrics_history.csv')
    if os.path.exists(fpath):
        best_f1 = -1.0
        best_row = None
        try:
            with open(fpath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    f1 = float(row.get('test_f1', -1))
                    if f1 > best_f1:
                        best_f1 = f1
                        best_row = row
            if best_row:
                results.append((folder, float(best_row['test_f1']), float(best_row['test_acc']), float(best_row['test_precision']), float(best_row['test_recall'])))
        except Exception as e:
            pass

results.sort(key=lambda x: x[1], reverse=True)
print(f'{"Folder":<35} | {"Test F1":<8} | {"Test Acc":<8} | {"Test Pre":<8} | {"Test Rec":<8}')
print('-'*80)
for r in results:
    print(f'{r[0]:<35} | {r[1]:<8.4f} | {r[2]:<8.4f} | {r[3]:<8.4f} | {r[4]:<8.4f}')
