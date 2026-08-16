import os
import json
import logging
import sys
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def deduplicate_dataset(dataset_dir, suffixes):
    """
    Deduplicate all files in a dataset directory by maintaining a global set of seen CODE HASHES.
    This handles duplicates that may exist across different splits or multiple runs.
    """
    if not os.path.isdir(dataset_dir):
        return
    
    splits = ["train", "valid", "test", "unmapped"]
    
    for suffix in suffixes:
        seen_hashes = set()
        total_dups = 0
        total_unique = 0
        
        logger.info(f"--- Deduplicating {os.path.basename(dataset_dir)} ({suffix}) by Code Content ---")
        
        for split in splits:
            file_name = f"{split}{suffix}"
            file_path = os.path.join(dataset_dir, file_name)
            if not os.path.exists(file_path):
                continue
                
            temp_path = file_path + ".tmp"
            file_dup_count = 0
            file_unique_count = 0
            
            with open(file_path, 'r', encoding='utf-8') as fin, open(temp_path, 'w', encoding='utf-8') as fout:
                for line in fin:
                    if not line.strip():
                        continue
                    try:
                        # Full JSON parse to get the "code" field reliably
                        data = json.loads(line)
                        code = data.get('code', '')
                        
                        # Generate SHA-256 of the code content
                        code_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
                        
                        if code_hash not in seen_hashes:
                            seen_hashes.add(code_hash)
                            fout.write(line)
                            file_unique_count += 1
                        else:
                            file_dup_count += 1
                            
                    except Exception as e:
                        logger.error(f"Error parsing line in {file_path}: {e}")
                        continue
            
            os.replace(temp_path, file_path)
            logger.info(f"  {file_name}: Unique: {file_unique_count}, Duplicates: {file_dup_count}")
            total_dups += file_dup_count
            total_unique += file_unique_count
            
        logger.info(f"Finished {os.path.basename(dataset_dir)} ({suffix}). Total Unique: {total_unique}, Total Removed: {total_dups}")

def main():
    base_dir = "/home/azure/PRISM-VD/PRISM-VD-Enhanced"
    processed_dir = os.path.join(base_dir, "data", "processed")
    
    datasets = ["BigVul", "Devign", "Reveal"]
    suffixes = ["_causal.jsonlines", "_uscp.jsonlines"]
    
    for dataset in datasets:
        dataset_dir = os.path.join(processed_dir, dataset)
        deduplicate_dataset(dataset_dir, suffixes)
        
    logger.info("Code-level deduplication phase complete.")

if __name__ == "__main__":
    main()
