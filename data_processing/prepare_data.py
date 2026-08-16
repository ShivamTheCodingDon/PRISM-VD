import json
import logging
import os
import argparse
import pandas as pd
from tqdm import tqdm
from atlas_adapter import parse_code_to_graph_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def worker_process_entry(task_args):
    """
    Worker function to process a single sample in a separate process.
    """
    (entry, text_col, label_col, lang, fallback_to_cpp, retry_all, 
     example_idx, file_name, mapping_key, assigned_split) = task_args
    
    import time
    from atlas_adapter import parse_code_to_graph_data
    
    code = entry.get(text_col, entry.get('func', ''))
    
    # Coerce label: check for explicit override first (e.g. inferred from filename)
    label_override = entry.get('label_override')
    if label_override is not None:
        label = int(label_override)
    else:
        raw_label = entry.get(label_col, entry.get('target', 0))
        try:
            label = int(raw_label)
        except:
            label = 0
            
    row_lang = 'unknown'
    try:
        # Extract lang from dataset row if present, normalize to "c" or "cpp"
        row_lang = entry.get('lang', lang)
        if isinstance(row_lang, str):
            row_lang = row_lang.lower().strip()
            row_lang = 'cpp' if row_lang in ['c++', 'cpp', 'cc', 'cxx'] else 'c'
        else:
            row_lang = lang if lang in ['c', 'cpp'] else 'c'

        # Generate ATLAS Causal Graph with fallback retry mechanism
        start_t = time.time()
        try:
            graph_data = parse_code_to_graph_data(code, lang=row_lang)
            
            # 2026 Optimization: Aggressive Retry on Empty CFG (Section 4.1 Improvement)
            # If parse succeeded but resulted in 0 edges, it's likely a C/C++ mismatch (e.g. :: or new)
            if not graph_data.get('cfg_edges') and (fallback_to_cpp or retry_all):
                fallback_lang = 'cpp' if row_lang == 'c' else 'c'
                graph_data = parse_code_to_graph_data(code, lang=fallback_lang)
                row_lang = fallback_lang
        except Exception as atlas_err:
            if fallback_to_cpp or retry_all:
                fallback_lang = 'cpp' if row_lang == 'c' else 'c'
                graph_data = parse_code_to_graph_data(code, lang=fallback_lang)
                row_lang = fallback_lang 
            else:
                raise atlas_err
        
        elapsed_t = time.time() - start_t
        
        output_obj = {
            "id": example_idx,
            "file_name": file_name,
            "code": code,
            "label": label,
            "processing_time_sec": round(elapsed_t, 4),
            "graph_data": graph_data
        }
        return {"status": "success", "data": output_obj, "split": assigned_split, "lang": row_lang}
    except Exception as e:
        from atlas_adapter import WorkerTimeoutError
        # Check if it's a hard signal timeout or a soft exception
        err_msg = str(e)
        if isinstance(e, WorkerTimeoutError):
            err_msg = "HARD_TIMEOUT_SIGNAL_ALARM (60s)"
        return {"status": "error", "id": example_idx, "mapping_key": mapping_key, "split": assigned_split, "error": err_msg, "code": code}


def prepare_dataset(
    input_data_path: str, 
    output_dir: str, 
    split_file_path: str = None, 
    split_key_col: str = 'example_index', 
    split_val_col: str = 'split',
    text_col: str = 'code',
    label_col: str = 'label',
    format: str = 'jsonlines',
    lang: str = 'c',
    limit: int = None,
    fallback_to_cpp: bool = False,
    retry_all: bool = False,
    no_wait: bool = False,
    max_workers: int = None,
    save_individual_graphs: bool = False,
    skip_empty: bool = True
):
    """
    Reads raw dataset, applies train/val/test splits based on an external mapping, 
    and generates ATLAS causal graphs outputting into model-ready jsonlines.
    """
    os.makedirs(output_dir, exist_ok=True)
    import time
    global_start_t = time.time()
    
    # ATLAS Initialization Warm-up skipped (moved to wait_for_atlas.py)
    logger.info("Starting data preparation.")
    
    if not isinstance(input_data_path, list):
        input_data_path = [input_data_path]
    
    # Load dataset
    # Tracking for resuming via SHA256 of code content (stable across restarts for all dataset formats)
    import hashlib as _hashlib
    processed_ids = set()  # stores sha256(code) hashes
    for val in ['train', 'valid', 'test', 'unmapped']:
        check_path = os.path.join(output_dir, f'{val}_causal.jsonlines')
        if os.path.exists(check_path):
            logger.info(f"Indexing finished code hashes from {check_path}...")
            with open(check_path, 'r', encoding='utf-8') as cf:
                for line in cf:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            code = data.get('code', '')
                            if code:
                                processed_ids.add(_hashlib.sha256(code.encode('utf-8')).hexdigest())
                        except:
                            pass
                            
    failed_log_path = os.path.join(output_dir, 'failed_causal.txt')
    # failed log stores code hashes too (updated below)
    if os.path.exists(failed_log_path):
        with open(failed_log_path, 'r', encoding='utf-8') as ff:
            for line in ff:
                if line.strip():
                    processed_ids.add(line.strip())
                    
    if processed_ids:
        logger.info(f"Resume Mode: Found {len(processed_ids)} already processed code hashes. Skipping them during file stream.")

    # Load split map if provided
    split_map = {}
    if split_file_path:
        logger.info(f"Loading split map from {split_file_path}")
        if split_file_path.endswith('.csv'):
            split_df = pd.read_csv(split_file_path)
        elif split_file_path.endswith('.xlsx'):
            split_df = pd.read_excel(split_file_path)
        else:
            raise ValueError("Split file should be .csv or .xlsx")
        split_map = dict(zip(split_df[split_key_col].astype(str), split_df[split_val_col]))

    # Setup output files
    out_files = {
        'train': open(os.path.join(output_dir, 'train_causal.jsonlines'), 'a', encoding='utf-8'),
        'valid': open(os.path.join(output_dir, 'valid_causal.jsonlines'), 'a', encoding='utf-8'),
        'test': open(os.path.join(output_dir, 'test_causal.jsonlines'), 'a', encoding='utf-8'),
        'unmapped': open(os.path.join(output_dir, 'unmapped_causal.jsonlines'), 'a', encoding='utf-8')
    }
    
    if save_individual_graphs:
        for split_name in ['train', 'valid', 'test', 'unmapped']:
            os.makedirs(os.path.join(output_dir, "individual_graphs", split_name), exist_ok=True)

    success_count = 0
    err_count = 0
    skip_count = 0
    
    # Pre-import to speed up loop
    import hashlib
    
    # Task Generator for True Streaming
    def task_generator(pbar_ref=None):
        nonlocal skip_count
        count_found = 0
        
        # Decide which column to use for ID check to avoid .get() overhead
        id_col = 'example_index'
        
        for path in input_data_path:
            if format == 'csv':
                # Deterministic check for available columns to avoid slow .get()
                sample_chunk = pd.read_csv(path, nrows=1)
                cols = sample_chunk.columns.tolist()
                id_col = 'example_index' if 'example_index' in cols else ('id' if 'id' in cols else ('Unnamed: 0' if 'Unnamed: 0' in cols else None))
                
                for chunk in pd.read_csv(path, chunksize=5000):
                    for _, entry in chunk.iterrows():
                        # Use SHA256 of code as stable skip key
                        raw_code = str(entry.get(text_col, entry.get('func', '')))
                        code_hash = _hashlib.sha256(raw_code.encode('utf-8')).hexdigest()

                        if id_col:
                            example_idx = str(entry[id_col])
                        else:
                            example_idx = str(count_found + skip_count)

                        if code_hash in processed_ids:
                            skip_count += 1
                            if pbar_ref:
                                pbar_ref.update(1)
                                if skip_count % 1000 == 0:
                                    success_rate = (success_count / (success_count + err_count)) * 100 if (success_count + err_count) > 0 else 100
                                    pbar_ref.set_postfix({
                                        "Done": success_count, 
                                        "Skipped": skip_count, 
                                        "Error": err_count,
                                        "Success": f"{success_rate:.1f}%"
                                    }, refresh=False)
                            continue
                        
                        count_found += 1
                        file_name = str(entry.get('file_name', entry.get('folder_name', example_idx)))
                        mapping_key = example_idx if split_key_col == 'example_index' else file_name
                        
                        # Split assignment
                        if split_map:
                            assigned_split = split_map.get(mapping_key, 'unmapped')
                        else:
                            hash_val = int(hashlib.md5((mapping_key + "vd_seed").encode()).hexdigest(), 16)
                            prob = (hash_val % 100) / 100.0
                            if prob < 0.8: assigned_split = 'train'
                            elif prob < 0.9: assigned_split = 'valid'
                            else: assigned_split = 'test'
                        
                        if assigned_split not in out_files: assigned_split = 'unmapped'
                        
                        yield (entry.to_dict(), text_col, label_col, lang, fallback_to_cpp, retry_all, 
                              example_idx, file_name, mapping_key, assigned_split)
                        
                        if limit and (count_found + skip_count) >= limit:
                            return
            else:
                # Streaming for JSONLines OR plain JSON arrays (e.g. devign.json, vulnerables.json)
                logger.info(f"Streaming file: {path}")
                with open(path, 'r', encoding='utf-8') as f:
                    # Detect format: JSON array vs newline-delimited (JSONLines)
                    # Peek at first character to decide strategy
                    first_char = ''
                    while True:
                        char = f.read(1)
                        if not char: break
                        if not char.isspace():
                            first_char = char
                            break
                    f.seek(0)
                    
                    # Label Inference: If filename suggests label and it's Reveal/JSON format
                    label_inf = None
                    if 'vulnerables.json' in path.lower() and 'non-vulnerables.json' not in path.lower():
                        label_inf = 1
                    elif 'non-vulnerables.json' in path.lower():
                        label_inf = 0

                    if first_char == '[':
                        # Full JSON array - load it into memory
                        entries_iterator = json.load(f)
                    else:
                        # JSONLines - Stream line-by-line
                        entries_iterator = (json.loads(line) for line in f if line.strip())

                    for entry_in in entries_iterator:
                        # Flatten list-of-lists just in case
                        if isinstance(entry_in, list):
                            entries_inner = entry_in
                        else:
                            entries_inner = [entry_in]
                        for entry in entries_inner:
                            # Inject label inference if found
                            if label_inf is not None:
                                entry['label_override'] = label_inf
                            # Robust ID detection: Check common keys, fallback to counter
                            example_idx = str(entry.get('example_index', 
                                              entry.get('id', 
                                              entry.get('Unnamed: 0', 
                                              str(count_found + skip_count)))))

                            # Use SHA256 of code as stable skip key (works across restarts)
                            raw_code = str(entry.get(text_col, entry.get('func', entry.get('code', ''))))
                            code_hash = _hashlib.sha256(raw_code.encode('utf-8')).hexdigest()

                            if code_hash in processed_ids:
                                skip_count += 1
                                if pbar_ref is not None:
                                    pbar_ref.update(1)
                                    if skip_count % 500 == 0:
                                        success_rate = (success_count / (success_count + err_count)) * 100 if (success_count + err_count) > 0 else 100
                                        pbar_ref.set_postfix({
                                            "Done": success_count, 
                                            "Skipped": skip_count, 
                                            "Error": err_count,
                                            "Success": f"{success_rate:.1f}%"
                                        }, refresh=False)
                                continue

                            count_found += 1
                            file_name = str(entry.get('file_name', entry.get('folder_name', example_idx)))
                            mapping_key = example_idx if split_key_col == 'example_index' else file_name
                            
                            # Split assignment
                            if split_map:
                                assigned_split = split_map.get(mapping_key, 'unmapped')
                            else:
                                hash_val = int(hashlib.md5((mapping_key + "vd_seed").encode()).hexdigest(), 16)
                                prob = (hash_val % 100) / 100.0
                                if prob < 0.8: assigned_split = 'train'
                                elif prob < 0.9: assigned_split = 'valid'
                                else: assigned_split = 'test'
                            
                            if assigned_split not in out_files: assigned_split = 'unmapped'
                            
                            yield (entry, text_col, label_col, lang, fallback_to_cpp, retry_all, 
                                  example_idx, file_name, mapping_key, assigned_split)

                            if limit and (count_found + skip_count) >= limit:
                                return

    import multiprocessing as mp
    num_cpus = max_workers if max_workers else max(1, os.cpu_count() - 1)
    logger.info(f"Starting True-Streaming Pool with {num_cpus} workers.")
    
    # Start pool immediately, count rows in background thread
    import threading

    # Use a premium format for the progress bar
    bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    pbar = tqdm(total=None, desc="ATLAS Processing", unit=" samples", bar_format=bar_fmt)

    def _count_rows(pbar_ref):
        total_samples = 0
        if not limit:
            for path in input_data_path:
                if format == 'csv':
                    try:
                        total_samples += len(pd.read_csv(path, usecols=[0]))
                    except Exception as e:
                        logger.warning(f"Could not count rows in CSV {path}: {e}")
                else:
                    try:
                        with open(path, 'rb') as f:
                            total_samples += sum(1 for _ in f)
                    except Exception as e:
                        logger.warning(f"Could not count rows in JSONLines {path}: {e}")
        else:
            total_samples = limit
        
        # Update the progress bar total DIRECTLY from the thread for instant feedback
        if pbar_ref is not None:
            actual_total = max(0, total_samples)
            pbar_ref.total = actual_total
            pbar_ref.refresh()
            logger.info(f"[Background Counter Done] Total dataset size: {actual_total} rows.")

    count_thread = threading.Thread(target=_count_rows, args=(pbar,), daemon=True)
    count_thread.start()

    # 2026 Optimization: Set maxtasksperchild to prevent memory leaks and cumulative slowdowns
    with mp.Pool(processes=num_cpus, maxtasksperchild=100) as pool:
        # Pass the pbar to the generator so it can update it while skipping
        result_iter = pool.imap_unordered(worker_process_entry, task_generator(pbar_ref=pbar), chunksize=2)
        
        for result in result_iter:
            # We no longer need to check true_remaining_holder here, the thread does it
            
            success_rate = (success_count / (success_count + err_count)) * 100 if (success_count + err_count) > 0 else 100
            status_dict = {
                "Done": success_count, 
                "Skipped": skip_count, 
                "Error": err_count,
                "Success": f"{success_rate:.1f}%"
            }

            if result["status"] == "success":
                # 2026 SOTA: Dataset Pruning (PRISM-VD Paper Section 5.3)
                # Discard samples with empty danger structures/semantics if requested
                graph_data = result['data'].get('graph_data', {})
                # For standard causal, we check semantic_paths
                has_semantics = bool(graph_data.get('semantic_paths'))
                
                if skip_empty and not has_semantics:
                    skip_count += 1
                    status_dict["Skipped"] = skip_count
                    pbar.set_postfix(status_dict)
                    continue

                success_count += 1
                status_dict["Done"] = success_count
                pbar.update(1) # MOVE THE BAR!
                out_files[result["split"]].write(json.dumps(result["data"]) + '\n')
                
                if save_individual_graphs:
                    safe_id = "".join([c for c in str(result['data']['id']) if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
                    ind_path = os.path.join(output_dir, "individual_graphs", result["split"], f"{safe_id}_graph.json")
                    with open(ind_path, "w", encoding="utf-8") as f:
                        json.dump(result["data"], f, indent=2)

                pbar.set_postfix(status_dict)
                
                # Verbose periodic log for background visibility
                if success_count % 5000 == 0:
                    elapsed = time.time() - global_start_t
                    rate = success_count / elapsed if elapsed > 0 else 0
                    # Use pbar.total for remaining estimate if available
                    if pbar.total and rate > 0:
                        remaining_est = (pbar.total - pbar.n) / rate
                    else:
                        remaining_est = '?'
                    logger.info(
                        f"[Progress] Done: {success_count} | Skipped: {skip_count} | Error: {err_count} | Success: {success_rate:.1f}%"
                        f" | Rate: {rate:.1f} samples/s"
                        f" | ETA: {remaining_est:.0f}s" if isinstance(remaining_est, float) else
                        f" | Rate: {rate:.1f} samples/s"
                    )
            else:
                err_count += 1
                status_dict["Error"] = err_count
                # Log the code hash of failed samples so they are skipped on retry
                try:
                    _failed_code = result.get('data', {}).get('code', '') if 'data' in result else ''
                    _failed_hash = _hashlib.sha256(_failed_code.encode('utf-8')).hexdigest() if _failed_code else result.get('id', '')
                    with open(failed_log_path, 'a', encoding='utf-8') as ff:
                        ff.write(_failed_hash + '\n')
                except:
                    pass
                pbar.set_postfix(status_dict)

            
    for f in out_files.values():
        f.close()
        
    global_end_t = time.time()
    total_elapsed = global_end_t - global_start_t
    logger.info(f"Successfully processed {success_count} examples. Skipped {skip_count}. Failed {err_count}. Total time taken: {total_elapsed:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industry-Level Data Preparation Pipeline for PRISM-VD++")
    parser.add_argument("--input", type=str, required=True, nargs='+', help="Raw input data files (CSV, JSON, or JSONLines)")
    parser.add_argument("--format", type=str, default="jsonlines", choices=["jsonlines", "csv", "json"])
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/val/test jsonlines")
    parser.add_argument("--split_file", type=str, help="CSV or Excel mapping items to train/valid/test")
    parser.add_argument("--split_key_col", type=str, default="example_index", help="Column name in split file acting as key (e.g. example_index or folder_name)")
    parser.add_argument("--split_val_col", type=str, default="split", help="Column name containing 'train', 'valid', or 'test'")
    parser.add_argument("--text_col", type=str, default="code", help="Column name in data containing source code")
    parser.add_argument("--label_col", type=str, default="label", help="Column name in data containing vulnerability label (1/0)")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cpp"], help="Source code language (c or cpp)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fallback_to_cpp", action="store_true")
    parser.add_argument("--retry_all", action="store_true")
    parser.add_argument("--no_wait", action="store_true", help="Skip the 300s ATLAS initialization timer")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel processes (defaults to CPU count - 1)")
    parser.add_argument("--save_individual_graphs", action="store_true", help="Save each graph as an individual JSON file")
    parser.add_argument("--skip_empty", type=str2bool, default=True, help="Discard samples with zero semantic paths (Section 5.3)")
    
    args = parser.parse_args()
    
    prepare_dataset(
        input_data_path=args.input,
        output_dir=args.output_dir,
        split_file_path=args.split_file,
        split_key_col=args.split_key_col,
        split_val_col=args.split_val_col,
        text_col=args.text_col,
        label_col=args.label_col,
        format=args.format,
        lang=args.lang,
        limit=args.limit,
        fallback_to_cpp=args.fallback_to_cpp,
        retry_all=args.retry_all,
        no_wait=args.no_wait,
        max_workers=args.max_workers,
        save_individual_graphs=args.save_individual_graphs,
        skip_empty=args.skip_empty
    )