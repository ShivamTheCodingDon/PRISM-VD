import json
import logging
import os
import argparse
import pandas as pd
from tqdm import tqdm
from atlas_adapter import parse_code_to_graph_data_uscp

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

def worker_process_entry_uscp(task_args):
    """
    Worker function to process a single USCP sample in a separate process.
    """
    (entry, text_col, label_col, lang, fallback_to_cpp, retry_all, 
     example_idx, file_name, mapping_key, assigned_split, disable_dsg_filter) = task_args
    
    import time
    from atlas_adapter import parse_code_to_graph_data_uscp
    
    code = entry.get(text_col, entry.get('func', entry.get('code', '')))
    
    # Coerce label
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

        # Generate USCP graph data with fallback retry mechanism
        start_t = time.time()
        try:
            graph_data = parse_code_to_graph_data_uscp(code, lang=row_lang, disable_dsg_filter=disable_dsg_filter)
            
            # 2026 Optimization: Aggressive Retry on Empty CFG (Section 4.1 Improvement)
            # If parse succeeded but resulted in 0 edges, it's likely a C/C++ mismatch (e.g. :: or new)
            if not graph_data.get('cfg_edges') and (fallback_to_cpp or retry_all):
                fallback_lang = 'cpp' if row_lang == 'c' else 'c'
                graph_data = parse_code_to_graph_data_uscp(code, lang=fallback_lang, disable_dsg_filter=disable_dsg_filter)
                row_lang = fallback_lang
        except Exception as atlas_err:
            if fallback_to_cpp or retry_all:
                fallback_lang = 'cpp' if row_lang == 'c' else 'c'
                graph_data = parse_code_to_graph_data_uscp(code, lang=fallback_lang, disable_dsg_filter=disable_dsg_filter)
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


def prepare_dataset_uscp(
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
    skip_empty: bool = True,
    disable_dsg_filter: bool = False
):
    """
    2026 USCP Data Preparation Pipeline.
    Generates USCP (Universal Structural Causal Paths) alongside old causal paths.
    Outputs to *_uscp.jsonlines files — never touches _causal.jsonlines.
    """
    os.makedirs(output_dir, exist_ok=True)
    import time
    global_start_t = time.time()
    
    # ATLAS Initialization Warm-up skipped (moved to wait_for_atlas.py)
    logger.info("Starting USCP data preparation.")
    
    if not isinstance(input_data_path, list):
        input_data_path = [input_data_path]
    
    # Tracking for resuming.
    # KEY FORMAT: sha256(code) + ":" + str(label)
    # This ensures the same function body with DIFFERENT labels (vul=1 vs vul=0 / benign)
    # are treated as distinct entries and BOTH get processed.
    import hashlib as _hashlib
    processed_ids = set()  # stores "<code_hash>:<label>" composite keys
    for val in ['train', 'valid', 'test', 'unmapped']:
        check_path = os.path.join(output_dir, f'{val}_uscp.jsonlines')
        if os.path.exists(check_path):
            logger.info(f"Indexing finished USCP hashes from {check_path}...")
            with open(check_path, 'r', encoding='utf-8') as cf:
                for line in cf:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            code = data.get('code', '')
                            label_val = str(data.get('label', '0'))
                            if code:
                                code_hash = _hashlib.sha256(code.encode('utf-8')).hexdigest()
                                processed_ids.add(f"{code_hash}:{label_val}")
                        except:
                            pass
                            
    failed_log_path = os.path.join(output_dir, 'failed_uscp.txt')
    # When --retry_all is set, do NOT load the failed log into processed_ids.
    # This allows previously-failed samples to be retried on the next run.
    if os.path.exists(failed_log_path) and not retry_all:
        logger.info("Loading failed_uscp.txt into skip list (use --retry_all to force-retry failed samples).")
        with open(failed_log_path, 'r', encoding='utf-8') as ff:
            for line in ff:
                if line.strip():
                    processed_ids.add(line.strip())
    elif os.path.exists(failed_log_path) and retry_all:
        logger.info("--retry_all is set: ignoring failed_uscp.txt — all previously failed samples will be retried.")
                    
    if processed_ids:
        logger.info(f"Resume Mode (USCP): Found {len(processed_ids)} already processed (code_hash:label) keys. Skipping them during file stream.")

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
        'train': open(os.path.join(output_dir, 'train_uscp.jsonlines'), 'a', encoding='utf-8'),
        'valid': open(os.path.join(output_dir, 'valid_uscp.jsonlines'), 'a', encoding='utf-8'),
        'test': open(os.path.join(output_dir, 'test_uscp.jsonlines'), 'a', encoding='utf-8'),
        'unmapped': open(os.path.join(output_dir, 'unmapped_uscp.jsonlines'), 'a', encoding='utf-8')
    }
    
    if save_individual_graphs:
        for split_name in ['train', 'valid', 'test', 'unmapped']:
            os.makedirs(os.path.join(output_dir, "individual_graphs", split_name), exist_ok=True)

    success_count = 0
    err_count = 0
    skip_count = 0

    # ── Detailed skip-reason counters ──────────────────────────────────────────
    skip_already_done  = 0   # composite_key already in processed_ids (resume)
    skip_empty_graph   = 0   # ATLAS returned 0 uscp_paths
    skip_failed_before = 0   # was in failed_uscp.txt (only when retry_all=False)

    # ── Label-distribution counters ────────────────────────────────────────────
    saved_vul    = 0         # label == 1
    saved_benign = 0         # label == 0

    # ── Per-split counters ─────────────────────────────────────────────────────
    split_counts = {'train': 0, 'valid': 0, 'test': 0, 'unmapped': 0}

    # ── Detailed skip/error log file ───────────────────────────────────────────
    detail_log_path = os.path.join(output_dir, 'uscp_detail.log')
    detail_log = open(detail_log_path, 'a', encoding='utf-8')
    import datetime as _dt
    detail_log.write(
        f"\n{'='*80}\n"
        f"Run started: {_dt.datetime.now().isoformat()}\n"
        f"Input      : {input_data_path}\n"
        f"text_col   : {text_col}  |  label_col: {label_col}\n"
        f"retry_all  : {retry_all}  |  skip_empty: {skip_empty}\n"
        f"{'='*80}\n"
    )
    detail_log.flush()

    import hashlib
    
    def task_generator_uscp(pbar_ref=None):
        nonlocal skip_count, skip_already_done
        count_found = 0
        
        for path in input_data_path:
            if format == 'csv':
                # Deterministic check for available columns
                sample_chunk = pd.read_csv(path, nrows=1)
                cols = sample_chunk.columns.tolist()
                id_col = 'example_index' if 'example_index' in cols else ('id' if 'id' in cols else None)
                
                for chunk in pd.read_csv(path, chunksize=5000):
                    for _, entry in chunk.iterrows():
                        example_idx = str(entry[id_col]) if id_col else str(count_found + skip_count)
                        if example_idx in processed_ids:
                            skip_count += 1
                            if pbar_ref is not None:
                                pbar_ref.update(1)
                                if skip_count % 500 == 0:
                                    pbar_ref.set_postfix({"skip_uscp": skip_count}, refresh=False)
                            continue
                        
                        count_found += 1
                        file_name = str(entry.get('file_name', entry.get('folder_name', example_idx)))
                        mapping_key = example_idx if split_key_col == 'example_index' else file_name
                        
                        if split_map: assigned_split = split_map.get(mapping_key, 'unmapped')
                        else:
                            hash_val = int(hashlib.md5((mapping_key + "vd_seed").encode()).hexdigest(), 16)
                            prob = (hash_val % 100) / 100.0
                            if prob < 0.8: assigned_split = 'train'
                            elif prob < 0.9: assigned_split = 'valid'
                            else: assigned_split = 'test'
                        
                        yield (entry.to_dict(), text_col, label_col, lang, fallback_to_cpp, retry_all, 
                              example_idx, file_name, mapping_key, assigned_split, disable_dsg_filter)
            else:
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
                        _entries_iterator = json.load(f)
                    else:
                        # JSONLines - Stream line-by-line
                        _entries_iterator = (json.loads(line) for line in f if line.strip())
                
                    for entry_in in _entries_iterator:
                        if isinstance(entry_in, list):
                            _entries_inner = entry_in
                        else:
                            _entries_inner = [entry_in]
                        for entry in _entries_inner:
                            # Inject label inference if found
                            if label_inf is not None:
                                entry['label_override'] = label_inf
                            # Robust ID detection: Check common keys, fallback to counter
                            example_idx = str(entry.get('example_index', 
                                              entry.get('id', 
                                              entry.get('Unnamed: 0', 
                                              str(count_found + skip_count)))))

                            # Use SHA256(code)+label as stable composite skip key.
                            # Using label in the key means vul=0 and vul=1 of the SAME
                            # function body are treated as distinct entries (both processed).
                            raw_code = str(entry.get(text_col, entry.get('func', entry.get('code', ''))))
                            code_hash = _hashlib.sha256(raw_code.encode('utf-8')).hexdigest()
                            # Resolve label early so we can build the composite key
                            _label_override = entry.get('label_override')
                            if _label_override is not None:
                                _entry_label = str(int(_label_override))
                            else:
                                _raw_label = entry.get(label_col, entry.get('target', 0))
                                try:
                                    _entry_label = str(int(_raw_label))
                                except:
                                    _entry_label = '0'
                            composite_key = f"{code_hash}:{_entry_label}"
                            
                            if composite_key in processed_ids:
                                skip_count += 1
                                skip_already_done += 1
                                if pbar_ref is not None:
                                    pbar_ref.update(1)
                                    if skip_already_done % 1000 == 0:
                                        success_rate = (success_count / (success_count + err_count)) * 100 if (success_count + err_count) > 0 else 100
                                        pbar_ref.set_postfix({
                                            "Done": success_count,
                                            "Vul": saved_vul,
                                            "Benign": saved_benign,
                                            "Skip(done)": skip_already_done,
                                            "Skip(empty)": skip_empty_graph,
                                            "Error": err_count,
                                        }, refresh=False)
                                        logger.info(
                                            f"[SKIP-RESUME] {skip_already_done} samples skipped (already processed). "
                                            f"Latest skipped id={example_idx} label={_entry_label}"
                                        )
                                continue
                            
                            count_found += 1
                            file_name = str(entry.get('file_name', entry.get('folder_name', example_idx)))
                            mapping_key = example_idx if split_key_col == 'example_index' else file_name
                            
                            if split_map: assigned_split = split_map.get(mapping_key, 'unmapped')
                            else:
                                hash_val = int(hashlib.md5((mapping_key + "vd_seed").encode()).hexdigest(), 16)
                                prob = (hash_val % 100) / 100.0
                                if prob < 0.8: assigned_split = 'train'
                                elif prob < 0.9: assigned_split = 'valid'
                                else: assigned_split = 'test'

                            yield (entry, text_col, label_col, lang, fallback_to_cpp, retry_all, 
                                  example_idx, file_name, mapping_key, assigned_split, disable_dsg_filter)
                            
                            if limit and (count_found + skip_count) >= limit:
                                return

    import multiprocessing as mp
    if max_workers:
        num_cpus = max_workers
    else:
        num_cpus = max(1, os.cpu_count() - 1)
    
    logger.info(f"Starting USCP True-Streaming Pool with {num_cpus} workers.")
    
    # Start pool immediately, count rows in background thread
    import threading
    bar_fmt = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    pbar = tqdm(total=None, desc="USCP Processing ", unit=" samples", bar_format=bar_fmt)

    def _count_rows_uscp(pbar_ref):
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
        
        if pbar_ref is not None and pbar_ref.total is None:
            actual_total = max(0, total_samples)
            pbar_ref.total = actual_total
            pbar_ref.refresh()
            logger.info(f"[USCP Background Counter Done] Total dataset size: {actual_total} rows.")

    count_thread = threading.Thread(target=_count_rows_uscp, args=(pbar,), daemon=True)
    count_thread.start()

    # 2026 Optimization: Set maxtasksperchild to prevent memory leaks and cumulative slowdowns
    with mp.Pool(processes=num_cpus, maxtasksperchild=100) as pool:
        # Pass the pbar to the generator so it can update it while skipping
        result_iter = pool.imap_unordered(worker_process_entry_uscp, task_generator_uscp(pbar_ref=pbar), chunksize=2)
        
        for result in result_iter:
            elapsed_now = time.time() - global_start_t
            total_seen  = success_count + err_count + skip_count
            success_rate = (success_count / (success_count + err_count)) * 100 if (success_count + err_count) > 0 else 100
            status_dict = {
                "✓": success_count,
                "Vul": saved_vul,
                "Ben": saved_benign,
                "SkipDone": skip_already_done,
                "SkipEmpty": skip_empty_graph,
                "Err": err_count,
            }

            if result["status"] == "success":
                graph_data = result['data'].get('graph_data', {})
                has_uscp   = bool(graph_data.get('uscp_paths'))
                cfg_edges  = len(graph_data.get('cfg_edges', []))
                dfg_edges  = len(graph_data.get('dfg_edges', []))
                uscp_paths = len(graph_data.get('uscp_paths', []))
                sample_id  = result['data'].get('id', '?')
                sample_lbl = result['data'].get('label', '?')
                sample_lang= result.get('lang', '?')

                # ── skip_empty pruning ──────────────────────────────────────
                if skip_empty and not has_uscp:
                    skip_count += 1
                    skip_empty_graph += 1
                    status_dict["SkipEmpty"] = skip_empty_graph
                    pbar.set_postfix(status_dict, refresh=False)
                    # Write to detail log every time an empty-graph is skipped
                    detail_log.write(
                        f"[SKIP-EMPTY] id={sample_id} label={sample_lbl} lang={sample_lang} "
                        f"cfg={cfg_edges} dfg={dfg_edges} uscp={uscp_paths}\n"
                    )
                    if skip_empty_graph % 500 == 0:
                        detail_log.flush()
                        logger.warning(
                            f"[SKIP-EMPTY] {skip_empty_graph} samples skipped so far because "
                            f"ATLAS returned 0 uscp_paths. "
                            f"(cfg={cfg_edges} dfg={dfg_edges} on last sample id={sample_id})"
                        )
                    continue

                # ── success path ────────────────────────────────────────────
                success_count += 1
                split_counts[result["split"]] += 1
                if int(sample_lbl) == 1:
                    saved_vul    += 1
                else:
                    saved_benign += 1

                status_dict["✓"]   = success_count
                status_dict["Vul"] = saved_vul
                status_dict["Ben"] = saved_benign
                pbar.update(1)
                out_files[result["split"]].write(json.dumps(result["data"]) + '\n')

                if save_individual_graphs:
                    safe_id = "".join([c for c in str(sample_id) if c.isalnum() or c in (' ', '.', '_', '-')]).rstrip()
                    ind_path = os.path.join(output_dir, "individual_graphs", result["split"], f"{safe_id}_graph.json")
                    with open(ind_path, "w", encoding="utf-8") as f:
                        json.dump(result["data"], f, indent=2)

                pbar.set_postfix(status_dict, refresh=False)

                # Periodic detailed logger every 1000 successes
                if success_count % 1000 == 0:
                    rate = success_count / elapsed_now if elapsed_now > 0 else 0
                    remaining_est = ((pbar.total - pbar.n) / rate) if (pbar.total and rate > 0) else None
                    eta_str = f"{remaining_est:.0f}s" if remaining_est else "?"
                    logger.info(
                        f"\n"
                        f"  ┌─────────────────── USCP Progress ───────────────────┐\n"
                        f"  │ Saved   : {success_count:>7}  (Vul={saved_vul}, Benign={saved_benign})\n"
                        f"  │ Skip    : {skip_count:>7}  (Resume={skip_already_done}, EmptyGraph={skip_empty_graph})\n"
                        f"  │ Errors  : {err_count:>7}\n"
                        f"  │ Splits  : train={split_counts['train']} valid={split_counts['valid']} "
                        f"test={split_counts['test']} unmapped={split_counts['unmapped']}\n"
                        f"  │ Rate    : {rate:.1f} samples/s  |  ETA: {eta_str}\n"
                        f"  └─────────────────────────────────────────────────────┘"
                    )
                    detail_log.flush()

            else:
                err_count += 1
                err_id    = result.get('id', '?')
                err_msg   = result.get('error', 'unknown')
                err_code  = result.get('code', '')
                err_snip  = repr(err_code[:100]) if err_code else '<no code>'
                status_dict["Err"] = err_count

                # Always write error details to the detail log
                detail_log.write(
                    f"[ERROR] id={err_id} | error={err_msg} | code_snip={err_snip}\n"
                )
                if err_count % 100 == 0:
                    detail_log.flush()

                # Log every error to the Python logger too (first 500, then every 100)
                if err_count <= 500 or err_count % 100 == 0:
                    logger.warning(
                        f"[ERROR #{err_count}] id={err_id} | {err_msg} | code: {err_snip}"
                    )

                # Blacklist only when retry_all is off
                if not retry_all:
                    try:
                        _failed_label = '0'
                        if err_code:
                            _failed_hash = _hashlib.sha256(err_code.encode('utf-8')).hexdigest()
                            _composite_fail_key = f"{_failed_hash}:{_failed_label}"
                        else:
                            _composite_fail_key = str(err_id)
                        with open(failed_log_path, 'a', encoding='utf-8') as ff:
                            ff.write(_composite_fail_key + '\n')
                    except:
                        pass

                pbar.set_postfix(status_dict, refresh=False)

            
    for f in out_files.values():
        f.close()

    global_end_t   = time.time()
    total_elapsed  = global_end_t - global_start_t
    avg_rate       = success_count / total_elapsed if total_elapsed > 0 else 0

    summary = (
        f"\n{'='*70}\n"
        f"  USCP GENERATION COMPLETE — {_dt.datetime.now().isoformat()}\n"
        f"{'='*70}\n"
        f"  Total elapsed   : {total_elapsed:.1f}s  ({total_elapsed/60:.1f} min)\n"
        f"  Avg throughput  : {avg_rate:.2f} samples/s\n"
        f"\n"
        f"  ── Saved ──────────────────────────────────────────\n"
        f"    Total saved   : {success_count}\n"
        f"    Vulnerable    : {saved_vul}  (label=1)\n"
        f"    Benign        : {saved_benign}  (label=0)\n"
        f"\n"
        f"  ── Splits ─────────────────────────────────────────\n"
        f"    train         : {split_counts['train']}\n"
        f"    valid         : {split_counts['valid']}\n"
        f"    test          : {split_counts['test']}\n"
        f"    unmapped      : {split_counts['unmapped']}\n"
        f"\n"
        f"  ── Skipped ────────────────────────────────────────\n"
        f"    Total skipped : {skip_count}\n"
        f"    Already done  : {skip_already_done}  (resume dedup)\n"
        f"    Empty graph   : {skip_empty_graph}  (0 uscp_paths from ATLAS)\n"
        f"\n"
        f"  ── Errors ─────────────────────────────────────────\n"
        f"    Total errors  : {err_count}\n"
        f"    Detail log    : {detail_log_path}\n"
        f"    Failed log    : {failed_log_path}\n"
        f"{'='*70}\n"
    )
    logger.info(summary)
    detail_log.write(summary)
    detail_log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2026 USCP Data Preparation Pipeline for PRISM-VD++")
    parser.add_argument("--input", type=str, required=True, nargs='+', help="Raw input data files (CSV, JSON, or JSONLines)")
    parser.add_argument("--format", type=str, default="jsonlines", choices=["jsonlines", "csv", "json"])
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save train/val/test _uscp.jsonlines")
    parser.add_argument("--split_file", type=str, help="CSV or Excel mapping items to train/valid/test")
    parser.add_argument("--split_key_col", type=str, default="example_index")
    parser.add_argument("--split_val_col", type=str, default="split")
    parser.add_argument("--text_col", type=str, default="code")
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cpp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--fallback_to_cpp", action="store_true")
    parser.add_argument("--retry_all", action="store_true")
    parser.add_argument("--no_wait", action="store_true", help="Skip the 300s ATLAS initialization timer")
    parser.add_argument("--max_workers", type=int, default=None, help="Max parallel processes (defaults to CPU count - 1)")
    parser.add_argument("--save_individual_graphs", action="store_true", help="Save each graph as an individual JSON file")
    parser.add_argument("--skip_empty", type=str2bool, default=True, help="Discard samples with zero USCP paths (Section 5.3)")
    parser.add_argument("--disable_dsg_filter", action="store_true", help="Disable DSG node filtering when building DFG paths to prevent empty paths")
    
    args = parser.parse_args()
    
    prepare_dataset_uscp(
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
        skip_empty=args.skip_empty,
        disable_dsg_filter=args.disable_dsg_filter
    )
