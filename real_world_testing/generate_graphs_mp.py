"""
Graph Generation Wrapper (Multiprocessing)
===========================================
Takes functions.jsonlines → produces test_uscp.jsonlines with ATLAS graph data.
Uses multiprocessing to speed up processing for large datasets.

Usage:
    python generate_graphs_mp.py \
        --input functions.jsonlines \
        --output test_uscp.jsonlines \
        --workers 25
"""

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data_processing'))
sys.path.insert(0, PROJECT_ROOT)

def process_single_function(entry_str, lang, fallback, disable_dsg_filter):
    """
    Worker function to process a single JSON line.
    Must be top-level for ProcessPoolExecutor.
    """
    # Import inside worker to ensure ATLAS tree-sitter loads per-process
    from atlas_adapter import parse_code_to_graph_data_uscp

    entry = json.loads(entry_str)
    sample_id = str(entry.get('id', ''))
    code = entry.get('code', '')
    label = entry.get('label', 0)
    file_name = entry.get('file_name', '')
    func_name = entry.get('func_name', '')

    if not code.strip():
        return {'status': 'error', 'msg': 'empty code', 'id': sample_id}

    parse_lang = 'cpp' if any(file_name.endswith(e) for e in ('.cpp', '.cc', '.cxx')) else lang

    try:
        start_t = time.time()
        try:
            graph_data = parse_code_to_graph_data_uscp(code, lang=parse_lang, disable_dsg_filter=disable_dsg_filter)
        except Exception:
            if fallback:
                alt_lang = 'cpp' if parse_lang == 'c' else 'c'
                graph_data = parse_code_to_graph_data_uscp(code, lang=alt_lang, disable_dsg_filter=disable_dsg_filter)
            else:
                raise

        elapsed = time.time() - start_t
        
        has_uscp = bool(graph_data.get('uscp_paths'))
        
        output_record = {
            "id": sample_id,
            "file_name": file_name,
            "func_name": func_name,
            "code": code,
            "label": label,
            "processing_time_sec": round(elapsed, 4),
            "graph_data": graph_data,
        }
        
        for key in ('cve_id', 'cwe', 'project'):
            if key in entry:
                output_record[key] = entry[key]

        cve_id = entry.get('cve_id', '?') if label == 1 else None

        return {
            'status': 'success',
            'record': output_record,
            'id': sample_id,
            'has_uscp': has_uscp,
            'is_cve': label == 1,
            'cve_id': cve_id,
            'func_name': func_name,
            'elapsed': elapsed,
            'cfg': len(graph_data.get('cfg_edges', [])),
            'dfg': len(graph_data.get('dfg_edges', [])),
            'uscp': len(graph_data.get('uscp_paths', []))
        }

    except Exception as e:
        return {'status': 'error', 'msg': str(e), 'id': sample_id, 'func': func_name, 'file': file_name}

def generate_uscp_graphs_mp(
    input_path: str,
    output_path: str,
    lang: str = 'c',
    fallback: bool = True,
    limit: int = None,
    skip_empty: bool = False,
    disable_dsg_filter: bool = False,
    workers: int = 10
):
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Workers: {workers}")

    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    if limit:
        total_lines = min(total_lines, limit)

    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        processed_ids.add(str(json.loads(line).get('id', '')))
                    except:
                        pass
        if processed_ids:
            logger.info(f"Resume: {len(processed_ids)} samples already processed")

    success = 0
    errors = 0
    skipped_empty = 0
    skipped_resume = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as in_f, \
         open(output_path, 'a', encoding='utf-8') as out_f, \
         ProcessPoolExecutor(max_workers=workers) as executor:

        pbar = tqdm(total=total_lines, desc="Generating Graphs", unit="func")
        
        futures = {}
        for line_num, line in enumerate(in_f):
            if not line.strip(): continue
            
            # Very fast preliminary ID check before parsing full JSON to speed up resume
            if '"id":' in line:
                pass # Parse properly inside worker to be safe, or we can parse here.
            
            entry = json.loads(line)
            sample_id = str(entry.get('id', line_num))
            
            if sample_id in processed_ids:
                skipped_resume += 1
                pbar.update(1)
                continue
                
            future = executor.submit(process_single_function, line, lang, fallback, disable_dsg_filter)
            futures[future] = sample_id
            
            if limit and len(futures) >= (limit - skipped_resume):
                break

        for future in as_completed(futures):
            res = future.result()
            
            if res['status'] == 'error':
                errors += 1
                if errors <= 20:
                    logger.warning(f"Error on {res.get('func', '?')} ({res.get('file', '?')}): {res['msg'][:100]}")
            else:
                if skip_empty and not res['has_uscp']:
                    skipped_empty += 1
                else:
                    out_f.write(json.dumps(res['record']) + '\n')
                    out_f.flush()
                    success += 1
                    
                    if res['is_cve']:
                        tqdm.write(
                            f"  ★ CVE {res['cve_id']}: {res['func_name']} | "
                            f"cfg={res['cfg']} dfg={res['dfg']} uscp={res['uscp']} | "
                            f"{res['elapsed']:.1f}s"
                        )
                        
            pbar.update(1)
            pbar.set_postfix(ok=success, err=errors, skip_e=skipped_empty)

        pbar.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"Graph Generation Complete (Multiprocessing)")
    logger.info(f"{'='*60}")
    logger.info(f"  Success      : {success}")
    logger.info(f"  Errors       : {errors}")
    logger.info(f"  Skipped empty: {skipped_empty}")
    logger.info(f"  Skipped resume: {skipped_resume}")
    logger.info(f"  Output       : {output_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate USCP graphs using multiprocessing")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="test_uscp.jsonlines")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cpp"])
    parser.add_argument("--no_fallback", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_empty", action="store_true")
    parser.add_argument("--disable_dsg_filter", action="store_true")
    parser.add_argument("--workers", type=int, default=10, help="Number of CPU processes to use")

    args = parser.parse_args()

    generate_uscp_graphs_mp(
        input_path=args.input,
        output_path=args.output,
        lang=args.lang,
        fallback=not args.no_fallback,
        limit=args.limit,
        skip_empty=args.skip_empty,
        disable_dsg_filter=args.disable_dsg_filter,
        workers=args.workers
    )
