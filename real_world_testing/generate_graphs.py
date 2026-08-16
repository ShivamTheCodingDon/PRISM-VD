"""
Graph Generation Wrapper for Real-World Vulnerability Testing
==============================================================
Takes functions.jsonlines → produces test_uscp.jsonlines with ATLAS graph data.
Thin wrapper around the existing atlas_adapter.parse_code_to_graph_data_uscp().

Usage:
    python generate_graphs.py \
        --input functions.jsonlines \
        --output test_uscp.jsonlines \
        --limit 50
"""

import argparse
import json
import logging
import os
import sys
import time
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Add parent paths so we can import the existing ATLAS adapter ──────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'data_processing'))
sys.path.insert(0, PROJECT_ROOT)


def generate_uscp_graphs(
    input_path: str,
    output_path: str,
    lang: str = 'c',
    fallback: bool = True,
    limit: int = None,
    skip_empty: bool = False,
    disable_dsg_filter: bool = False,
):
    """
    Process each function through ATLAS to generate USCP graph data.
    
    Args:
        input_path: Path to functions.jsonlines (from extract_functions.py)
        output_path: Path for output test_uscp.jsonlines
        lang: Default language ('c' or 'cpp')
        fallback: Try alternate language if first parse fails
        limit: Max samples to process
        skip_empty: Skip samples with 0 uscp_paths
        disable_dsg_filter: Disable DSG node filtering
    """
    from atlas_adapter import parse_code_to_graph_data_uscp

    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Count total lines for progress bar
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for line in f if line.strip())
    if limit:
        total_lines = min(total_lines, limit)

    # Resume support: load already-processed IDs
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        processed_ids.add(str(rec.get('id', '')))
                    except:
                        pass
        if processed_ids:
            logger.info(f"Resume: {len(processed_ids)} samples already processed")

    success = 0
    errors = 0
    skipped_empty = 0
    skipped_resume = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    from tqdm import tqdm

    with open(input_path, 'r', encoding='utf-8') as in_f, \
         open(output_path, 'a', encoding='utf-8') as out_f:

        pbar = tqdm(total=total_lines, desc="Generating USCP Graphs", unit="func")

        for line_num, line in enumerate(in_f):
            if not line.strip():
                continue

            entry = json.loads(line)
            sample_id = str(entry.get('id', line_num))

            # Resume check
            if sample_id in processed_ids:
                skipped_resume += 1
                pbar.update(1)
                continue

            code = entry.get('code', '')
            label = entry.get('label', 0)
            file_name = entry.get('file_name', '')
            func_name = entry.get('func_name', '')

            if not code.strip():
                errors += 1
                pbar.update(1)
                continue

            # Determine language
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

                # Check for empty USCP paths
                has_uscp = bool(graph_data.get('uscp_paths'))
                if skip_empty and not has_uscp:
                    skipped_empty += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=success, err=errors, skip_e=skipped_empty)
                    continue

                # Build output record matching the format expected by dataset_dynamic.py
                output_record = {
                    "id": sample_id,
                    "file_name": file_name,
                    "func_name": func_name,
                    "code": code,
                    "label": label,
                    "processing_time_sec": round(elapsed, 4),
                    "graph_data": graph_data,
                }

                # Copy CVE metadata if present
                for key in ('cve_id', 'cwe', 'project'):
                    if key in entry:
                        output_record[key] = entry[key]

                out_f.write(json.dumps(output_record) + '\n')
                out_f.flush()
                success += 1

                # Log CVE hits
                if label == 1:
                    cve_id = entry.get('cve_id', '?')
                    n_cfg = len(graph_data.get('cfg_edges', []))
                    n_dfg = len(graph_data.get('dfg_edges', []))
                    n_uscp = len(graph_data.get('uscp_paths', []))
                    tqdm.write(
                        f"  ★ CVE {cve_id}: {func_name} | "
                        f"cfg={n_cfg} dfg={n_dfg} uscp={n_uscp} | "
                        f"{elapsed:.1f}s"
                    )

            except Exception as e:
                errors += 1
                if errors <= 20:
                    logger.warning(f"Error on {func_name} ({file_name}): {str(e)[:100]}")

            pbar.update(1)
            pbar.set_postfix(ok=success, err=errors, skip_e=skipped_empty)

            if limit and (success + errors + skipped_empty) >= limit:
                break

        pbar.close()

    logger.info(f"\n{'='*60}")
    logger.info(f"Graph Generation Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Success      : {success}")
    logger.info(f"  Errors       : {errors}")
    logger.info(f"  Skipped empty: {skipped_empty}")
    logger.info(f"  Skipped resume: {skipped_resume}")
    logger.info(f"  Output       : {output_path}")
    logger.info(f"{'='*60}")

    return success, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate USCP graphs from extracted functions")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to functions.jsonlines from extract_functions.py")
    parser.add_argument("--output", type=str, default="test_uscp.jsonlines",
                        help="Output USCP jsonlines file")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cpp"])
    parser.add_argument("--no_fallback", action="store_true",
                        help="Don't try alternate language on parse failure")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_empty", action="store_true",
                        help="Skip samples with 0 uscp_paths")
    parser.add_argument("--disable_dsg_filter", action="store_true",
                        help="Disable DSG node filtering")

    args = parser.parse_args()

    generate_uscp_graphs(
        input_path=args.input,
        output_path=args.output,
        lang=args.lang,
        fallback=not args.no_fallback,
        limit=args.limit,
        skip_empty=args.skip_empty,
        disable_dsg_filter=args.disable_dsg_filter,
    )
