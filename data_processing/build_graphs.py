import json
import logging
import os
import argparse
from tqdm import tqdm
import traceback

from atlas_adapter import parse_code_to_graph_data_uscp as parse_code


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_dataset(input_json_path: str, output_jsonlines_path: str, fallback_to_cpp: bool = False, retry_all: bool = False):
    """
    Reads a dataset of vulnerabilities (JSON) with 'code' and 'label',
    extracts the ATLAS CFG, DFG, and Causal Paths, and writes to a JSONLines file.
    """
    if not os.path.exists(input_json_path):
        logger.error(f"Input file not found: {input_json_path}")
        return

    # ATLAS Initialization Warm-up
    logger.info("Waiting 5 minutes (300s) for ATLAS backend initialization...")
    import time
    for i in range(300, 0, -10):
        if i % 60 == 0:
            logger.info(f"ATLAS Warm-up: {i//60} min remaining...")
        time.sleep(min(i, 10))
    logger.info("ATLAS Warm-up complete. Processing graphs...")

    logger.info(f"Loading dataset from {input_json_path}...")
    if input_json_path.endswith('.jsonlines') or input_json_path.endswith('.jsonl'):
        data = []
        with open(input_json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
    logger.info(f"Loaded {len(data)} examples. Processing graphs using ATLAS...")
    
    os.makedirs(os.path.dirname(output_jsonlines_path), exist_ok=True)
    
    success_count = 0
    err_count = 0
    with open(output_jsonlines_path, 'w', encoding='utf-8') as f_out:
        pbar = tqdm(data, desc="Building Graphs")
        for idx, entry in enumerate(pbar):
            code = entry.get('code', entry.get('func_before', ''))
            label = int(entry.get('label', entry.get('vul', 0)))
            
            if "file_name" in entry:
                file_name = entry['file_name']
            else:
                file_name = entry.get('file_path', f"sample_{idx}").split('/')[-1]
                
            row_lang = 'unknown'
            try:
                # Extract lang from dataset if present, normalize to "c" or "cpp"
                row_lang = entry.get('lang', 'c')
                if isinstance(row_lang, str):
                    row_lang = row_lang.lower().strip()
                    row_lang = 'cpp' if row_lang in ['c++', 'cpp'] else 'c'
                else:
                    row_lang = 'c'

                # Use ATLAS USCP Parser (2026 SOTA) for exhaustive multi-view extraction
                try:
                    graph_data = parse_code(code, lang=row_lang)

                except Exception as atlas_err:
                    if fallback_to_cpp or retry_all:
                        fallback_lang = 'cpp' if row_lang == 'c' else 'c'
                        logger.debug(f"[{file_name}] Failed with lang={row_lang}, retrying with lang={fallback_lang}. Error: {atlas_err}")
                        graph_data = parse_code_to_graph_data(code, lang=fallback_lang)
                        row_lang = fallback_lang
                    else:
                        raise atlas_err
                
                output_obj = {
                    "id": idx,
                    "file_name": file_name,
                    "code": code,
                    "label": label,
                    "graph_data": graph_data
                }
                
                f_out.write(json.dumps(output_obj) + '\n')
                success_count += 1
            except Exception as e:
                logger.debug(f"Failed to process {file_name}: {e}")
                err_count += 1

            pbar.set_postfix({
                "success": success_count,
                "failed": err_count,
                "lang": row_lang,
                "file": file_name[:15]
            })
                
    logger.info(f"Successfully processed {success_count}/{len(data)} examples.")
    logger.info(f"Output saved to {output_jsonlines_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build ATLAS Graphs for PRISM-VD++")
    parser.add_argument("--input", type=str, required=True, help="Input JSON file with 'code' and 'label' fields")
    parser.add_argument("--output", type=str, required=True, help="Output JSONLines file")
    parser.add_argument("--fallback_to_cpp", action="store_true", help="If c fails, try cpp and vice versa")
    parser.add_argument("--retry_all", action="store_true", help="Retry with alternative language on failure")
    
    args = parser.parse_args()
    process_dataset(args.input, args.output, args.fallback_to_cpp, args.retry_all)
