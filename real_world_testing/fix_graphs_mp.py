import re

with open("generate_graphs.py", "r") as f:
    content = f.read()

# Add multiprocessing imports
if "concurrent.futures" not in content:
    content = content.replace("import time\n", "import time\nimport concurrent.futures\n")

# Modify generate_uscp_graphs signature to accept workers
content = re.sub(
    r'def generate_uscp_graphs\(.*?\):',
    'def generate_uscp_graphs(\n    input_path: str,\n    output_path: str,\n    lang: str = "c",\n    fallback: bool = True,\n    limit: int = None,\n    skip_empty: bool = False,\n    disable_dsg_filter: bool = False,\n    workers: int = 1,\n):',
    content,
    flags=re.DOTALL
)

# Replace the sequential loop with a multiprocessing loop
old_loop = """            try:
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

                out_f.write(json.dumps(output_record) + '\\n')
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

        pbar.close()"""

# But wait, replacing that logic is hard. 
# It's better to just write a new file completely so we don't mess up.
