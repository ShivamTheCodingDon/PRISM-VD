"""
C/C++ Function Extractor for Real-World Vulnerability Testing
===============================================================
Uses tree-sitter to parse .c/.cpp files and extract function definitions.
Recursively walks a source repo directory and outputs a functions.jsonlines file.

Known CVE functions are auto-labeled as label=1; all others as label=0.

Usage:
    python extract_functions.py \
        --source_dir /media/user1/One\ Touch1/00\ Data/realdatavul/FFmpeg \
        --project ffmpeg \
        --output functions.jsonlines \
        --limit 100
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── tree-sitter setup ─────────────────────────────────────────────────────────
try:
    import tree_sitter_c as tsc
    import tree_sitter_cpp as tscpp
    from tree_sitter import Language, Parser
    C_LANGUAGE = Language(tsc.language(), "c")
    CPP_LANGUAGE = Language(tscpp.language(), "cpp")
    TREESITTER_AVAILABLE = False
except Exception:
    TREESITTER_AVAILABLE = False
    logger.warning("tree-sitter not available. Install: pip install tree-sitter tree-sitter-c tree-sitter-cpp")

# ── Import CVE database ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nday_cve_db import is_function_vulnerable, get_cves_for_project


# ── Regex fallback for function extraction ────────────────────────────────────
# Matches: <return_type> <func_name>(<params>) {
FUNC_REGEX = re.compile(
    r'^[\w\s\*]+?\s+(\w+)\s*\([^)]*\)\s*\{',
    re.MULTILINE
)


def extract_functions_treesitter(file_path: str, lang: str = 'c') -> list:
    """
    Extract function definitions from a C/C++ file using tree-sitter.
    Returns list of dicts: {func_name, code, start_line, end_line}
    """
    if not TREESITTER_AVAILABLE:
        return extract_functions_regex(file_path)

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source_code = f.read()
    except Exception as e:
        logger.debug(f"Cannot read {file_path}: {e}")
        return []

    source_bytes = source_code.encode('utf-8')

    parser = Parser()
    if lang == 'cpp':
        parser.language = CPP_LANGUAGE
    else:
        parser.language = C_LANGUAGE

    try:
        tree = parser.parse(source_bytes)
    except Exception as e:
        logger.debug(f"Parse error for {file_path}: {e}")
        return []

    functions = []
    root = tree.root_node

    def _walk(node):
        if node.type == 'function_definition':
            # Find the function declarator to get the name
            func_name = _get_func_name(node)
            if func_name:
                start = node.start_byte
                end = node.end_byte
                code_text = source_bytes[start:end].decode('utf-8', errors='replace')
                # Skip trivially small functions (< 3 lines)
                line_count = code_text.count('\n') + 1
                if line_count >= 3:
                    functions.append({
                        'func_name': func_name,
                        'code': code_text,
                        'start_line': node.start_point[0] + 1,
                        'end_line': node.end_point[0] + 1,
                    })
        for child in node.children:
            _walk(child)

    _walk(root)
    return functions


def _get_func_name(func_node):
    """Extract function name from a function_definition node."""
    # Walk the declarator subtree to find the identifier
    declarator = None
    for child in func_node.children:
        if 'declarator' in child.type:
            declarator = child
            break

    if declarator is None:
        return None

    # Recursively find the identifier (handles pointer_declarator, etc.)
    return _find_identifier(declarator)


def _find_identifier(node):
    """Recursively find the first identifier in a declarator subtree."""
    if node.type == 'identifier':
        return node.text.decode('utf-8', errors='replace')
    if node.type == 'field_identifier':
        return node.text.decode('utf-8', errors='replace')
    for child in node.children:
        result = _find_identifier(child)
        if result:
            return result
    return None


def extract_functions_regex(file_path: str) -> list:
    """
    Fallback function extractor using a robust Regex.
    Less accurate than tree-sitter, but handles standard C/C++ function definitions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source_code = f.read()
    except Exception:
        return []

    functions = []
    
    for match in FUNC_REGEX.finditer(source_code):
        func_name = match.group(1)
        start_pos = match.start()
        
        # Count braces to find the end
        brace_count = 0
        end_pos = start_pos
        in_body = False
        
        for i in range(start_pos, len(source_code)):
            if source_code[i] == '{':
                brace_count += 1
                in_body = True
            elif source_code[i] == '}':
                brace_count -= 1
                if in_body and brace_count == 0:
                    end_pos = i + 1
                    break
                    
        if end_pos > start_pos and in_body and brace_count == 0:
            code = source_code[start_pos:end_pos]
            line_count = code.count('\n') + 1
            if line_count >= 3:
                functions.append({
                    "func_name": func_name,
                    "code": code,
                    
                    "label": 0
                })
    return functions


def walk_source_tree(source_dir: str, extensions: tuple = ('.c', '.cpp', '.cc', '.cxx', '.h')) -> list:
    """Recursively find all C/C++ source files, skipping test/doc directories."""
    skip_dirs = {'test', 'tests', 'doc', 'docs', 'Documentation', 'examples',
                 'samples', '.git', '__pycache__', 'node_modules', 'build', 'scripts'}
    files = []
    for root, dirs, filenames in os.walk(source_dir):
        # Skip non-source directories
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in filenames:
            if any(fname.endswith(ext) for ext in extensions):
                files.append(os.path.join(root, fname))
    return sorted(files)


def extract_and_label(
    source_dir: str,
    project: str,
    output_path: str,
    limit: int = None,
    sample_benign: int = 0,
    lang: str = 'c',
):
    """
    Main extraction pipeline:
    1. Walk source tree for .c/.cpp files
    2. Extract functions via tree-sitter
    3. Auto-label using CVE database
    4. Write to JSONL output

    Args:
        source_dir: Root of the source code repository
        project: Project name (linux, openssl, ffmpeg, qemu, xen, libav)
        output_path: Output JSONL file path
        limit: Max number of functions to extract (None = all)
        sample_benign: If > 0, limit benign samples per CVE file to this number
        lang: Default language for tree-sitter parsing ('c' or 'cpp')
    """
    source_dir = os.path.abspath(source_dir)
    logger.info(f"Scanning source tree: {source_dir}")
    logger.info(f"Project: {project}")

    # Show known CVEs for this project
    cves = get_cves_for_project(project)
    logger.info(f"Known CVEs for {project}: {len(cves)}")
    for cve in cves:
        logger.info(f"  {cve['cve_id']}: {cve['func_name']} in {cve['file_pattern']}")

    # Find source files
    source_files = walk_source_tree(source_dir)
    logger.info(f"Found {len(source_files)} source files")

    total_functions = 0
    total_vulnerable = 0
    total_benign = 0
    func_id = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as out_f:
        from tqdm import tqdm
        for file_path in tqdm(source_files, desc="Extracting functions", unit="file"):
            # Determine language from extension
            file_lang = 'cpp' if any(file_path.endswith(e) for e in ('.cpp', '.cc', '.cxx')) else lang

            # Extract functions
            functions = extract_functions_treesitter(file_path, lang=file_lang)

            # Get relative path for CVE matching
            rel_path = os.path.relpath(file_path, source_dir)

            for func in functions:
                # Check if this function matches a known CVE
                cve_match = is_function_vulnerable(project, rel_path, func['func_name'])

                if cve_match:
                    label = 1
                    total_vulnerable += 1
                    cve_id = cve_match['cve_id']
                    cwe = cve_match['cwe']
                    logger.info(f"  ★ FOUND CVE {cve_id}: {func['func_name']} in {rel_path}")
                else:
                    label = 0
                    total_benign += 1
                    cve_id = None
                    cwe = None

                record = {
                    "id": func_id,
                    "file_name": rel_path,
                    "func_name": func['func_name'],
                    "code": func['code'],
                    "label": label,
                    "project": project,
                    "start_line": func['start_line'],
                    "end_line": func['end_line'],
                }
                if cve_id:
                    record["cve_id"] = cve_id
                    record["cwe"] = cwe

                out_f.write(json.dumps(record) + '\n')
                func_id += 1
                total_functions += 1

                if limit and total_functions >= limit:
                    break

            if limit and total_functions >= limit:
                break

    logger.info(f"\n{'='*60}")
    logger.info(f"Extraction Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Total functions : {total_functions}")
    logger.info(f"  Vulnerable (1)  : {total_vulnerable}")
    logger.info(f"  Benign (0)      : {total_benign}")
    logger.info(f"  Output          : {output_path}")
    logger.info(f"{'='*60}")

    return total_functions, total_vulnerable, total_benign


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract C/C++ functions from source repos for vulnerability testing")
    parser.add_argument("--source_dir", type=str, required=True,
                        help="Root directory of the source code repository")
    parser.add_argument("--project", type=str, required=True,
                        choices=["linux", "openssl", "ffmpeg", "qemu", "xen", "libav"],
                        help="Project name for CVE matching")
    parser.add_argument("--output", type=str, default="functions.jsonlines",
                        help="Output JSONL file path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of functions to extract")
    parser.add_argument("--lang", type=str, default="c", choices=["c", "cpp"],
                        help="Default language for parsing")

    args = parser.parse_args()

    extract_and_label(
        source_dir=args.source_dir,
        project=args.project,
        output_path=args.output,
        limit=args.limit,
        lang=args.lang,
    )
