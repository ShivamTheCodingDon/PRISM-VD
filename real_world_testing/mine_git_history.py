"""
Automated Git Miner for N-Day Vulnerability Datasets
=====================================================
Scans the git history of a repository to find security patches,
extracts the vulnerable function (before patch) and the benign function (after patch),
and outputs them to a massive JSONLines dataset.

Usage:
    python mine_git_history.py \
        --repo_path /media/user1/One\ Touch1/00\ Data/realdatavul/FFmpeg \
        --project ffmpeg \
        --output cve_db_auto.jsonlines \
        --max_commits 5000
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

# Add local path for extract_functions.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_functions import extract_functions_treesitter

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import git
except ImportError:
    logger.error("GitPython is required. Run: conda run -n vulai pip install GitPython unidiff")
    sys.exit(1)

# Keywords indicating a security fix
SECURITY_KEYWORDS = [
    "cve-", "security fix", "vulnerability", "buffer overflow", 
    "out of bounds", "out-of-bounds", "use after free", "use-after-free", 
    "memory leak", "heap overflow", "stack overflow", "integer overflow",
    "null pointer", "null dereference", "divide by zero",
    "xss", "sql injection", "dos", "denial of service"
]

def is_security_commit(message: str) -> bool:
    """Check if the commit message implies a security fix."""
    msg = message.lower()
    return any(keyword in msg for keyword in SECURITY_KEYWORDS)

def extract_cve_id(message: str) -> str:
    """Attempt to extract CVE ID from commit message."""
    match = re.search(r'(cve-\d{4}-\d{4,7})', message, re.IGNORECASE)
    return match.group(1).upper() if match else None

def get_file_content_at_commit(repo, commit, file_path):
    """Get the raw content of a file at a specific commit."""
    try:
        return commit.tree[file_path].data_stream.read().decode('utf-8', errors='replace')
    except (KeyError, Exception):
        return None

def write_temp_file(content, ext):
    """Write content to a temporary file and return the path."""
    import tempfile
    fd, path = tempfile.mkstemp(suffix=ext)
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(content)
    return path

def mine_repository(repo_path: str, project: str, output_path: str, max_commits: int = None):
    """
    Mine the repository for security commits and extract function pairs.
    """
    logger.info(f"Opening Git repository: {repo_path}")
    try:
        repo = git.Repo(repo_path)
    except git.exc.InvalidGitRepositoryError:
        logger.error(f"Not a valid git repository: {repo_path}")
        sys.exit(1)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    total_commits_scanned = 0
    security_commits_found = 0
    function_pairs_extracted = 0
    global_func_id = 0

    from tqdm import tqdm

    # Open output file in append mode so we don't lose data if it crashes
    with open(output_path, 'a', encoding='utf-8') as out_f:
        
        # Iterate over all commits in the main branch
        commits = list(repo.iter_commits('HEAD', max_count=max_commits))
        logger.info(f"Scanning {len(commits)} commits...")
        
        pbar = tqdm(commits, desc="Mining Git History")
        
        for commit in pbar:
            total_commits_scanned += 1
            
            # Check if it's a security commit
            if not is_security_commit(commit.message):
                continue
                
            security_commits_found += 1
            cve_id = extract_cve_id(commit.message)
            
            # Skip merge commits (they usually have > 1 parent)
            if len(commit.parents) != 1:
                continue
                
            parent_commit = commit.parents[0]
            
            # Get the diff between parent and this commit
            diffs = parent_commit.diff(commit, create_patch=True)
            
            for diff in diffs:
                # We only care about C/C++ files that were modified (not added/deleted)
                                    
                file_path = diff.a_path
                if not file_path or not any(file_path.endswith(ext) for ext in ('.c', '.cpp', '.cc', '.cxx', '.h', '.hpp')):
                    continue
                    
                lang = 'cpp' if file_path.endswith(('.cpp', '.cc', '.cxx', '.hpp')) else 'c'
                
                # Get file content before and after
                before_content = get_file_content_at_commit(repo, parent_commit, file_path)
                after_content = get_file_content_at_commit(repo, commit, file_path)
                
                if not before_content or not after_content:
                    continue
                    
                # Write to temp files for tree-sitter parsing
                before_tmp = write_temp_file(before_content, '.c' if lang == 'c' else '.cpp')
                after_tmp = write_temp_file(after_content, '.c' if lang == 'c' else '.cpp')
                
                try:
                    # Extract all functions
                    before_funcs = extract_functions_treesitter(before_tmp, lang=lang)
                    after_funcs = extract_functions_treesitter(after_tmp, lang=lang)
                    
                    # Convert to dictionaries for easy lookup by function name
                    before_dict = {f['func_name']: f for f in before_funcs}
                    after_dict = {f['func_name']: f for f in after_funcs}
                    
                    # Find functions that exist in both but have DIFFERENT code
                    # (These are the functions that were patched!)
                    for func_name, b_func in before_dict.items():
                        if func_name in after_dict:
                            a_func = after_dict[func_name]
                            
                            # If the code changed, we found our vulnerable/benign pair!
                            if b_func['code'] != a_func['code']:
                                
                                # Write Vulnerable (Before)
                                vul_record = {
                                    "id": f"{project}_auto_{global_func_id}_vul",
                                    "file_name": file_path,
                                    "func_name": func_name,
                                    "code": b_func['code'],
                                    "label": 1,
                                    "project": project,
                                    "commit_hash": commit.hexsha,
                                }
                                if cve_id: vul_record["cve_id"] = cve_id
                                
                                # Write Patched (After)
                                safe_record = {
                                    "id": f"{project}_auto_{global_func_id}_safe",
                                    "file_name": file_path,
                                    "func_name": func_name,
                                    "code": a_func['code'],
                                    "label": 0,
                                    "project": project,
                                    "commit_hash": commit.hexsha,
                                }
                                if cve_id: safe_record["cve_id"] = cve_id
                                
                                out_f.write(json.dumps(vul_record) + '\n')
                                out_f.write(json.dumps(safe_record) + '\n')
                                out_f.flush()
                                
                                function_pairs_extracted += 1
                                global_func_id += 1
                                
                                tqdm.write(f"  ★ Extracted pair: {func_name} in {file_path} ({cve_id or 'No CVE ID'})")
                
                finally:
                    # Clean up temp files
                    if os.path.exists(before_tmp): os.remove(before_tmp)
                    if os.path.exists(after_tmp): os.remove(after_tmp)
                    
            pbar.set_postfix(sec=security_commits_found, pairs=function_pairs_extracted)

    logger.info(f"\n{'='*60}")
    logger.info("Git Mining Complete")
    logger.info(f"{'='*60}")
    logger.info(f"  Commits scanned       : {total_commits_scanned}")
    logger.info(f"  Security commits      : {security_commits_found}")
    logger.info(f"  Function pairs found  : {function_pairs_extracted}")
    logger.info(f"  Total samples generated: {function_pairs_extracted * 2}")
    logger.info(f"  Output saved to       : {output_path}")
    logger.info(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mine Git history for N-Day Vulnerabilities")
    parser.add_argument("--repo_path", type=str, required=True, help="Path to the Git repository")
    parser.add_argument("--project", type=str, required=True, help="Project name (e.g., ffmpeg)")
    parser.add_argument("--output", type=str, default="cve_db_auto.jsonlines", help="Output JSONL file")
    parser.add_argument("--max_commits", type=int, default=None, help="Limit number of commits to scan")
    
    args = parser.parse_args()
    
    mine_repository(
        repo_path=args.repo_path,
        project=args.project,
        output_path=args.output,
        max_commits=args.max_commits
    )
