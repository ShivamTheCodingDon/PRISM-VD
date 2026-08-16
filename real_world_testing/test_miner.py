import git
import sys
sys.path.insert(0, ".")
from extract_functions import extract_functions_regex
repo = git.Repo("/media/user1/One Touch1/00 Data/realdatavul/FFmpeg")
commits = list(repo.iter_commits('HEAD', max_count=50))
for commit in commits:
    if not commit.parents: continue
    parent = commit.parents[0]
    for diff in parent.diff(commit):
        if diff.change_type == 'M' and diff.a_path and diff.a_path.endswith('.c'):
            try:
                before = parent.tree[diff.a_path].data_stream.read().decode('utf-8')
                after = commit.tree[diff.b_path].data_stream.read().decode('utf-8')
                import tempfile, os
                fb = tempfile.mktemp(suffix='.c'); fa = tempfile.mktemp(suffix='.c')
                with open(fb, 'w') as f: f.write(before)
                with open(fa, 'w') as f: f.write(after)
                b_funcs = extract_functions_regex(fb)
                a_funcs = extract_functions_regex(fa)
                print(f"Commit {commit.hexsha[:8]} {diff.a_path}: {len(b_funcs)} before, {len(a_funcs)} after")
                os.remove(fb); os.remove(fa)
            except Exception as e:
                print(e)
