import git
import sys
sys.path.insert(0, ".")
from extract_functions import extract_functions_regex

repo = git.Repo("/media/user1/One Touch1/00 Data/realdatavul/FFmpeg")
# Get a specific known commit that modifies C code
commit_hash = "8fdafc81" # one from our earlier test
commit = repo.commit(commit_hash)
parent = commit.parents[0]

for diff in parent.diff(commit, create_patch=True):
    if diff.change_type == 'M' and diff.a_path.endswith('.c'):
        print(f"Diff file: {diff.a_path}")
        before = parent.tree[diff.a_path].data_stream.read().decode('utf-8', errors='replace')
        after = commit.tree[diff.b_path].data_stream.read().decode('utf-8', errors='replace')
        
        import tempfile, os
        fb = tempfile.mktemp(suffix='.c'); fa = tempfile.mktemp(suffix='.c')
        with open(fb, 'w') as f: f.write(before)
        with open(fa, 'w') as f: f.write(after)
        
        b_funcs = extract_functions_regex(fb)
        a_funcs = extract_functions_regex(fa)
        print(f"Before functions: {len(b_funcs)}")
        print(f"After functions: {len(a_funcs)}")
        
        before_dict = {f['func_name']: f for f in b_funcs}
        after_dict = {f['func_name']: f for f in a_funcs}
        
        for func_name, b_func in before_dict.items():
            if func_name in after_dict:
                a_func = after_dict[func_name]
                if b_func['code'] != a_func['code']:
                    print(f"FOUND MODIFIED FUNCTION: {func_name}")
        os.remove(fb); os.remove(fa)
