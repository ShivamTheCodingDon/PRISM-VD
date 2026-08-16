import git
repo = git.Repo("/media/user1/One Touch1/00 Data/realdatavul/FFmpeg")
commit = repo.commit("8fdafc81")
parent = commit.parents[0]

for diff in parent.diff(commit, create_patch=True):
    print(f"Diff: type={diff.change_type}, a_path={diff.a_path}, b_path={diff.b_path}")
