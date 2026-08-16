with open("mine_git_history.py", "r") as f:
    lines = f.read()

import re
lines = re.sub(r'if diff\.change_type != \'M\':\n\s*continue\n', '', lines)

with open("mine_git_history.py", "w") as f:
    f.write(lines)
