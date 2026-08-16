with open("extract_functions.py", "r") as f:
    lines = f.read()

import re
lines = re.sub(
    r'"start_line": start_line,\n\s*"end_line": start_line \+ line_count - 1,',
    r'',
    lines
)

with open("extract_functions.py", "w") as f:
    f.write(lines)
