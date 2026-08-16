with open("extract_functions.py", "r") as f:
    lines = f.read()

import re

new_func = """def extract_functions_regex(file_path: str) -> list:
    \"\"\"
    Fallback function extractor using a robust Regex.
    Less accurate than tree-sitter, but handles standard C/C++ function definitions.
    \"\"\"
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            source_code = f.read()
    except Exception:
        return []

    functions = []
    
    for match in FUNC_REGEX.finditer(source_code):
        func_name = match.group(1)
        start_pos = match.start()
"""

lines = re.sub(r'def extract_functions_regex\(source_code: str\) -> list:.*?start_pos = match\.start\(\)', new_func, lines, flags=re.DOTALL)

with open("extract_functions.py", "w") as f:
    f.write(lines)
