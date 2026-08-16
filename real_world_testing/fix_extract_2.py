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
            line_count = code.count('\\n') + 1
            if line_count >= 3:
                functions.append({
                    "func_name": func_name,
                    "code": code,
                    "label": 0
                })
    return functions
"""

# Match from `def extract_functions_regex` to the end of the function (which we assume is up to the return statement or similar)
# We will just replace it by simple string replacement since we just wrote it
old_func = """def extract_functions_regex(file_path: str) -> list:
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
        start_pos = match.start()"""

lines = lines.replace(old_func, new_func)

with open("extract_functions.py", "w") as f:
    f.write(lines)
