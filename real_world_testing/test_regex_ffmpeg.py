import re
import sys

def extract_functions_regex(source_code: str) -> list:
    # A more robust regex for C function definitions
    # Matches:
    # 1. Optional static/inline/macros at the start of line
    # 2. Return type
    # 3. Function name
    # 4. Parameters (handles newlines, but ignores complex function pointers for simplicity)
    # 5. Optional macros after parameters
    # 6. Opening brace
    
    # We will use a simpler approach: find all opening braces at the start of a line,
    # or preceded by a closing parenthesis and optional whitespace/macros.
    
    functions = []
    
    # Regex to match the signature part before the opening brace
    # It looks for an identifier, followed by parentheses containing anything (balanced ideally, but we use a non-greedy dotall),
    # followed by an opening brace.
    # To prevent matching if/while/for, we exclude them.
    
    pattern = re.compile(
        r'^[ \t]*(?!if|while|for|switch|return|sizeof|catch)(?:[\w\*]+\s+)+(\w+)\s*\([^;{}]*\)\s*\{',
        re.MULTILINE
    )
    
    for match in pattern.finditer(source_code):
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
                functions.append(func_name)
                
    return functions

with open("/media/user1/One Touch1/00 Data/realdatavul/FFmpeg/libavformat/http.c", "r") as f:
    code = f.read()

funcs = extract_functions_regex(code)
print(f"Found {len(funcs)} functions.")
print(funcs[:20])
