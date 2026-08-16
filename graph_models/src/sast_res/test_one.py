import json
import subprocess

def main():
    jsonlines_path = "test_uscp.jsonlines"
    
    with open(jsonlines_path, "r") as f:
        first_line = f.readline()
        
    data = json.loads(first_line.strip())
    code = data["code"]
    id_str = data["id"]
    
    file_path = f"test_{id_str}.c"
    with open(file_path, "w") as f:
        f.write(code)
        
    print(f"Created {file_path}")
    print("=" * 40)
    
    print("Running flawfinder (without --csv)...")
    res1 = subprocess.run(["flawfinder", file_path], capture_output=True, text=True)
    print("STDOUT:")
    print(res1.stdout)
    print("STDERR:")
    print(res1.stderr)
    print("=" * 40)
    
    print("Running flawfinder (with --csv --quiet)...")
    res2 = subprocess.run(["flawfinder", "--csv", "--quiet", file_path], capture_output=True, text=True)
    print("STDOUT:")
    print(res2.stdout)
    print("STDERR:")
    print(res2.stderr)
    print("=" * 40)
    
    print("Running cppcheck (default)...")
    res3 = subprocess.run(["cppcheck", file_path], capture_output=True, text=True)
    print("STDOUT:")
    print(res3.stdout)
    print("STDERR:")
    print(res3.stderr)
    print("=" * 40)
    
    print("Running cppcheck (with severity template)...")
    res4 = subprocess.run(["cppcheck", "--quiet", "--template={severity}", file_path], capture_output=True, text=True)
    print("STDOUT:")
    print(res4.stdout)
    print("STDERR:")
    print(res4.stderr)
    print("=" * 40)

if __name__ == "__main__":
    main()
