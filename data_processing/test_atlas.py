import sys
sys.path.append('/home/azure/PRISM-VD/PRISM-VD-Enhanced/data_processing')
from atlas_adapter import parse_code_to_graph_data_uscp
import json

# Test 1: Vulnerable code (strcpy with bounds check)
vuln_code = """
int copy_data(char *input, int len) {
    char buf[10];
    if (len < 10) {
        strcpy(buf, input);
    }
    return 0;
}
"""

print("=" * 60)
print("TEST 1: Vulnerable Code (strcpy + bounds check)")
print("=" * 60)
result = parse_code_to_graph_data_uscp(vuln_code, 'c')

print(f"\n--- OLD METHOD ---")
print(f"causal_paths ({len(result['causal_paths'])}): {result['causal_paths']}")
print(f"semantic_paths ({len(result['semantic_paths'])}): {result['semantic_paths']}")
print(f"path_guards ({len(result['path_guards'])}): {result['path_guards']}")

print(f"\n--- NEW USCP METHOD ---")
print(f"uscp_paths ({len(result['uscp_paths'])}): {result['uscp_paths']}")
print(f"uscp_guards ({len(result['uscp_guards'])}): {result['uscp_guards']}")
print(f"uscp_node_roles:")
for nid, role in result['uscp_node_roles'].items():
    if role != 'OTHER':
        node_label = result['nodes'].get(nid, {}).get('code', '?')
        print(f"  Node {nid} ({role}): {node_label}")

# Test 2: Safe code (pure math)
safe_code = """
int add(int a, int b) {
    int c = a + b;
    return c;
}
"""

print("\n" + "=" * 60)
print("TEST 2: Safe Code (pure math, no pointers/calls)")
print("=" * 60)
result2 = parse_code_to_graph_data_uscp(safe_code, 'c')
print(f"uscp_paths ({len(result2['uscp_paths'])}): {result2['uscp_paths']}")
print(f"uscp_guards ({len(result2['uscp_guards'])}): {result2['uscp_guards']}")
print(f"uscp_node_roles (non-OTHER):")
for nid, role in result2['uscp_node_roles'].items():
    if role != 'OTHER':
        node_label = result2['nodes'].get(nid, {}).get('code', '?')
        print(f"  Node {nid} ({role}): {node_label}")
