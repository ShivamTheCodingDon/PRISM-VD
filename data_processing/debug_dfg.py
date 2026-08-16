import json
import os
import networkx as nx
from atlas_adapter import generate_atlas_graph, extract_causal_paths

def debug_dfg_emptiness(code, lang="c"):
    print(f"--- Debugging Code ({lang}) ---")
    graph = generate_atlas_graph(code, lang)
    
    # 1. Total edges in ATLAS graph
    all_edges = list(graph.edges(data=True))
    print(f"Total edges in raw graph: {len(all_edges)}")
    
    # 2. Raw DFG-like edges
    raw_df_edges = [
        (u, v, d) for u, v, d in all_edges 
        if any(keyword in str(d.get('edge_type', '')).lower() 
        for keyword in ('dfg', 'ddg', 'sdfg', 'flow', 'def', 'dependency'))
    ]
    print(f"Total DFG-like edges in raw graph: {len(raw_df_edges)}")
    
    # 3. Pruned DFG edges (using current logic)
    causal_paths, semantic_paths, path_guards = extract_causal_paths(graph)
    print(f"Extracted Causal Paths: {len(causal_paths)}")
    print(f"Extracted Semantic Paths: {len(semantic_paths)}")
    
    # Check if pruning removed everything
    if len(raw_df_edges) > 0 and len(causal_paths) == 0:
        print("WARNING: Pruning removed ALL DFG edges!")
    
    return len(raw_df_edges), len(causal_paths)

# Example code that might be "empty"
code_samples = [
    "void test(int x) { int y = x + 1; }", # Simple
    "void test() { return; }",              # No flow
    "void test(char *s) { strcpy(s, 'a'); }" # External call
]

for code in code_samples:
    debug_dfg_emptiness(code)
    print("-" * 30)
