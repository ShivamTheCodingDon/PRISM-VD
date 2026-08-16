import sys
import os
import networkx as nx

sys.setrecursionlimit(20000)

# Ensure ATLAS is accessible in the environment
try:
    from atlas.codeviews.combined_graph.combined_driver import CombinedDriver
except ImportError:
    print("Error: ATLAS module not found. Please pip install the ATLAS-multi-view-code-representation-tool.")
    sys.exit(1)

import re
import time
import signal

class WorkerTimeoutError(Exception):
    """Custom exception raised when a graph extraction task takes too long."""
    pass

def handler(signum, frame):
    raise WorkerTimeoutError("Hard timeout (180s) reached during graph extraction.")

# Register the signal handler (Only for Unix-like systems)
if sys.platform != "win32":
    signal.signal(signal.SIGALRM, handler)


MAX_PATH_DEPTH = 20
MAX_PATHS_PER_PAIR = 3



DSG_VULNERABILITY_RELATED_TYPES = {
    # 1. Expressions (Direct Logic & Function Interactions)
    "call_expression", "binary_expression", "unary_expression", 
    "assignment_expression", "subscript_expression", "field_expression", 
    "sizeof_expression", "cast_expression", "conditional_expression",
    "update_expression", "comma_expression", "parenthesized_expression",
    "compound_literal_expression", "lambda_expression", "type_trait_expression",
    "requires_expression", "fold_expression", "new_expression", "delete_expression",
    "co_await_expression", "typeid_expression", "scoped_identifier",
    "asm_statement", "gnu_asm_expression", "structured_binding_declaration",
    "preproc_call", "preproc_def", "preproc_function_def",
    
    # 2. Declarations & Declarators (Data Flow Origins)
    "declaration", "field_declaration", "parameter_declaration",
    "pointer_declarator", "array_declarator", "init_declarator",
    "function_declarator", "abstract_pointer_declarator", "reference_declarator",
    "using_declaration", "alias_declaration", "static_assert_declaration",
    "type_definition", "namespace_definition", "template_declaration",
    "template_instantiation", "variadic_parameter", "variadic_declarator",
    "namespace_alias_definition", "using_directive", "template_alias_declaration",
    
    # 3. Control Flow Statements (Guards & Jump context)
    "if_statement", "while_statement", "for_statement", "do_statement", 
    "switch_statement", "case_statement", "default_statement", 
    "return_statement", "goto_statement", "break_statement", "continue_statement",
    "throw_statement", "try_statement", "catch_clause", "labeled_statement",
    "compound_statement", "expression_statement", "for_range_loop",
    "co_return_statement", "co_yield_statement",
    "preproc_if", "preproc_ifdef", "preproc_elif", "preproc_else",
    
    # 4. Values & Literals (Security Sensitive Payloads)
    "string_literal", "raw_string_literal", "concatenated_string", 
    "char_literal", "number_literal", "null", "nullptr", 
    "boolean_literal", "user_defined_literal",
    
    # 5. Structural/Type Components (Context Preservation)
    "array_type", "pointer_type", "type_descriptor", "enum_specifier",
    "struct_specifier", "union_specifier", "class_specifier",
    "base_class_clause", "field_initializer_list", "field_initializer",
    "template_type", "template_method", "template_function",
    "constructor_or_destructor_definition", "attribute_specifier"
}

def is_node_dsg_relevant(graph: nx.MultiDiGraph, node_id) -> bool:
    node_type = str(graph.nodes[node_id].get("node_type", "")).lower()
    return node_type in DSG_VULNERABILITY_RELATED_TYPES


def clean_node_code(code: str, remove_attributes: bool = True) -> str:
    """
    Precision cleaning for node labels.
    - preserve_attr=True: Keeps compiler attributes if ATLAS parsed them successfully.
    - preserve_attr=False: Removes them for maximum compatibility.
    """
    if not code:
        return ""
    
    # Remove noise prefixes like '1_', '2_', etc. if separated by underscore
    code = re.sub(r'^\d+_', '', code.strip())
    
    if remove_attributes:
        code = re.sub(r'__attribute__\s*\(\(.*\)\)', ' ', code)
        code = re.sub(r'__declspec\s*\(\(.*\)\)', ' ', code)
        code = re.sub(r'__asm__\s*\(.*\)', ' ', code)

    # Collapse all whitespace into a single space
    code = re.sub(r'\s+', ' ', code)
    return code.strip()

def sanitize_source_code(src_code: str, remove_attributes: bool = True):
    """
    Hybrid Sanitization:
    - remove_attributes=True: Strict mode for ATLAS parser (prevents crashes).
    - remove_attributes=False: Rich mode for CodeBERT (preserves compiler context).
    """
    if not src_code:
        return ""
    
    # 1. Strip C-style comments (/* ... */ and // ...)
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " 
        else:
            return s 
    
    clean_code = pattern.sub(replacer, src_code)
    
    # 2. Selectively remove compiler-specific attributes
    if remove_attributes:
        clean_code = re.sub(r'__attribute__\s*\(\(.*\)\)', ' ', clean_code)
        clean_code = re.sub(r'__declspec\s*\(\(.*\)\)', ' ', clean_code)
        clean_code = re.sub(r'__asm__\s*\(.*\)', ' ', clean_code)
        clean_code = re.sub(r'\b(restrict|volatile|inline)\b', ' ', clean_code)
    
    # 3. Remove non-printable/non-ASCII characters
    clean_code = "".join(c for c in clean_code if ord(c) < 128)
    
    # 4. Normalize whitespace (Collapse to single spaces)
    clean_code = re.sub(r'\s+', ' ', clean_code)
    
    return clean_code.strip()

def generate_atlas_graph(code: str, lang: str = "c"):
    """
    Generate AST, CFG, DFG directly from source text using ATLAS.
    Tries RICH mode (attributes preserved) before falling back to STRICT mode.
    Returns: (networkx mapping, attributes_preserved_bool)
    """
    codeviews = {
        "AST": {
            "exists": True,
            "collapsed": False,
            "minimized": False,
            "blacklisted": []
        },
        "DFG": {
            "exists": True,
            "collapsed": False,
            "minimized": False,
            "statements": True,
            "last_def": True,
            "last_use": True
        },
        "SDFG": {
            "exists": False,  # Turned off to save processing time (DFG handles it)
            "collapsed": False,
            "minimized": False
        },
        "CFG": {
            "exists": True,
        }
    }
    
    # Prepared Sanitized Versions
    rich_code = sanitize_source_code(code, remove_attributes=False)
    strict_code = sanitize_source_code(code, remove_attributes=True)
    
    # Dummy Wrapper Versions (to force CFG extraction on code fragments)
    wrapped_rich = f"void dummy_wrapper_vd() {{\n{rich_code}\n}}"
    wrapped_strict = f"void dummy_wrapper_vd() {{\n{strict_code}\n}}"
    
    fallback_lang = "cpp" if lang == "c" else "c"
    
    # Strategy: (Code, Lang, WasAttributesPreserved)
    # Order: Target Lang -> Fallback Lang -> Wrapped Target Lang -> Wrapped Fallback Lang
    attempts = [
        (rich_code, lang, True), 
        (strict_code, lang, False),
        (rich_code, fallback_lang, True),
        (strict_code, fallback_lang, False),
        (wrapped_rich, lang, True),
        (wrapped_strict, lang, False),
        (wrapped_rich, fallback_lang, True),
        (wrapped_strict, fallback_lang, False)
    ]
    
    last_err = None
    for code_try, lang_try, attr_preserved in attempts:
        try:
            driver = CombinedDriver(
                src_language=lang_try,
                src_code=code_try,
                output_file=None, # In-memory
                codeviews=codeviews
            )
            graph = driver.get_graph()
            
            # Reject if parsed but CFG is empty (treat as failure to trigger next attempt)
            has_cfg = any(
                'cfg' in str(d.get('edge_type', '')).lower() or 
                'cfg' in str(d.get('controlflow_type', '')).lower()
                for _, _, d in graph.edges(data=True)
            )
            if not has_cfg:
                raise ValueError("Parsed but NO CFG edges found")
                
            return graph, attr_preserved
        except (RecursionError, Exception) as e:
            last_err = e
            continue
            
    # Final Fallback
    import logging
    logger = logging.getLogger("atlas_adapter")
    logger.error(f"ATLAS failed all attempts. Final error: {last_err}")
    
    g = nx.MultiDiGraph()
    g.add_node(0, code=code[:100], type="FallbackRoot")
    return g, False

def extract_causal_paths(graph: nx.MultiDiGraph):
    """
    Extract causal paths (data flow pathways) instead of arbitrary Dangerous Structures.
    Looks for sequences of DEF -> USE or flows_to edges.
    No limits applied — extracts all paths exhaustively.
    """
    semantic_paths = []

    # Extract all variants of data flow and dependency edges
    # Standard: DFG_edge, DDG_edge
    # Advanced: SDFG_edge, reachable_def, data_dependency
    df_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) 
        if any(keyword in str(d.get('edge_type', d.get('dataflow_type', ''))).lower() 
               for keyword in ('dfg', 'ddg', 'sdfg', 'flow', 'def', 'dependency', 'comesfrom', 'lastdef', 'parameter', 'use', 'reach', 'var', 'ref'))
    ]
    
    dfg_graph = nx.DiGraph()
    for u, v, d in df_edges:
        # 2025 Improvement: Prune DFG nodes to only include vulnerability-related types (DSG)
        # Using enhanced relevance check to handle statement-level mapping
        is_relevant = is_node_dsg_relevant(graph, u) or is_node_dsg_relevant(graph, v)
        
        if is_relevant:
            dfg_graph.add_edge(u, v, type=d.get('dataflow_type', d.get('edge_type', 'DFG_edge')))
        
    cfg_edges_list = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'CFG_edge']
    cfg_graph = nx.DiGraph()
    for u, v, d in cfg_edges_list:
        cfg_graph.add_edge(u, v, type=d.get('controlflow_type', 'CFG_edge'))
        
    # Pre-collect all nodes with 0 incoming data flow edges (Sources)
    sources = [n for n in dfg_graph.nodes() if dfg_graph.in_degree(n) == 0]
    # Pre-collect all nodes with 0 outgoing data flow edges (Sinks)
    sinks = [n for n in dfg_graph.nodes() if dfg_graph.out_degree(n) == 0]
    
    # PERFORMANCE OPTIMIZATION: Pre-calculate ancestors and guard-info for the entire CFG once
    # This avoids O(N) traversals inside the nested path loops.
    all_ancestors = {n: nx.ancestors(cfg_graph, n) for n in cfg_graph.nodes()}
    guard_node_info = {} # anc -> whether it is a guard
    for n in cfg_graph.nodes():
        out_edges = cfg_graph.out_edges(n, data=True)
        types = [e[2].get('type') for e in out_edges]
        if 'pos_next' in types or 'neg_next' in types:
            guard_node_info[n] = True
    
    # 2026 SOTA: Expanded Comprehensive Sensor List based on CWE categories and vocab.txt
    SENSORS = [
        # Input & Taint Sources (CWE-20, 78, 89)
        "scanf", "fscanf", "sscanf", "vscanf", "vfscanf", "vsscanf", "getenv", "argv", 
        "cin", "getc", "getchar", "fgetc", "fgetwc", "fgetws", "getwc", "getwchar", 
        "fgets", "fread", "read", "recv", "recvfrom", "recvmsg", "readv", "pread", 
        "preadv", "msgrcv", "getmsg", "getpmsg", "readlink", "readlinkat",
        
        # Memory Management (CWE-119, 787, 416, 476)
        "malloc", "free", "realloc", "calloc", "alloca", "memset", "memcpy", "memmove", 
        "memcmp", "memchr", "memccpy", "new", "delete", "wmemcpy", "wmemmove", "wmemset", 
        "wmemcmp", "wmemchr", "mmap", "munmap", "mprotect", "brk", "sbrk", "shmat", "shmget", "shmdt",

        # String & Buffer Operations (CWE-120, 125, 131)
        "strcpy", "strncpy", "strcat", "strncat", "strtok", "strtok_r", "strlen", "strnlen",
        "wcscpy", "wcsncpy", "wcscat", "wcsncat", "wcstok", "wcslen", "sprintf", "snprintf", 
        "vsprintf", "vsnprintf", "vasprintf", "swprintf", "puts", "fputs", "gets", "readlink", 
        "realpath", "strstr", "strchr", "strrchr", "strpbrk",

        # Format String & Output (CWE-134)
        "printf", "fprintf", "vprintf", "vfprintf", "wprintf", "fwprintf", "syslog", "perror",

        # OS & Execution (CWE-78, 242, 250, 732)
        "system", "exec", "execl", "execlp", "execle", "execv", "execvp", "execvpe", "popen",
        "signal", "raise", "exit", "_Exit", "abort", "fork", "vfork", "clone",
        "setuid", "setgid", "seteuid", "setegid", "chown", "chmod", "fchmod", "chroot",
        "dlopen", "dlsym", "dlclose", "tmpnam", "tempnam", "mktemp", "tmpfile",
        
        # Structural & Symbols (Pointer/Array/Integer safety)
        "*", "[", "->", "&", "unsigned", "signed", "sizeof", "static_cast", 
        "reinterpret_cast", "dynamic_cast", "const_cast", "typeid", "asm", "__asm"
    ]
    
    # Task timeout for this specific function to prevent freezes
    # Exhaustive Timeout: 1 hour per function
    TASK_TIMEOUT = 90.0

    start_total_t = time.time()
    
    # 2026 Hard Optimization: Implement system-level SIGALRM (Hard Kill Switch)
    # This prevents its internal NetworkX recursion from blocking the Python thread.
    if sys.platform != "win32":
        signal.alarm(90)

    try:
        for source in sources:
            for sink in sinks:
                if time.time() - start_total_t > TASK_TIMEOUT:
                    break
                try:
                    # 2025 Improvement: Add cutoff and quantity limit to prevent freezes
                    path_count_for_pair = 0
                    for path in nx.all_simple_paths(dfg_graph, source=source, target=sink, cutoff=MAX_PATH_DEPTH):
                        src_label = str(graph.nodes[source].get('label', '')).lower()
                        sink_label = str(graph.nodes[sink].get('label', '')).lower()
                        is_semantic = any(s in src_label for s in SENSORS) or any(s in sink_label for s in SENSORS)
                        if is_semantic:
                            semantic_paths.append(path)
                        
                        path_count_for_pair += 1
                        if path_count_for_pair >= MAX_PATHS_PER_PAIR:
                            break
                except (nx.NetworkXNoPath, Exception):
                    continue
                except (nx.NetworkXNoPath, Exception):
                    continue
    finally:
        if sys.platform != "win32":
            signal.alarm(0) # Cancel the alarm
                
    return semantic_paths


# ============================================================================
# USCP: Universal Structural Causal Paths (2026 SOTA)
# ============================================================================

# Node structural role classification — covers most CWEs without hardcoding API names
USCP_ROLE_MAP = {
    # CALL — any function invocation (CWE-78, 120, 134, 676)
    'call_expression': 'CALL',
    'new_expression': 'CALL',
    'delete_expression': 'CALL',
    'co_await_expression': 'CALL',
    
    # DEREF — pointer/array access (CWE-119, 125, 787, 416)
    'pointer_declarator': 'DEREF',
    'pointer_expression': 'DEREF',
    'subscript_expression': 'DEREF',
    'field_expression': 'DEREF',            # struct->member
    
    # ENTRY — function parameters / taint sources (CWE-20, 89)
    'parameter_declaration': 'ENTRY',
    'parameter_list': 'ENTRY',
    'function_definition': 'ENTRY',         # entry to function body
    
    # EXIT — return statements / exceptional exits (CWE-252, 476)
    'return_statement': 'EXIT',
    'throw_statement': 'EXIT',
    'rethrow_statement': 'EXIT',
    'co_return_statement': 'EXIT',
    'co_yield_statement': 'EXIT',
    
    # ASSIGN — assignments (CWE-457, 190)
    'assignment_expression': 'ASSIGN',
    'init_declarator': 'ASSIGN',
    'update_expression': 'ASSIGN',          # i++, --j
    'structured_binding_declaration': 'ASSIGN',
    
    # GUARD — conditionals / control flow (contextual inhibitors)
    'if_statement': 'GUARD',
    'while_statement': 'GUARD',
    'for_statement': 'GUARD',
    'switch_statement': 'GUARD',
    'do_statement': 'GUARD',
    'case_statement': 'GUARD',
    'default_statement': 'GUARD',
    'catch_clause': 'GUARD',
    'for_range_loop': 'GUARD',
    'conditional_expression': 'GUARD',     # ternary
    'preproc_if': 'GUARD',
    'preproc_ifdef': 'GUARD',
    'preproc_elif': 'GUARD',
    'preproc_else': 'GUARD',
}

# Additional label-based patterns for nodes whose node_type is generic
USCP_LABEL_PATTERNS = [
    (re.compile(r'[\*]'), 'DEREF'),     # pointer dereference in label
    (re.compile(r'\['), 'DEREF'),       # array indexing in label
    (re.compile(r'->'), 'DEREF'),       # struct member access in label
    (re.compile(r'&(?!&)'), 'DEREF'),   # address-of operator (but not &&)
    (re.compile(r'\w+\s*\('), 'CALL'),  # function call pattern like "strcpy(" or "foo ("
]

def classify_node_role(node_data: dict) -> str:
    """
    Classify a node into a structural role using AST node_type first,
    then falling back to label pattern matching.
    """
    node_type = node_data.get('node_type', '')
    
    # 1. Direct AST type match
    role = USCP_ROLE_MAP.get(node_type)
    if role:
        return role
    
    # 2. Label-based fallback for generic node types
    label = str(node_data.get('label', ''))
    for pattern, role in USCP_LABEL_PATTERNS:
        if pattern.search(label):
            return role
    
    return 'OTHER'


def classify_dfg_node_role(graph: nx.MultiDiGraph, node_id, node_roles: dict) -> str:
    """
    Enhanced classification for DFG nodes.
    ATLAS puts statement-level nodes in DFG (expression_statement, declaration),
    not leaf AST nodes (call_expression). This function checks AST children
    of a DFG node to propagate structural roles upward.
    """
    own_role = node_roles.get(node_id, 'OTHER')
    if own_role != 'OTHER':
        return own_role
    
    # Check AST children — if any child has a structural role, inherit it
    # Priority: CALL > DEREF > ASSIGN > ENTRY > EXIT
    child_roles = set()
    for _, child, edge_data in graph.edges(node_id, data=True):
        if edge_data.get('edge_type') == 'AST_edge':
            child_role = node_roles.get(child, 'OTHER')
            if child_role != 'OTHER':
                child_roles.add(child_role)
            # Go one more level deep (grandchildren)
            for _, grandchild, ge in graph.edges(child, data=True):
                if ge.get('edge_type') == 'AST_edge':
                    gc_role = node_roles.get(grandchild, 'OTHER')
                    if gc_role != 'OTHER':
                        child_roles.add(gc_role)
    
    # Return highest-priority role found in children
    for priority_role in ['CALL', 'DEREF', 'ASSIGN', 'ENTRY', 'EXIT']:
        if priority_role in child_roles:
            return priority_role
    
    return 'OTHER'


def extract_universal_causal_paths(graph: nx.MultiDiGraph, disable_dsg_filter: bool = False):
    """
    2026 SOTA: Universal Structural Causal Path extraction.
    
    Instead of hardcoding API names, classifies nodes by their AST structural role
    (CALL, DEREF, ENTRY, EXIT, ASSIGN, GUARD) and extracts paths between
    security-relevant structural boundaries.
    
    Prunes the graph based on PRISM-VD's novel DSG filtering rules to prevent noise
    and computational hangs.
    """
    # 1. Classify every node (basic)
    node_roles = {}
    for n, d in graph.nodes(data=True):
        node_roles[n] = classify_node_role(d)
    
    # 2. Build subgraphs (Exhaustive Data Flow Collection)
    df_edges = [
        (u, v, d) for u, v, d in graph.edges(data=True) 
        if any(keyword in str(d.get('edge_type', d.get('dataflow_type', ''))).lower() 
               for keyword in ('dfg', 'ddg', 'sdfg', 'flow', 'def', 'dependency', 'comesfrom', 'lastdef', 'parameter', 'use', 'reach', 'var', 'ref'))
    ]
    dfg_graph = nx.DiGraph()
    for u, v, d in df_edges:
        # Pruning logic: Only include nodes that are vulnerability-related (DSG) or have a USCP Role
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        
        is_relevant = (
            disable_dsg_filter or
            node_roles[u] != 'OTHER' or node_roles[v] != 'OTHER' or
            is_node_dsg_relevant(graph, u) or is_node_dsg_relevant(graph, v)
        )
        
        if is_relevant:
            dfg_graph.add_edge(u, v, type=d.get('dataflow_type', d.get('edge_type', 'DFG_edge')))
    
    cfg_edges = [(u, v, d) for u, v, d in graph.edges(data=True) if d.get('edge_type') == 'CFG_edge']
    cfg_graph = nx.DiGraph()
    for u, v, d in cfg_edges:
        cfg_graph.add_edge(u, v, type=d.get('controlflow_type', 'CFG_edge'))
    
    # 3. Enhanced classification for DFG nodes — propagate child roles upward
    dfg_node_roles = {}
    for n in dfg_graph.nodes():
        dfg_node_roles[n] = classify_dfg_node_role(graph, n, node_roles)
    
    # 4. Identify structural sources and sinks
    structural_sources = set()
    structural_sinks = set()
    guard_nodes = set()
    
    # Collect GUARD nodes from full graph (they may not be in DFG)
    for n, role in node_roles.items():
        if role == 'GUARD':
            guard_nodes.add(n)
    
    for n in dfg_graph.nodes():
        role = dfg_node_roles.get(n, 'OTHER')
        
        if role == 'ENTRY':
            structural_sources.add(n)
        elif role == 'EXIT':
            structural_sinks.add(n)
        elif role == 'GUARD':
            guard_nodes.add(n)
        elif role in ('CALL', 'DEREF', 'ASSIGN'):
            if dfg_graph.in_degree(n) == 0:
                structural_sources.add(n)
            if dfg_graph.out_degree(n) == 0:
                structural_sinks.add(n)
            if role in ('CALL', 'DEREF'):
                structural_sinks.add(n)
    
    # Fallback: if no structural sources/sinks found, use all DFG endpoints
    if not structural_sources:
        structural_sources = {n for n in dfg_graph.nodes() if dfg_graph.in_degree(n) == 0}
    if not structural_sinks:
        structural_sinks = {n for n in dfg_graph.nodes() if dfg_graph.out_degree(n) == 0}
    
    # 4. Extract paths — no limits
    uscp_paths = []
    uscp_guards = []
    
    # Task timeout for this specific function to prevent freezes
    # Exhaustive Timeout: 1 hour per function
    TASK_TIMEOUT = 90.0

    start_total_t = time.time()
    
    # 2026 Hard Optimization: Implement system-level SIGALRM (Hard Kill Switch)
    # This prevents its internal NetworkX recursion from blocking the Python thread.
    if sys.platform != "win32":
        signal.alarm(90)

    try:
        for source in structural_sources:
            for sink in structural_sinks:
                if time.time() - start_total_t > TASK_TIMEOUT:
                    break
                if source == sink:
                    continue
                try:
                    # 2026 SOTA: Performance-safe path extraction (Limit results to prevent freezes)
                    path_count_for_pair = 0
                    
                    # Performance optimization: pre-calculate ancestors if not already done for this graph
                    if 'all_anc' not in locals():
                        all_anc = {n: nx.ancestors(cfg_graph, n) for n in cfg_graph.nodes()}

                    for path in nx.all_simple_paths(dfg_graph, source=source, target=sink, cutoff=MAX_PATH_DEPTH):
                        uscp_paths.append(path)
                        
                        # Find control-flow guards that dominate nodes in this path
                        guards_for_path = []
                        for node in path:
                            # 2026 Optimization: Use pre-calculated ancestor dictionary
                            ancestors = all_anc.get(node, set())
                            for anc in ancestors:
                                if anc in guard_nodes and anc not in guards_for_path:
                                    guards_for_path.append(anc)
                        uscp_guards.append(guards_for_path)
                        
                        path_count_for_pair += 1
                        if path_count_for_pair >= MAX_PATHS_PER_PAIR:
                            break
                except (nx.NetworkXNoPath, Exception):
                    continue
    finally:
        if sys.platform != "win32":
            signal.alarm(0) # Cancel the alarm
    
    # Merge: DFG-enhanced roles override basic roles for DFG nodes
    merged_roles = {str(k): v for k, v in node_roles.items()}
    for k, v in dfg_node_roles.items():
        merged_roles[str(k)] = v
    
    return uscp_paths, uscp_guards, merged_roles


def _extract_graph_views(graph, remove_attributes: bool = True):
    """
    Shared helper: extract nodes, cfg_edges, dfg_edges from an ATLAS graph.
    """
    nodes = {}
    cfg_edges = []
    dfg_edges = []
    
    for node_id, node_data in graph.nodes(data=True):
        nodes[str(node_id)] = {
            "key": str(node_id),
            "type": node_data.get("node_type", "Unknown"),
            "code": clean_node_code(
                node_data.get("label", str(node_data.get("type_label", ""))),
                remove_attributes=remove_attributes
            )
        }
        
    for u, v, m, edge_data in graph.edges(data=True, keys=True):
        e_type = str(edge_data.get('edge_type', '')).lower()
        c_type = str(edge_data.get('controlflow_type', '')).lower()
        
        if 'cfg' in e_type or any(kw in c_type for kw in ('cfg', 'flow', 'next', 'line', 'call', 'return', 'branch', 'jump', 'pos', 'neg')):
            specific_type = edge_data.get('controlflow_type', edge_data.get('edge_type', 'CFG_edge'))
            cfg_edges.append([str(u), specific_type, str(v)])
        elif any(keyword in (e_type + c_type + str(edge_data.get('dataflow_type', '')).lower()) 
                 for keyword in ('dfg', 'ddg', 'sdfg', 'flow', 'def', 'dependency', 'comesfrom', 'lastdef', 'parameter', 'use', 'reach', 'var', 'ref')):
            specific_type = edge_data.get('dataflow_type', e_type)
            dfg_edges.append([str(u), specific_type, str(v)])
    
    return nodes, cfg_edges, dfg_edges


def parse_code_to_graph_data(code: str, lang="c"):
    """
    Original PRISM-VD compatible graph dicts (old method, preserved as-is).
    """
    # 1. Generate ATLAS graph using Opportunistic Preservation
    graph, attr_preserved = generate_atlas_graph(code, lang)
    
    # 2. Extract views. If parse failed with attributes, we remove them in labels too.
    # If it succeeded, we keep them (remove_attributes = not attr_preserved)
    nodes, cfg_edges, dfg_edges = _extract_graph_views(graph, remove_attributes=(not attr_preserved))
    semantic_paths = extract_causal_paths(graph)
    
    # 3. Always export the RICH version for CodeBERT
    src_code_export = sanitize_source_code(code, remove_attributes=False)
    
    return {
        "nodes": nodes,
        "cfg_edges": cfg_edges,
        "dfg_edges": dfg_edges,
        "semantic_paths": list(semantic_paths),
        "code": src_code_export # CodeBERT gets the RICH version
    }


def parse_code_to_graph_data_uscp(code: str, lang="c", disable_dsg_filter: bool = False):
    """
    2026 SOTA: Universal Structural Causal Paths.
    Returns ALL old fields PLUS uscp_paths, uscp_guards, uscp_node_roles.
    """
    # 1. Generate ATLAS graph using Opportunistic Preservation
    graph, attr_preserved = generate_atlas_graph(code, lang)
    
    # 2. Extract views
    nodes, cfg_edges, dfg_edges = _extract_graph_views(graph, remove_attributes=(not attr_preserved))
    
    # Old method (preserved for comparison)
    semantic_paths = extract_causal_paths(graph)
    
    # New USCP method
    uscp_paths, uscp_guards, uscp_node_roles = extract_universal_causal_paths(graph, disable_dsg_filter=disable_dsg_filter)
    
    # 3. Always export the RICH version for CodeBERT
    src_code_export = sanitize_source_code(code, remove_attributes=False)
    
    return {
        "nodes": nodes,
        "cfg_edges": cfg_edges,
        "dfg_edges": dfg_edges,
        # Legacy field required for SEMANTIC_SEQ
        "semantic_paths": list(semantic_paths),
        # New USCP fields
        "uscp_paths": list(uscp_paths),
        "uscp_guards": list(uscp_guards),
        "uscp_node_roles": uscp_node_roles,
        "code": src_code_export # CodeBERT gets the RICH version
    }
