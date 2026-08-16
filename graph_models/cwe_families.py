"""
CWE-Family Aware Seeding for CTA-RWR Slicing
=============================================
Auto-detects the most relevant CWE families per sample by scanning node labels
against curated CWE-family-specific source/sink/propagator dictionaries.

Sources: atlas_adapter.py SENSORS list (lines 288-318), USCP_ROLE_MAP (lines 375-421)
         CWE definitions from MITRE (https://cwe.mitre.org)

Used by: dataset_dynamic.py → _cta_rwr_slice()
"""

# =============================================================================
# CWE FAMILY DEFINITIONS
# Each family has:
#   sources     : Functions where untrusted/dangerous data ENTERS
#   sinks       : Functions where data CAUSES a vulnerability
#   propagators : Functions that PASS data through (taint-preserving)
#   roles       : USCP structural roles that are high-signal for this CWE
#   patterns    : Label patterns indicative of this CWE
# =============================================================================

CWE_FAMILIES = {
    # ── CWE-119/787/125: Buffer Overflow / Out-of-bounds Read/Write ──────────
    "MEMORY_BUFFER": {
        "cwe_ids": [119, 125, 131, 787],
        "sources": {
            "malloc", "calloc", "realloc", "alloca", "mmap", "new", 
            "sbrk", "brk", "shmat", "shmget", "mprotect",
        },
        "sinks": {
            "memcpy", "memmove", "memset", "memcmp", "memchr", "memccpy",
            "wmemcpy", "wmemmove", "wmemset", "wmemcmp", "wmemchr",
            "strcpy", "strncpy", "strcat", "strncat",
            "wcscpy", "wcsncpy", "wcscat", "wcsncat", "wcstok", "wcslen",
            "strtok", "strtok_r", "realpath",
            "sprintf", "snprintf", "vsprintf", "vsnprintf", "vasprintf", "swprintf",
            "gets", "fgets", "fread", "read", "recv", "recvfrom", "pread",
        },
        "propagators": {
            "strlen", "strnlen", "sizeof", "memchr",
        },
        "roles": {"DEREF"},
        "patterns": ["[", "*", "->", "&", "sizeof"],
    },

    # ── CWE-416/415: Use-After-Free / Double-Free ───────────────────────────
    "USE_AFTER_FREE": {
        "cwe_ids": [416, 415],
        "sources": {
            "malloc", "calloc", "realloc", "alloca", "new", "mmap",
        },
        "sinks": {
            "free", "delete", "munmap", "shmdt", "realloc",
        },
        "propagators": {
            "memcpy", "memmove", "strcpy", "strncpy",
        },
        "roles": {"DEREF", "CALL"},
        "patterns": ["*", "->", "free", "&"],
    },

    # ── CWE-476: NULL Pointer Dereference ────────────────────────────────────
    "NULL_POINTER": {
        "cwe_ids": [476],
        "sources": {
            "malloc", "calloc", "realloc", "fopen", "dlopen",
            "strstr", "strchr", "strrchr", "strpbrk",
            "getenv", "dlsym", "argv", "cin",
        },
        "sinks": set(),   # Any dereference of unchecked pointer
        "propagators": set(),
        "roles": {"DEREF"},
        "patterns": ["*", "->", "[", "NULL", "nullptr", "&"],
    },

    # ── CWE-78/134/89: Injection (Command/Format/SQL) ────────────────────────
    "INJECTION": {
        "cwe_ids": [78, 134, 89, 90],
        "sources": {
            "scanf", "fscanf", "sscanf", "vscanf", "vfscanf", "vsscanf",
            "read", "recv", "recvfrom", "recvmsg", "fread", "fgets",
            "getenv", "getc", "getchar", "fgetc", "pread", "readv", 
            "readlink", "readlinkat", "msgrcv", "getmsg", "getpmsg",
            "argv", "cin", "fgetwc", "fgetws", "getwc", "getwchar",
        },
        "sinks": {
            "system", "popen", "exec", "execl", "execlp", "execle",
            "execv", "execvp", "execvpe",
            "printf", "fprintf", "vprintf", "vfprintf", "wprintf", "fwprintf",
            "sprintf", "vsprintf", "syslog", "puts", "fputs", "perror",
        },
        "propagators": {
            "strcpy", "strncpy", "strcat", "strncat", "snprintf", "vasprintf", "swprintf",
        },
        "roles": {"CALL"},
        "patterns": ["%", "$", "{", "}"],
    },

    # ── CWE-190/191: Integer Overflow / Wrap-around ──────────────────────────
    "INTEGER_ERROR": {
        "cwe_ids": [190, 191, 128, 681],
        "sources": {
            "scanf", "fscanf", "sscanf", "vscanf", "vfscanf", "vsscanf",
            "getenv", "argv", "cin", "read", "fread",
        },
        "sinks": {
            "malloc", "calloc", "realloc", "alloca", "new",
            "memcpy", "memmove", "memset", "strncpy", "strncat",
        },
        "propagators": {
            "strlen", "strnlen", "sizeof",
        },
        "roles": {"ASSIGN"},
        "patterns": ["+", "-", "*", "/", "unsigned", "signed", "static_cast"],
    },

    # ── CWE-250/114/732: Privilege / Dynamic Loading ─────────────────────────
    "PRIVILEGE": {
        "cwe_ids": [250, 114, 732, 377],
        "sources": {
            "getuid", "getgid", "geteuid", "getegid",
        },
        "sinks": {
            "setuid", "setgid", "seteuid", "setegid",
            "chown", "chmod", "fchmod", "chroot",
            "dlopen", "dlsym", "dlclose",
            "tmpnam", "tempnam", "mktemp", "tmpfile",
            "fork", "vfork", "clone", "signal", "raise", "exit", "_Exit", "abort",
        },
        "propagators": set(),
        "roles": {"CALL"},
        "patterns": ["sudo", "root", "chmod", "0777"],
    },
}

# Flat sets for quick "is this relevant at all?" checks
ALL_CWE_SOURCES = set()
ALL_CWE_SINKS = set()
ALL_CWE_PROPAGATORS = set()

for _fam in CWE_FAMILIES.values():
    ALL_CWE_SOURCES.update(_fam["sources"])
    ALL_CWE_SINKS.update(_fam["sinks"])
    ALL_CWE_PROPAGATORS.update(_fam["propagators"])


def auto_detect_cwe_families(node_labels, uscp_roles_dict=None, node_keys=None, top_k=2):
    """
    Auto-detect the most relevant CWE families for a given code sample.
    
    Scores each CWE family based on the number of source and sink nodes
    found in the sample's node labels. Returns the top-K families.
    
    Args:
        node_labels:    List of cleaned node label strings
        uscp_roles_dict: Dict of node_key -> USCP role (optional)
        node_keys:      List of node keys corresponding to node_labels
        top_k:          Number of top families to return (default: 2)
    
    Returns:
        List of (family_name, family_dict, score) tuples, sorted by score descending
    """
    family_scores = {}
    
    for fam_name, fam_data in CWE_FAMILIES.items():
        source_count = 0
        sink_count = 0
        role_count = 0
        pattern_count = 0
        
        for i, label in enumerate(node_labels):
            label_lower = label.lower()
            
            # Count source hits
            if any(s in label_lower for s in fam_data["sources"]):
                source_count += 1
            
            # Count sink hits
            if any(s in label_lower for s in fam_data["sinks"]):
                sink_count += 1
            
            # Count pattern hits
            if any(p in label for p in fam_data["patterns"]):
                pattern_count += 1
            
            # Count structural role matches
            if uscp_roles_dict and node_keys and i < len(node_keys):
                nid = node_keys[i]
                if uscp_roles_dict.get(nid, "OTHER") in fam_data["roles"]:
                    role_count += 1
        
        # Score = source hits * sink hits + role bonus + pattern bonus
        # The multiplication ensures we need BOTH sources AND sinks for high score
        score = (source_count * max(sink_count, 1)) + (role_count * 0.5) + (pattern_count * 0.3)
        
        # Special case: NULL_POINTER has no explicit sinks, rely on patterns + roles
        if fam_name == "NULL_POINTER" and sink_count == 0:
            score = source_count * (pattern_count * 0.5 + role_count * 0.5)
        
        family_scores[fam_name] = (fam_data, score)
    
    # Sort by score, return top-K
    ranked = sorted(family_scores.items(), key=lambda x: x[1][1], reverse=True)
    result = [(name, data, score) for name, (data, score) in ranked[:top_k] if score > 0]
    
    # Fallback: if no family scored > 0, return the two most general families
    if not result:
        result = [
            ("MEMORY_BUFFER", CWE_FAMILIES["MEMORY_BUFFER"], 0.1),
            ("INJECTION", CWE_FAMILIES["INJECTION"], 0.1),
        ]
    
    return result


def get_cwe_sources_sinks(node_keys, node_labels, uscp_roles_dict=None, top_k=2):
    """
    Main entry point: auto-detect CWE families and return combined
    source/sink/propagator node indices.
    
    Returns:
        source_indices:     Set of node indices that are taint sources
        sink_indices:       Set of node indices that are exploit sinks
        propagator_indices: Set of node indices that pass taint through  
        guard_indices:      Set of node indices that are GUARD conditions
        detected_families:  List of detected family names
    """
    families = auto_detect_cwe_families(
        node_labels, uscp_roles_dict, node_keys, top_k=top_k
    )
    
    # Combine source/sink/propagator sets from top-K families
    combined_sources = set()
    combined_sinks = set()
    combined_propagators = set()
    combined_roles = set()
    combined_patterns = []
    detected_names = []
    
    for fam_name, fam_data, score in families:
        detected_names.append(fam_name)
        combined_sources.update(fam_data["sources"])
        combined_sinks.update(fam_data["sinks"])
        combined_propagators.update(fam_data["propagators"])
        combined_roles.update(fam_data["roles"])
        combined_patterns.extend(fam_data["patterns"])
    
    # Classify nodes
    source_indices = set()
    sink_indices = set()
    propagator_indices = set()
    guard_indices = set()
    
    for i, (nid, label) in enumerate(zip(node_keys, node_labels)):
        label_lower = label.lower()
        role = (uscp_roles_dict or {}).get(nid, "OTHER")
        
        # Guards
        if role == "GUARD":
            guard_indices.add(i)
            continue
        
        # Sources
        if any(s in label_lower for s in combined_sources):
            source_indices.add(i)
        
        # Sinks
        if any(s in label_lower for s in combined_sinks):
            sink_indices.add(i)
        
        # Pattern-based sinks (for NULL_POINTER etc.)
        if role in combined_roles and any(p in label for p in combined_patterns):
            sink_indices.add(i)
        
        # Propagators
        if any(s in label_lower for s in combined_propagators):
            propagator_indices.add(i)
    
    return source_indices, sink_indices, propagator_indices, guard_indices, detected_names


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Relational Edge Weights for MR-RWR
# Derived from atlas_adapter.py edge_type classifications
# ─────────────────────────────────────────────────────────────────────────────

# Edge type → transition weight in Random Walk
# Higher weight = stronger signal for vulnerability analysis
EDGE_TYPE_WEIGHTS = {
    # DFG edges — strongest signal for taint tracking
    "comesFrom":    1.0,
    "parameter":    1.0,
    
    # CFG control flow — structural context
    "next":         0.6,
    "next_line":    0.5,
    "first_next_line": 0.5,
    "pos_next":     0.7,   # Branch taken = stronger signal
    "neg_next":     0.7,   # Branch not taken = stronger signal
    "jump_next":    0.4,
    "case_next":    0.6,
    "loop_control": 0.8,   # Loop bounds are vulnerability-critical
    "loop_update":  0.7,
    "loop_carried": 0.7,
    "switch_case":  0.5,
    "switch_exit":  0.4,
    "program_entry": 0.3,
    
    # Call/Return
    "function_call":     0.9,
    "function_return":   0.8,
    "method_call":       0.9,
    "method_return":     0.8,
    "constructor_call":  0.7,
    "constructor_return": 0.6,
    "implicit_return":   0.5,
    
    # Exception
    "try_next":    0.5,
    "catch_next":  0.6,
    "throw_exit":  0.7,
    
    # Synthetic edge types (from dataset_dynamic.py)
    "CAUSAL_SEQ":  0.8,
    "SEMANTIC_SEQ": 0.7,
    "GUARD_EDGE":  0.9,   # Guards are critical for vulnerability context
    "USCP_SEQ":    0.8,
    "USCP_GUARD":  0.9,
    
    # Fallback
    "UNKNOWN":     0.3,
}

def get_edge_weight(edge_type_str):
    """Get the transition weight for an edge type string."""
    # Strip |suffix if present (e.g., function_call|101 -> function_call)
    clean_type = edge_type_str.split('|')[0] if '|' in edge_type_str else edge_type_str
    return EDGE_TYPE_WEIGHTS.get(clean_type, 0.3)
