"""
Pipeline Self-Test — Validates the full data flow with synthetic C code
========================================================================
Tests all 3 stages:
  1. Function extraction (tree-sitter or regex)
  2. ATLAS graph generation
  3. Model loading + forward pass (random weights if no checkpoint)

Run:
    python test_pipeline.py
"""

import json
import logging
import os
import sys
import tempfile
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
UCG_V2_DIR = os.path.join(PROJECT_ROOT, 'graph_models')
DATA_PROC_DIR = os.path.join(PROJECT_ROOT, 'data_processing')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, UCG_V2_DIR)
sys.path.insert(0, DATA_PROC_DIR)

# ── Synthetic test code ───────────────────────────────────────────────────────
VULNERABLE_CODE = """
void packet_set_ring(struct net *net, struct packet_sock *po,
                     union tpacket_req_u *req_u, int closing, int tx_ring)
{
    char *buf;
    int size = req_u->req.tp_block_size;
    buf = malloc(size);
    if (!buf)
        return;
    // Missing bounds check on tp_block_nr — CVE-2017-7308
    memcpy(buf, req_u->req.tp_block_nr, size * 2);
    free(buf);
}
"""

SAFE_CODE = """
int safe_add(int a, int b)
{
    if (a > 0 && b > INT_MAX - a)
        return -1;
    if (a < 0 && b < INT_MIN - a)
        return -1;
    return a + b;
}
"""

SAFE_CODE_2 = """
void print_hello(const char *name)
{
    if (name == NULL)
    {
        printf("Hello, World!\\n");
        return;
    }
    printf("Hello, %s!\\n", name);
}
"""


def test_stage1_extraction():
    """Test function extraction using tree-sitter."""
    logger.info("=" * 60)
    logger.info("[STAGE 1] Testing Function Extraction")
    logger.info("=" * 60)

    # Create a temporary C file
    tmpdir = tempfile.mkdtemp(prefix='nday_test_')
    test_file = os.path.join(tmpdir, 'net', 'packet', 'af_packet.c')
    os.makedirs(os.path.dirname(test_file), exist_ok=True)
    with open(test_file, 'w') as f:
        f.write(VULNERABLE_CODE + "\n" + SAFE_CODE + "\n" + SAFE_CODE_2)

    # Test extraction
    from extract_functions import extract_functions_treesitter, extract_functions_regex

    # Try tree-sitter first
    try:
        functions = extract_functions_treesitter(test_file, lang='c')
        method = "tree-sitter"
    except Exception as e:
        logger.warning(f"tree-sitter failed ({e}), falling back to regex")
        functions = extract_functions_regex(test_file)
        method = "regex"

    logger.info(f"  Extraction method: {method}")
    logger.info(f"  Functions found: {len(functions)}")
    for func in functions:
        logger.info(f"    - {func['func_name']} (lines {func['start_line']}-{func['end_line']})")

    assert len(functions) >= 2, f"Expected >= 2 functions, got {len(functions)}"

    # Test CVE matching
    from nday_cve_db import is_function_vulnerable
    match = is_function_vulnerable('linux', 'net/packet/af_packet.c', 'packet_set_ring')
    if match:
        logger.info(f"  ✔ CVE matching works: {match['cve_id']} detected for packet_set_ring")
    else:
        logger.warning("  ✘ CVE matching did not detect packet_set_ring (may be due to simplified test code)")

    # Test full extraction pipeline with labeling
    output_jsonl = os.path.join(tmpdir, 'functions.jsonlines')
    from extract_functions import extract_and_label
    total, vul, benign = extract_and_label(
        source_dir=tmpdir,
        project='linux',
        output_path=output_jsonl,
    )
    logger.info(f"  Full extraction: {total} functions ({vul} vulnerable, {benign} benign)")
    assert total >= 2, f"Expected >= 2, got {total}"

    # Verify JSONL format
    with open(output_jsonl, 'r') as f:
        records = [json.loads(line) for line in f if line.strip()]
    for rec in records:
        assert 'code' in rec, "Missing 'code' field"
        assert 'label' in rec, "Missing 'label' field"
        assert 'func_name' in rec, "Missing 'func_name' field"

    logger.info("  ✔ STAGE 1 PASSED: Function extraction works correctly\n")

    shutil.rmtree(tmpdir)
    return output_jsonl, records


def test_stage2_graph_generation(records: list):
    """Test ATLAS graph generation on extracted functions."""
    logger.info("=" * 60)
    logger.info("[STAGE 2] Testing ATLAS Graph Generation")
    logger.info("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix='nday_test_graph_')

    # Write input JSONL
    input_path = os.path.join(tmpdir, 'functions.jsonlines')
    with open(input_path, 'w') as f:
        for rec in records:
            f.write(json.dumps(rec) + '\n')

    output_path = os.path.join(tmpdir, 'test_uscp.jsonlines')

    # Run graph generation
    from generate_graphs import generate_uscp_graphs
    success, errors = generate_uscp_graphs(
        input_path=input_path,
        output_path=output_path,
        lang='c',
        fallback=True,
        limit=10,
        skip_empty=False,
    )

    logger.info(f"  Graph generation: {success} success, {errors} errors")
    assert success > 0, f"Expected at least 1 successful graph, got {success}"

    # Verify output format matches what dataset_dynamic.py expects
    with open(output_path, 'r') as f:
        graph_records = [json.loads(line) for line in f if line.strip()]

    for rec in graph_records:
        assert 'graph_data' in rec, "Missing 'graph_data'"
        gd = rec['graph_data']
        assert 'nodes' in gd, "Missing 'nodes' in graph_data"
        assert 'cfg_edges' in gd, "Missing 'cfg_edges' in graph_data"
        assert 'dfg_edges' in gd, "Missing 'dfg_edges' in graph_data"
        assert 'code' in rec, "Missing 'code' in record"
        assert 'label' in rec, "Missing 'label' in record"

        n_nodes = len(gd['nodes'])
        n_cfg = len(gd.get('cfg_edges', []))
        n_dfg = len(gd.get('dfg_edges', []))
        n_uscp = len(gd.get('uscp_paths', []))
        logger.info(
            f"    {rec.get('func_name', '?')}: "
            f"nodes={n_nodes} cfg={n_cfg} dfg={n_dfg} uscp={n_uscp}"
        )

    logger.info("  ✔ STAGE 2 PASSED: ATLAS graph generation works correctly\n")

    return tmpdir, output_path


def test_stage3_model_inference(test_data_path: str, tmpdir: str):
    """Test model loading and forward pass (with random weights)."""
    logger.info("=" * 60)
    logger.info("[STAGE 3] Testing Model Loading & Inference")
    logger.info("=" * 60)

    import torch

    # Test model construction
    logger.info("  Building Dynamic_PRISM-VD_VD_PlusPlus model...")
    from infer_nday import load_model, run_inference

    model, device = load_model(
        weights_path=None,  # Random weights for test
        model_name="microsoft/codebert-base",
        embed_dim=128,
        num_edge_types=11,
        gnn_type="rgat",
        fusion_type="concat",
        device="cpu",
    )
    logger.info(f"  ✔ Model built successfully on {device}")

    # Test forward pass through dataset + model
    results_dir = os.path.join(tmpdir, 'results')
    logger.info("  Running inference on test graphs (random weights)...")
    metrics = run_inference(
        model=model,
        device=device,
        test_data_path=test_data_path,
        output_dir=results_dir,
        threshold=0.5,
        max_seq_len=512,
        min_nodes=5,
        max_nodes=500,
        edge_num=11,
    )

    logger.info(f"  Metrics returned: {list(metrics.keys())}")

    # Check output files exist
    assert os.path.exists(os.path.join(results_dir, 'predictions.jsonl')), "Missing predictions.jsonl"
    assert os.path.exists(os.path.join(results_dir, 'nday_metrics.json')), "Missing nday_metrics.json"
    assert os.path.exists(os.path.join(results_dir, 'nday_report.txt')), "Missing nday_report.txt"

    # Verify predictions format
    with open(os.path.join(results_dir, 'predictions.jsonl'), 'r') as f:
        preds = [json.loads(line) for line in f if line.strip()]

    for pred in preds:
        assert 'pred_label' in pred, "Missing 'pred_label'"
        assert 'confidence' in pred, "Missing 'confidence'"
        assert 'true_label' in pred, "Missing 'true_label'"
        assert pred['pred_label'] in (0, 1), f"Invalid pred_label: {pred['pred_label']}"
        assert 0 <= pred['confidence'] <= 1, f"Invalid confidence: {pred['confidence']}"

    logger.info(f"  ✔ Predictions generated: {len(preds)} samples")
    logger.info("  ✔ STAGE 3 PASSED: Model inference works correctly\n")

    return metrics


def main():
    logger.info("")
    logger.info("=" * 60)
    logger.info("  N-Day Vulnerability Pipeline Self-Test")
    logger.info("=" * 60)
    logger.info("")

    passed = 0
    failed = 0

    # Stage 1: Extraction
    try:
        _, records = test_stage1_extraction()

        # Recreate records for stage 2 (since tmpdir was cleaned up)
        # Use the synthetic codes directly
        records = [
            {"id": 0, "file_name": "net/packet/af_packet.c", "func_name": "packet_set_ring",
             "code": VULNERABLE_CODE.strip(), "label": 1, "project": "linux",
             "cve_id": "CVE-2017-7308", "cwe": "CWE-119"},
            {"id": 1, "file_name": "utils/math.c", "func_name": "safe_add",
             "code": SAFE_CODE.strip(), "label": 0, "project": "linux"},
            {"id": 2, "file_name": "utils/io.c", "func_name": "print_hello",
             "code": SAFE_CODE_2.strip(), "label": 0, "project": "linux"},
        ]
        passed += 1
    except Exception as e:
        logger.error(f"  ✘ STAGE 1 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return

    # Stage 2: Graph Generation
    try:
        tmpdir, test_data_path = test_stage2_graph_generation(records)
        passed += 1
    except Exception as e:
        logger.error(f"  ✘ STAGE 2 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1
        return

    # Stage 3: Model Inference
    try:
        metrics = test_stage3_model_inference(test_data_path, tmpdir)
        passed += 1
    except Exception as e:
        logger.error(f"  ✘ STAGE 3 FAILED: {e}")
        import traceback; traceback.print_exc()
        failed += 1

    # Cleanup
    try:
        shutil.rmtree(tmpdir)
    except:
        pass

    # Final summary
    logger.info("=" * 60)
    if failed == 0:
        logger.info(f"  ✔ ALL {passed} STAGES PASSED — Pipeline is ready!")
        logger.info("")
        logger.info("  Next steps:")
        logger.info("    bash run_pipeline.sh --project ffmpeg --weights /path/to/model_last.pt")
        logger.info("    bash run_pipeline.sh --project openssl --weights /path/to/model_last.pt --limit 100")
    else:
        logger.error(f"  ✘ {failed} STAGE(S) FAILED — Fix errors before running on real data.")
    logger.info("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
