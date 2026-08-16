# PRISM-VD (Path-based Relational Integrated Structural Multi-modal Vulnerability Detector)

This repository contains the PRISM-VD framework, a state-of-the-art vulnerability detection platform utilizing both Graph Neural Networks (GNNs) and LLM baselines. 

The pipeline is fully divided into three core phases: **Environment Setup**, **Data Creation**, and **Model Training**.

---

## Phase 1: Environment Setup

Before running anything, ensure your environment is prepared by installing all required dependencies.

```bash
# Navigate to the project root
cd PRISM-VD/

# Install the Python requirements
pip install -r requirements.txt
```

---

## Phase 2: Data Creation & Graph Extraction

The **Data Creation** phase involves taking raw `C/C++` code and extracting multi-view structural graphs using the ATLAS adapter. We implement **USCP (Universal Structural Causal Paths)** extraction filtered by **DSG (Dangerous Structure Graph)** pruning rules to drastically reduce noise and computational limits.

To run the data creation pipeline:

```bash
cd data_processing/

# Run the SOTA USCP data preparation script
python prepare_data_uscp.py \
    --input raw_dataset.jsonlines \
    --format jsonlines \
    --output_dir ../data/processed/BigVul/ \
    --text_col "func_before" \
    --label_col "vul" \
    --lang c \
    --max_workers 8 \
    --skip_empty True
```

### What happens under the hood during this step?
1. **Sanitization**: Code is cleaned, attributes are selectively stripped, and formatting is normalized.
2. **Graph Extraction**: `atlas_adapter.py` initiates ATLAS to extract the AST, CFG, and DFG representations into a `MultiDiGraph`.
3. **DSG Filtering**: Nodes are pruned using JISA 2025 DSG rules (retaining only `call_expression`, `if_statement`, `pointer_declarator`, etc.).
4. **USCP Extraction**: AST nodes are classified into structural roles (`CALL`, `DEREF`, `ENTRY`, `EXIT`, `ASSIGN`, `GUARD`) to map causal data flow pathways between these security-relevant boundaries.

---

## Phase 3: Model Training

Once the `.jsonlines` datasets containing `uscp_paths` and graph topology are created, you can train the GNN models. The scripts support distributing the training across multiple GPUs seamlessly.

```bash
cd graph_models/src/

# Example 1: Train the RGAT model on the BigVul dataset using the Divul Script
# This uses Focal Loss and generates models inside bv_res_v2/
./run_train_Divul.sh BigVul

# Example 2: Train using Random Walk with Restart (RWR) path slicing
./run_train_rwr.sh BigVul

# Example 3: Train Hop-based models on the Devign dataset
./run_train_hop_devign.sh Devign
```

**Training Arguments Overview (Inside the Shell Scripts):**
- `--gnn`: Defines the backbone (e.g., `rgat`, `gatv2`, `gin`, `sage`, `ggcn`). *(See `graph_models/algorithms.md` for a detailed breakdown of all algorithms!)*
- `--loss_mode`: Supports highly imbalanced dataset loss architectures (`focal_only`, `wbce_only`).
- `--slice_method`: How causal paths are sliced (`vpc`, `cta_rwr`).
- `--fusion`: Graph + Text fusion approach (e.g., `gated`).
- `--pooling`: Pooling method across the graph (`attention`).

---

## Looking for the LLM Baselines?

For direct LLM-based detection (GPT-4, DeepSeek, Codestral) without graph integration, refer to the `llm_baselines/` directory and use the `run_devign.py` or `run_devign_multicpu.sh` evaluation wrappers.
