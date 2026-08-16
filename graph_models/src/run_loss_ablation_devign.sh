#!/bin/sh
# =============================================================================
# Loss Ablation Experiments for Devign (USCP split)
# =============================================================================
# Devign is approximately balanced (~45% vulnerable), so:
#   - pos_weight should be ~1.0 (no heavy class reweighting needed)
#   - focal alpha ~0.5 (equal class importance) or 0.25 (mild positive focus)
#
# Experiments:
#   1. focal_only   — Correct Focal Loss (alpha=0.5 for balanced data)
#   2. wbce_only    — Plain BCE (pos_weight=1.0, no focal, no smoothing)
#   3. wbce_pw12    — Slightly weighted BCE (pos_weight=1.2, mild push)
#   4. focal_lowdp  — Correct Focal + lower dropout (0.2 instead of 0.5)
#   5. combined_old — Legacy combined mode (your original for comparison)
#
# Usage:
#   bash run_loss_ablation_devign.sh             # runs ALL experiments
#   bash run_loss_ablation_devign.sh 1           # runs only Exp 1
#   bash run_loss_ablation_devign.sh 1 3         # runs Exp 1 and Exp 3
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${DATASET:-Devign}

# Devign data 
ln -sf "/media/user1/One Touch1/00 Data/PRISM-VD/data/processed" ~/dev

TRAIN_DATA=~/dev/${DATASET}/train_uscp.jsonlines
VAL_DATA=~/dev/${DATASET}/valid_uscp.jsonlines
TEST_DATA=~/dev/${DATASET}/test_uscp.jsonlines

DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

# Common flags (same architecture as your best Devign run, only loss changes)
COMMON="--dataset $DATASET \
  --train_data $TRAIN_DATA \
  --val_data $VAL_DATA \
  --test_data $TEST_DATA \
  --model_name microsoft/codebert-base \
  --batch_size 4 --grad_accum 16 --epochs 15 \
  --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
  --gnn rgat --slice_method cta_rwr \
  --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
  --ignore_empty_cfg --fexpn --num_bases 4 \
  --context_mode hop --context_ratio 0.5 \
  --plot_tsne --aux_loss_weight 0.8 --patience 5"

run () {
    local label="$1"; shift
    echo ""
    echo "============================================================"
    echo ">>> EXPERIMENT: $label"
    echo ">>> Command: python train_graph_models.py $*"
    echo "============================================================"
    python train_graph_models.py "$@"
    echo ">>> Done: $label"
    echo ""
}

# Decide which experiments to run
SELECTED="${@:-1 2 3 4 5}"

for EXP in $SELECTED; do
case $EXP in

# ─────────────────────────────────────────────────────────────────────────────
# Exp 1: CORRECT FOCAL ONLY (balanced dataset → alpha=0.5)
# ─────────────────────────────────────────────────────────────────────────────
# - mode=focal_only ensures pt is computed from UNWEIGHTED BCE
# - alpha=0.5 because Devign is balanced (equal weight to both classes)
# - gamma=2.0 for hard-example mining
# - pos_weight & label_smoothing ignored in focal_only mode
1)
run "Exp1: Correct Focal Only (alpha=0.5, gamma=2.0, balanced)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp1_focal_only" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp1_focal_only" \
    --loss_mode focal_only \
    --focal_alpha 0.5 --focal_gamma 2.0 \
    --pos_weight 1.0 --label_smoothing 0.0 \
    --dropout_prob 0.3
;;

# ─────────────────────────────────────────────────────────────────────────────
# Exp 2: PLAIN BCE (no weights, no focal, no smoothing)
# ─────────────────────────────────────────────────────────────────────────────
# - mode=wbce_only with pos_weight=1.0 = standard unweighted BCE
# - Simplest possible loss for a balanced dataset
# - This is your cleanest baseline
2)
run "Exp2: Plain BCE (pos_weight=1.0, no tricks)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp2_plain_bce" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp2_plain_bce" \
    --loss_mode wbce_only \
    --pos_weight 1.0 --focal_alpha 1.0 --focal_gamma 0.0 \
    --label_smoothing 0.0 \
    --dropout_prob 0.3
;;

# ─────────────────────────────────────────────────────────────────────────────
# Exp 3: SLIGHTLY WEIGHTED BCE (pos_weight=1.2, mild positive push)
# ─────────────────────────────────────────────────────────────────────────────
# - Devign is ~45% vuln, so mild pos_weight=1.2 slightly boosts recall
# - No focal, no smoothing
3)
run "Exp3: Weighted BCE (pos_weight=1.2, mild push)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp3_wbce_pw12" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp3_wbce_pw12" \
    --loss_mode wbce_only \
    --pos_weight 1.2 --focal_alpha 1.0 --focal_gamma 0.0 \
    --label_smoothing 0.0 \
    --dropout_prob 0.3
;;

# ─────────────────────────────────────────────────────────────────────────────
# Exp 4: CORRECT FOCAL + LOWER DROPOUT (0.2)
# ─────────────────────────────────────────────────────────────────────────────
# - Same as Exp 1 but dropout=0.2 instead of 0.3
# - Tests if your original dropout=0.5 was too aggressive
4)
run "Exp4: Correct Focal + Low Dropout (0.2)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp4_focal_lowdp" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp4_focal_lowdp" \
    --loss_mode focal_only \
    --focal_alpha 0.5 --focal_gamma 2.0 \
    --pos_weight 1.0 --label_smoothing 0.0 \
    --dropout_prob 0.2
;;

# ─────────────────────────────────────────────────────────────────────────────
# Exp 5: LEGACY COMBINED (your original broken mode for comparison)
# ─────────────────────────────────────────────────────────────────────────────
# - mode=combined reproduces the old FocalWCE_Loss behavior exactly
# - Same params as your original run (pos_weight=3.0 was way too high for balanced Devign!)
5)
run "Exp5: Legacy Combined (original broken — baseline)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp5_combined_old" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp5_combined_old" \
    --loss_mode combined \
    --focal_alpha 1.0 --focal_gamma 2.0 \
    --pos_weight 3.0 --label_smoothing 0.1 \
    --dropout_prob 0.5
;;

*)
echo "Unknown experiment number: $EXP (valid: 1-5)"
;;
esac
done

echo ""
echo "============================================================"
echo "✅ All selected experiments complete!"
echo "============================================================"
