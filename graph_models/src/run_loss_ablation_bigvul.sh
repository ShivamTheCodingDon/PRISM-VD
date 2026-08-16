#!/bin/sh
# =============================================================================
# Loss Ablation Experiments for BigVul (USCP split)
# =============================================================================

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

DATASET=${DATASET:-BigVul}

# Symlink (safe if already exists)
rm -rf ~/mlaf_data
[ ! -e ~/mlaf_data ] && ln -s "/media/user1/One Touch1/00 Data/PRISM-VD" ~/mlaf_data

TRAIN_DATA=~/mlaf_data/data/processed/${DATASET}/train_uscp.jsonlines
VAL_DATA=~/mlaf_data/data/processed/${DATASET}/valid_uscp.jsonlines
TEST_DATA=~/mlaf_data/data/processed/${DATASET}/test_uscp.jsonlines

DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

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
  --plot_tsne --aux_loss_weight 1 --patience 5"

run() {
    label="$1"
    shift
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
if [ $# -eq 0 ]; then
  SELECTED="1 2 3 4 5"
else
  SELECTED="$@"
fi

for EXP in $SELECTED; do
case $EXP in
1)
  run "Exp1: Correct Focal (alpha=0.75, gamma=2.0)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp1_focal_only" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp1_focal_only" \
    --loss_mode focal_only \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --pos_weight 1.0 --label_smoothing 0.0 \
    --dropout_prob 0.3
  ;;
2)
  run "Exp2: Weighted BCE (pos_weight=10)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp2_wbce_pw10" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp2_wbce_pw10" \
    --loss_mode wbce_only \
    --pos_weight 10.0 --focal_alpha 1.0 --focal_gamma 0.0 \
    --label_smoothing 0.0 \
    --dropout_prob 0.3
  ;;
3)
  run "Exp3: Weighted BCE (pos_weight=17)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp3_wbce_pw17" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp3_wbce_pw17" \
    --loss_mode wbce_only \
    --pos_weight 17.0 --focal_alpha 1.0 --focal_gamma 0.0 \
    --label_smoothing 0.0 \
    --dropout_prob 0.3
  ;;
4)
  run "Exp4: Correct Focal + Low Dropout (alpha=0.75, dp=0.2)" \
    $COMMON \
    --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_exp4_focal_lowdp" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_exp4_focal_lowdp" \
    --loss_mode focal_only \
    --focal_alpha 0.75 --focal_gamma 2.0 \
    --pos_weight 1.0 --label_smoothing 0.0 \
    --dropout_prob 0.2
  ;;
5)
  run "Exp5: Legacy Combined (baseline)" \
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
