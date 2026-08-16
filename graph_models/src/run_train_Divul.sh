#!/bin/sh
# For 2 GPUs (Model-Parallel): export CUDA_VISIBLE_DEVICES=0,1 | use --split_gpus | --batch_size 4 | --grad_accum 16
# For 1 GPU  (Single Device):  export CUDA_VISIBLE_DEVICES=0   | remove --split_gpus | --batch_size 2 | --grad_accum 32
# export CUDA_VISIBLE_DEVICES=0,1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Accepts dataset name as argument (default: BigVul)
DATASET=${1:-BigVul}

rm -rf ~/mlaf_data

# Ensure symlinks exist (set up once)
[ ! -e ~/mlaf_data ] && ln -s "/media/user1/One Touch1/00 Data/PRISM-VD" ~/mlaf_data

# Paths to the generated UCG jsonlines
TRAIN_DATA=~/mlaf_data/data/processed/${DATASET}/train_uscp.jsonlines
VAL_DATA=~/mlaf_data/data/processed/${DATASET}/valid_uscp.jsonlines
TEST_DATA=~/mlaf_data/data/processed/${DATASET}/test_uscp.jsonlines

# Lowercase dataset name safely
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

run () {
    echo "=========================================================="
    echo ">>> Running: $*"
    echo "=========================================================="
    "$@"
    echo ">>> Done."
    echo ""
}

# # =============================================================================
# # RGAT + RWR — Focal Loss (alpha=0.92, gamma=3.0)
# # Best for high imbalance: alpha~17/18 pushes toward minority (vuln) class
# # =============================================================================
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "bv_res/results_${DATASET_LOWER}_ucg_rgat_rwr_focal" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_rwr_focal" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --num_bases 4 \
#     --context_mode hop --context_ratio 0.5 \
#     --loss_mode focal_only --focal_alpha 0.92 --focal_gamma 3.0 \
#     --pos_weight 1.0 --label_smoothing 0.1 \
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# # =============================================================================
# # RGAT + RWR — Weighted BCE (pos_weight=3.0, 1:3 ratio — matches DiverseVul)
# # =============================================================================
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "bv_res/results_${DATASET_LOWER}_ucg_rgat_rwr_wbce_pw3" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_rwr_wbce_pw3" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --num_bases 4 \
#     --context_mode hop --context_ratio 0.5 \
#     --loss_mode wbce_only --pos_weight 3.0 \
#     --focal_alpha 1.0 --focal_gamma 0.0 --label_smoothing 0.2 \
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# # =============================================================================
# # RGAT + VPC — Focal Loss (alpha=0.92, gamma=3.0)
# # =============================================================================
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "bv_res/results_${DATASET_LOWER}_ucg_rgat_vpc_focal" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_vpc_focal" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --num_bases 4 \
#     --context_mode hop --context_ratio 0.5 \
#     --loss_mode focal_only --focal_alpha 0.92 --focal_gamma 3.0 \
#     --pos_weight 1.0 --label_smoothing 0.1 \
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# # =============================================================================
# # RGAT + VPC — Weighted BCE (pos_weight=3.0, 1:3 ratio — matches DiverseVul)
# # =============================================================================
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "bv_res/results_${DATASET_LOWER}_ucg_rgat_vpc_wbce_pw3" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_vpc_wbce_pw3" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --num_bases 4 \
#     --context_mode hop --context_ratio 0.5 \
#     --loss_mode wbce_only --pos_weight 3.0 \
#     --focal_alpha 1.0 --focal_gamma 0.0 --label_smoothing 0.2 \
#     --plot_tsne --aux_loss_weight 0.75 --patience 5


run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "bv_res_v2/results_${DATASET_LOWER}_ucg_rgat_vpc_wbce" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_vpc_wbce" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
    --dropout_prob 0.3 --gnn rgat --slice_method vpc \
    --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --num_bases 4 \
    --context_mode hop --context_ratio 0.5 \
    --loss_mode wbce_only --pos_weight 3.0 \
    --focal_alpha 1.0 --focal_gamma 2.0 --label_smoothing 0.2 \
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "bv_res_v2/results_${DATASET_LOWER}_ucg_rgat_cta_wbce" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_cta_wbce" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --num_bases 4 \
    --context_mode hop --context_ratio 0.5 \
    --loss_mode wbce_only --pos_weight 3.0 \
    --focal_alpha 1.0 --focal_gamma 2.0 --label_smoothing 0.2 \
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "bv_res_v2/results_${DATASET_LOWER}_ucg_rgat_vpc_focal" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_vpc_focal" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
    --dropout_prob 0.3 --gnn rgat --slice_method vpc \
    --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --num_bases 4 \
    --context_mode hop --context_ratio 0.5 \
    --loss_mode focal_only --pos_weight 3.0 \
    --focal_alpha 1.0 --focal_gamma 2.0 --label_smoothing 0.2 \
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "bv_res_v2/results_${DATASET_LOWER}_ucg_rgat_cta_focal" \
    --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/BigVulW/weights_${DATASET_LOWER}_ucg_rgat_cta_focal" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.05 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --num_bases 4 \
    --context_mode hop --context_ratio 0.5 \
    --loss_mode focal_only --pos_weight 3.0 \
    --focal_alpha 1.0 --focal_gamma 2.0 --label_smoothing 0.2 \
    --aux_loss_weight 0.75 --patience 5