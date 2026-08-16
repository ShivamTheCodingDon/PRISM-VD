#!/bin/sh
# For 2 GPUs (Model-Parallel): export CUDA_VISIBLE_DEVICES=0,1 | use --split_gpus | --batch_size 4 | --grad_accum 16
# For 1 GPU  (Single Device):  export CUDA_VISIBLE_DEVICES=0   | remove --split_gpus | --batch_size 2 | --grad_accum 32
# export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Accepts dataset name as argument (default: Reveal)
DATASET=${1:-Reveal}

# Create symlink once (safe if already exists)
ln -sf "/media/user1/One Touch/00 Data/PRISM-VD/data/processed" ~/mlaf_data

# Paths to the generated UCG jsonlines (no spaces now!)
TRAIN_DATA=~/mlaf_data/${DATASET}/train_uscp.jsonlines
VAL_DATA=~/mlaf_data/${DATASET}/valid_uscp.jsonlines
TEST_DATA=~/mlaf_data/${DATASET}/test_uscp.jsonlines

# Lowercase dataset name safely
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

run () {
    echo "=========================================================="
    echo ">>> Running: $*"
    echo "=========================================================="
    eval "$*"
    echo ">>> Done."
    echo ""
}

# RGAT + RWR
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "results_${DATASET_LOWER}_ucg_rgat_rwr_cmhop" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.15 \
    --dropout_prob 0.6 --gnn rgat --slice_method cta_rwr \
    --fusion concat --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --focal_gamma 2.0 --num_bases 4 \
    --context_mode hop --pos_weight 3.0 --focal_alpha 1.0 \
    --context_ratio 0.35 --label_smoothing 0.1 \
    --plot_tsne --aux_loss_weight 0.6 --patience 5

# RGAT + VPC
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "results_${DATASET_LOWER}_ucg_rgat_vpc_cmhop" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.15 \
    --dropout_prob 0.6 --gnn rgat --slice_method vpc \
    --fusion concat --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --focal_gamma 2.0 --num_bases 4 \
    --context_mode hop --pos_weight 3.0 --focal_alpha 1.0 \
    --context_ratio 0.35 --label_smoothing 0.1 \
    --plot_tsne --aux_loss_weight 0.6 --patience 5

# GGCN + RWR
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "results_${DATASET_LOWER}_ucg_ggcn_rwr_cmhop" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.15 \
    --dropout_prob 0.6 --gnn ggcn --slice_method cta_rwr \
    --fusion concat --pooling attention --edge_num 11 --num_layers 2 \
    --ignore_empty_cfg --fexpn --focal_gamma 2.0 --num_bases 4 \
    --context_mode hop --pos_weight 3.0 --focal_alpha 1.0 \
    --context_ratio 0.35 --label_smoothing 0.1 \
    --plot_tsne --aux_loss_weight 0.6 --patience 5

# GGCN + VPC
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "results_${DATASET_LOWER}_ucg_ggcn_vpc_cmhop" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.15 \
    --dropout_prob 0.6 --gnn ggcn --slice_method vpc \
    --fusion concat --pooling attention --edge_num 11 --num_layers 2 \
    --ignore_empty_cfg --fexpn --focal_gamma 2.0 --num_bases 4 \
    --context_mode hop --pos_weight 3.0 --focal_alpha 1.0 \
    --context_ratio 0.35 --label_smoothing 0.1 \
    --plot_tsne --aux_loss_weight 0.6 --patience 5

# RGAT + DFS
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "results_${DATASET_LOWER}_ucg_rgat_dfs_cmhop" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.15 \
    --dropout_prob 0.6 --gnn rgat --slice_method dfs \
    --fusion concat --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --focal_gamma 2.0 --num_bases 4 \
    --context_mode hop --pos_weight 3.0 --focal_alpha 1.0 \
    --context_ratio 0.35 --label_smoothing 0.1 \
    --plot_tsne --aux_loss_weight 0.6 --patience 5
