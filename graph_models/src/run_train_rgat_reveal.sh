#!/bin/sh
# For 2 GPUs (Model-Parallel): export CUDA_VISIBLE_DEVICES=0,1 | use --split_gpus | --batch_size 4 | --grad_accum 16
# For 1 GPU  (Single Device):  export CUDA_VISIBLE_DEVICES=0   | remove --split_gpus | --batch_size 2 | --grad_accum 32
# export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Accepts dataset name as argument (default: Devign)
DATASET=${1:-Reveal}

# Paths to the generated UCG jsonlines
TRAIN_DATA="../../data/processed/${DATASET}/train_uscp.jsonlines"
VAL_DATA="../../data/processed/${DATASET}/valid_uscp.jsonlines"
TEST_DATA="../../data/processed/${DATASET}/test_uscp.jsonlines"

# Lowercase dataset name safely
DATASET_LOWER=$(echo "$DATASET" | tr '[:upper:]' '[:lower:]')

run () {
    echo "=========================================================="
    echo ">>> Running: $1"
    echo "=========================================================="
    eval "$1"
    echo ">>> Done."
    echo ""
}



run "python train_graph_models.py \
    --dataset $DATASET \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir results_${DATASET_LOWER}_ucg_rgat_rwr_v2 \
    --model_name microsoft/codebert-base \
    --batch_size 4 \
    --grad_accum 16 \
    --epochs 15 \
    --lr 5e-5 \
    --lr_scratch 1e-4 \
    --weight_decay 0.1 \
    --dropout_prob 0.5 \
    --gnn rgat \
    --slice_method cta_rwr \
    --fusion concat \
    --pooling attention \
    --edge_num 11 \
    --num_layers 1 \
    --ignore_empty_cfg \
    --fexpn \
    --focal_gamma 2.0 \
    --num_bases 4 \
    --pos_weight 3.0 \
    --focal_alpha 1.0 \
    --context_ratio 0.35 \
    --fixed_threshold 0.5 \
    --label_smoothing 0.2 \
    --plot_tsne \
    --aux_loss_weight 0.6 \
    --patience 5"

run "python train_graph_models.py \
    --dataset $DATASET \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir results_${DATASET_LOWER}_ucg_rgat_vpc_v2 \
    --model_name microsoft/codebert-base \
    --batch_size 4 \
    --grad_accum 16 \
    --epochs 15 \
    --lr 5e-5 \
    --lr_scratch 1e-4 \
    --weight_decay 0.1 \
    --dropout_prob 0.5 \
    --gnn rgat \
    --slice_method vpc \
    --fusion concat \
    --pooling attention \
    --edge_num 11 \
    --num_layers 1 \
    --ignore_empty_cfg \
    --fexpn \
    --focal_gamma 2.0 \
    --num_bases 4 \
    --pos_weight 3.0 \
    --focal_alpha 1.0 \
    --context_ratio 0.35 \
    --label_smoothing 0.0 \
    --plot_tsne \
    --aux_loss_weight 0.6 \  
    --patience 5"



run "python train_graph_models.py \
    --dataset $DATASET \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir results_${DATASET_LOWER}_ucg_ggcn_rwr_v2 \
    --model_name microsoft/codebert-base \
    --batch_size 4 \
    --grad_accum 16 \
    --epochs 15 \
    --lr 5e-5 \
    --lr_scratch 1e-4 \
    --weight_decay 0.1 \
    --dropout_prob 0.5 \
    --gnn ggcn \
    --slice_method cta_rwr \
    --fusion concat \
    --pooling attention \
    --edge_num 11 \
    --num_layers 2 \
    --ignore_empty_cfg \
    --fexpn \
    --focal_gamma 2.0 \
    --num_bases 4 \
    --pos_weight 3.0 \
    --focal_alpha 1.0 \
    --context_ratio 0.3 \
    --label_smoothing 0.0 \
    --plot_tsne \
    --aux_loss_weight 0.6 \
    --patience 5"

run "python train_graph_models.py \
    --dataset $DATASET \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir results_${DATASET_LOWER}_ucg_ggcn_vpc_v2 \
    --model_name microsoft/codebert-base \
    --batch_size 4 \
    --grad_accum 16 \
    --epochs 15 \
    --lr 5e-5 \
    --lr_scratch 1e-4 \
    --weight_decay 0.1 \
    --dropout_prob 0.5 \
    --gnn ggcn \
    --slice_method vpc \
    --fusion concat \
    --pooling attention \
    --edge_num 11 \
    --num_layers 2 \
    --ignore_empty_cfg \
    --fexpn \
    --focal_gamma 2.0 \
    --num_bases 4 \
    --pos_weight 3.0 \
    --focal_alpha 1.0 \
    --context_ratio 0.3 \
    --label_smoothing 0.0 \
    --plot_tsne \
    --aux_loss_weight 0.6 \
    --patience 5"


run "python train_graph_models.py \
    --dataset $DATASET \
    --train_data $TRAIN_DATA \
    --val_data $VAL_DATA \
    --test_data $TEST_DATA \
    --output_dir results_${DATASET_LOWER}_ucg_rgat_dfs_v2 \
    --model_name microsoft/codebert-base \
    --batch_size 4 \
    --grad_accum 16 \
    --epochs 15 \
    --lr 5e-5 \
    --lr_scratch 1e-4 \
    --weight_decay 0.1 \
    --dropout_prob 0.5 \
    --gnn rgat \
    --slice_method dfs \
    --fusion concat \
    --pooling attention \
    --edge_num 11 \
    --num_layers 1 \
    --ignore_empty_cfg \
    --fexpn \
    --focal_gamma 2.0 \
    --num_bases 4 \
    --pos_weight 3.0 \
    --focal_alpha 1.0 \
    --context_ratio 0.3 \
    --label_smoothing 0.0 \
    --plot_tsne \
    --aux_loss_weight 0.6 \
    --patience 5"