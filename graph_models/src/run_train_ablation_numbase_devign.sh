#!/bin/sh
# For 2 GPUs (Model-Parallel): export CUDA_VISIBLE_DEVICES=0,1 | use --split_gpus | --batch_size 4 | --grad_accum 16
# For 1 GPU  (Single Device):  export CUDA_VISIBLE_DEVICES=0   | remove --split_gpus | --batch_size 2 | --grad_accum 32
# export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Accepts dataset name as argument (default: Reveal)
DATASET=${1:-Devign}

rm -rf ~/mlaf_data_dev

# Ensure symlinks exist (set up once)
[ ! -e ~/mlaf_data_dev ] && ln -s "/media/user1/One Touch1/00 Data/PRISM-VD" ~/mlaf_data_dev

# Paths to the generated UCG jsonlines (no spaces now!)
TRAIN_DATA=~/mlaf_data_dev/data/processed/${DATASET}/train_uscp.jsonlines
VAL_DATA=~/mlaf_data_dev/data/processed/${DATASET}/valid_uscp.jsonlines
TEST_DATA=~/mlaf_data_dev/data/processed/${DATASET}/test_uscp.jsonlines

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

#########NumBaseAblation#######################

# # RGAT + RWR
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_nb2" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_nb2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 2 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.8 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_nb4" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_nb4" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 4 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_nb8" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_nb8" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 8 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_full" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_full" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# # RGAT + VPC
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_nb2" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_nb2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 2 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_nb4" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_nb4" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 4 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_nb8" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_nb8" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 8 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_nbfull" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_nbfull" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# # RGAT + DFS
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_dfs_nb2" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_dfs_nb2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method dfs \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 2 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_dfs_nb4" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_dfs_nb4" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method dfs \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 4 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_dfs_nb8" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_dfs_nb8" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method dfs \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 --num_bases 8 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_dfs_nbfull" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_dfs_nbfull" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method dfs \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 --max_guards 5\
#     --context_ratio 0.5 --label_smoothing 0.0 \
#     --plot_tsne --aux_loss_weight 0.8 --patience 5


# ## RWR
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_cr0" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_cr0" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.0 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_cr0.25" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_cr0.25" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.25 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_cr0.5" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_cr0.5" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_cr0.75" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_cr0.75" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.75 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_rwr_cr1" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_rwr_cr1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 1 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5


# ## VPC
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_cr0" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_cr0" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.0 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_cr0.25" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_cr0.25" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.25 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_cr0.5" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_cr0.5" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_cr0.75" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_cr0.75" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.75 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_cr1" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_cr1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 1 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5


# # Num layers

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_l1" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_l1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_vpc_l2" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_l2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# ## RWR layer

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_cta_l1" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_cta_l1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl/results_${DATASET_LOWER}_ucg_rgat_cta_l2" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_cta_l2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 1 --patience 5



# Loss 
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.25" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.25" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.25 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_loss0.25" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_vpc_loss0.25" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.25 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.5" \
#         --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.5" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.5 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_loss0.5" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_loss0.5" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.5 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.75" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_loss0.75" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_loss0.75" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_loss0.75" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# # Label Smoth
# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.1" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_smoth0.1" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_smoth0.1" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_smoth0.2" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_smoth0.2" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# # Max gaurd

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_maxg3" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_maxg3" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 3\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_maxg3" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_maxg3" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 3\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_maxg7" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_maxg7" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_cta_rwr_maxg7" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_cta_rwr_maxg7" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5


# # GAT

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gat_cta_max7" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_gat_cta_max7" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gat_vpc_max7" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_gat_vpc_max7" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gat --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# # GATv2

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gatv2_cta" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gatv2_cta" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gatv2 --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gatv2_vpc" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gatv2_vpc" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gatv2 --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# # GIN

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gin_cta" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gin_cta" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gin --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gin_vpc" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gin_vpc" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gin --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# # SAGE

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_sage_cta" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_sage_cta" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn sage --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_sage_vpc" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_sage_vpc" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn sage --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# # GCN

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gcn_cta" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gcn_cta" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gcn --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_gnn/results_${DATASET_LOWER}_ucg_gcn_vpc" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/gnn_abl/weights_${DATASET_LOWER}_ucg_gcn_vpc" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn gcn --slice_method vpc \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.2  --max_guards 7\
#     --plot_tsne --aux_loss_weight 0.6 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2_mcd" \
#     --save_weights_dir "/media/user1/One Touch1/00 Data/PRISM-VD/dvloss/weights_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2_mcd" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1\
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
#     --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5 --mc_dropout\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5


# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2_no_slice" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --no_slice \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
#     --pos_weight 1.0 --focal_alpha 1.0 \
#     --label_smoothing 0.0\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

# run python train_graph_models.py \
#     --dataset "$DATASET" \
#     --train_data "$TRAIN_DATA" \
#     --val_data "$VAL_DATA" \
#     --test_data "$TEST_DATA" \
#     --output_dir "Devign_Abl_loss/results_${DATASET_LOWER}_ucg_rgat_vpc_smoth0.2_no_slice_mcd" \
#     --model_name microsoft/codebert-base \
#     --batch_size 4 --grad_accum 16 --epochs 15 \
#     --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
#     --dropout_prob 0.3 --gnn rgat --no_slice \
#     --fusion gated --pooling attention --edge_num 11 --num_layers 1 \
#     --ignore_empty_cfg --loss_mode wbce_only --focal_gamma 0.0 \
#     --pos_weight 1.0 --focal_alpha 1.0 \
#     --label_smoothing 0.0 --mc_dropout\
#     --plot_tsne --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "Dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_con_att" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion concat --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_con_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion concat --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5
    
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_gated_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion gated --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5
    
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_weighted_attn" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion weighted --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5
    
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_weighted_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion weighted --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5
    
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_att_attn" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion attention --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5
    
run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_att_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion attention --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_bilinear_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion bilinear --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_bilinear_attn" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion bilinear --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_moe_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion moe --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_moe_attn" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion moe --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_highway_mean" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion highway --pooling mean --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_fus_highway_attn" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion highway --pooling attention --edge_num 11 --num_layers 1 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5

run python train_graph_models.py \
    --dataset "$DATASET" \
    --train_data "$TRAIN_DATA" \
    --val_data "$VAL_DATA" \
    --test_data "$TEST_DATA" \
    --output_dir "dev_Abl_fusion/results_${DATASET_LOWER}_ucg_rgat_rwr_l2rgat" \
    --model_name microsoft/codebert-base \
    --batch_size 4 --grad_accum 16 --epochs 15 \
    --lr 2e-5 --lr_scratch 1e-4 --weight_decay 0.1 \
    --dropout_prob 0.3 --gnn rgat --slice_method cta_rwr \
    --fusion gated --pooling attention --edge_num 11 --num_layers 2 \
    --ignore_empty_cfg --fexpn --loss_mode wbce_only --focal_gamma 0.0 \
    --context_mode random --pos_weight 1.0 --focal_alpha 1.0 \
    --context_ratio 0.5 --label_smoothing 0.0  --max_guards 5\
    --aux_loss_weight 0.75 --patience 5