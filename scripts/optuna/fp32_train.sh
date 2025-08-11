#!/bin/bash

# data_config
window_size=12 # 12 24 36

# model settings 
model_type="transformer" # "transformer" "lstm"

# experiment settings
num_epochs=100

# hw settings
subset_size=1

# optuna settings
n_trials=10
optuna_hw_target="energy"

# wandb settings
wandb_mode="online" #  "online" "offline" "disabled"

# execution
exp_base_save_dir=$(printf "exp_records_optuna_${n_trials}trails/float/${window_size}/${model_type}/")
wandb_project_name="fp32_${window_size}_${model_type}_${n_trials}"
set -x  
python optuna_search.py \
    --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
    --model_type="$model_type" --exp_base_save_dir="$exp_base_save_dir" \
    --num_epochs="$num_epochs" --n_trials="$n_trials" \
    --subset_size="$subset_size" --optuna_hw_target="$optuna_hw_target" \
    --window_size="$window_size" 
set +x  
