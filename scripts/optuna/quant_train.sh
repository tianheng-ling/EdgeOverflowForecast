#!/bin/bash

# data_config
declare -a window_size_options=(6) # 6 12 24

# model settings
model_type="lstm" # "transformer" "lstm" 

# experiment settings
num_epochs=100

# hw settings
subset_size=1

# optuna settings
n_trials=100
optuna_hw_target="energy"


# wandb settings
wandb_mode="online" #  "online" "offline" "disabled"

# execution
for window_size in "${window_size_options[@]}"; do
    exp_base_save_dir=$(printf "exp_records_optuna_new_${n_trials}trails/quant/${window_size}ws/${model_type}/")
    wandb_project_name="quant_new_${window_size}ws_${model_type}_${n_trials}"
    set -x  
    python optuna_search.py \
        --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
        --model_type="$model_type" --exp_base_save_dir="$exp_base_save_dir" \
        --num_epochs="$num_epochs" --n_trials="$n_trials" \
        --subset_size="$subset_size" --is_qat --optuna_hw_target="$optuna_hw_target" \
        --window_size="$window_size" --do_hw_simulation
    set +x  
done