#!/bin/bash

# wandb settings
wandb_project_name="FP32"
wandb_mode="disabled" #  "online" "offline" "disabled"

# data_config
declare -a model_type_options=("lstm")  # ("lstm" "transformer")
declare -a window_size_options=(12)

# experiment settings
batch_size=256
num_epochs=100
lr=0.001
exp_mode="test"
given_timestamp="2025-03-10_16-58-16"
num_exps=1

for ((i=1; i<=num_exps; i++)); do
    for model_type in "${model_type_options[@]}"; do
        for window_size in "${window_size_options[@]}"; do
                exp_base_save_dir=$(printf "exp_records/fp32/%s/%d-ws" "$model_type" "$window_size")
                export WINDOW_SIZE="$window_size"
                echo "Running Experiment: Task=$model_type,  Window Size=$window_size"
                set -x  
                python main.py \
                    --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
                    --model_type="$model_type" --window_size="$window_size" \
                    --exp_mode="$exp_mode" --exp_base_save_dir="$exp_base_save_dir" --lr="$lr" \
                    --given_timestamp="$given_timestamp" --batch_size="$batch_size" --num_epochs="$num_epochs" 
                set +x  
        done
    done
done