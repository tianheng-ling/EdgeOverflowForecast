#!/bin/bash

# wandb settings
wandb_project_name="Quant_LSTM_debug"
wandb_mode="online" #  "online" "offline" "disabled"

# data_config
declare -a model_type_options=("lstm") # "transformer")
declare -a window_size_options=(6) # 24)

# quantization settings
declare -a quant_bits_options=(4) # 6 4)

# experiment settings
batch_size=256
num_epochs=30
lr=0.001
exp_mode="train"
given_timestamp=""
num_exps=1

# HW simulation settings
subset_size=1

for ((i=1; i<=num_exps; i++)); do
    for model_type in "${model_type_options[@]}"; do
        for window_size in "${window_size_options[@]}"; do
            for quant_bits in "${quant_bits_options[@]}"; do
                exp_base_save_dir=$(printf "exp_records/quant/%s/%d-ws/%d-bit" "$model_type" "$window_size" "$quant_bits")
                export WINDOW_SIZE="$window_size"
                echo "Running Experiment: Model Type=$model_type,  Window Size=$window_size, Quant Bit=$quant_bits"
                set -x  
                python main.py \
                    --wandb_project_name="$wandb_project_name" --wandb_mode="$wandb_mode" \
                    --model_type="$model_type" --window_size="$window_size" \
                    --exp_mode="$exp_mode" --exp_base_save_dir="$exp_base_save_dir" --lr="$lr" \
                    --given_timestamp="$given_timestamp" --batch_size="$batch_size" --num_epochs="$num_epochs" \
                    --quant_bits="$quant_bits" --is_qat --subset_size="$subset_size"
                set +x  
            done
        done
    done
done