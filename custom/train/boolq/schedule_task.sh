#!/bin/bash

cd /data/xrd/Fine_Tuning/LLaMA-Factory || { echo "Failed to change directory"; exit 1; }

conda activate llamafactory || { echo "Failed to activate conda environment"; exit 1; }

run_tasks_at() {
  echo "Running tasks at $1..."
  case $1 in
    "3:00")
      custom/train/train.sh custom/train/boolq/lora_8B_train1_seed3.yaml -g 1
      custom/train/train.sh custom/train/boolq/lora_8B_train2_seed1.yaml -g 2
      custom/train/train.sh custom/train/boolq/lora_8B_train2_seed2.yaml -g 3
      ;;
    "5:30")
      custom/train/train.sh custom/train/boolq/lora_8B_train2_seed3.yaml -g 1
      custom/train/train.sh custom/train/boolq/lora_3B_train1_seed2.yaml -g 2
      custom/train/train.sh custom/train/boolq/lora_3B_train1_seed3.yaml -g 3
      ;;
    "8:00")
      custom/train/train.sh custom/train/boolq/lora_3B_train2_seed1.yaml -g 1
      custom/train/train.sh custom/train/boolq/lora_3B_train2_seed2.yaml -g 2
      custom/train/train.sh custom/train/boolq/lora_3B_train2_seed3.yaml -g 3
      ;;
  esac
}

# Parse the time passed as argument
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <time>"
  exit 1
fi

run_tasks_at "$1"

custom/train/train.sh custom/train/boolq/full_3B_train1_seed1.yaml -g 0
custom/train/train.sh custom/train/boolq/full_3B_train1_seed2.yaml -g 1
custom/train/train.sh custom/train/boolq/full_3B_train1_seed3.yaml -g 2
custom/train/train.sh custom/train/boolq/full_3B_train2_seed1.yaml -g 3


custom/train/train.sh custom/train/boolq/full_1B_train1_seed1.yaml -g 0
custom/train/train.sh custom/train/boolq/full_1B_train1_seed2.yaml -g 1
custom/train/train.sh custom/train/boolq/full_1B_train1_seed3.yaml -g 2
custom/train/train.sh custom/train/boolq/full_1B_train2_seed1.yaml -g 3