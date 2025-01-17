#!/bin/bash

# 默认参数值
LOG_FILE=""
TAIL_LOG=false
CUDA_VISIBLE_DEVICES=""
CONFIG_FILE=""

# 打印使用方法
usage() {
    echo "使用方法: $0 CONFIG_FILE [-l LOG_FILE] [-t] [-g GPU]"
    echo "  CONFIG_FILE 配置文件路径 (必需)"
    echo "  -l          日志文件路径 (可选)"
    echo "  -t          实时查看日志 (可选)"
    echo "  -g          指定 CUDA 可见 GPU (可选)"
    exit 1
}

# 校验至少提供配置文件路径
if [ $# -lt 1 ]; then
    usage
fi

# 提取配置文件路径
CONFIG_FILE="$1"
shift

# 参数解析
while getopts ":l:tg:" opt; do
    case "$opt" in
        l) LOG_FILE="$OPTARG" ;;
        t) TAIL_LOG=true ;;
        g) CUDA_VISIBLE_DEVICES="$OPTARG" ;;
        *) usage ;;
    esac
done

# 校验配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

# 设置日志文件默认值
CONFIG_DIR=$(dirname "$CONFIG_FILE")
if [ -z "$LOG_FILE" ]; then
    BASE_NAME=$(basename "$CONFIG_FILE" .yaml)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="$CONFIG_DIR/logs/$BASE_NAME"
    LOG_FILE="$LOG_DIR/$TIMESTAMP.log"
    mkdir -p "$LOG_DIR"
    echo "未提供日志文件路径, 默认日志文件为: $LOG_FILE"
else
    LOG_DIR=$(dirname "$LOG_FILE")
    mkdir -p "$LOG_DIR"
fi

# 启动训练任务
echo "开始运行训练任务, 配置文件: $CONFIG_FILE, 日志文件: $LOG_FILE"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" FORCE_TORCHRUN=1 nohup llamafactory-cli train "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
else
    FORCE_TORCHRUN=1 nohup llamafactory-cli train "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &
fi

PID=$!
echo "训练任务已启动, PID: $PID"
echo "PID: $PID" >> "$LOG_FILE"

# 实时查看日志
if [ "$TAIL_LOG" = true ]; then
    tail -f "$LOG_FILE"
fi

# custom/train/train.sh custom/train/boolq/lora_1B_train1_seed2.yaml -g 1
# custom/train/train.sh custom/train/boolq/lora_1B_train1_seed3.yaml -g 2
# custom/train/train.sh custom/train/boolq/lora_1B_train2_seed1.yaml -g 3