#!/bin/bash

if [ -z "$1" ]; then
    echo "错误: 必须提供配置文件路径 (CONFIG_FILE)"
    echo "使用方法: $0 CONFIG_FILE [LOG_FILE] [-t]"
    exit 1
fi

CONFIG_FILE="$1"
TAIL_LOG=false

CONFIG_DIR=$(dirname "$CONFIG_FILE")

if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 $CONFIG_FILE 不存在"
    exit 1
fi

if [ -z "$2" ] || [[ "$2" == "-t" ]]; then
    BASE_NAME=$(basename "$CONFIG_FILE" .yaml)
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    LOG_DIR="$CONFIG_DIR/logs/$BASE_NAME"
    LOG_FILE="$LOG_DIR/$TIMESTAMP.log"

    mkdir -p "$LOG_DIR"
    echo "未提供日志文件路径, 默认日志文件为: $LOG_FILE"

    if [[ "$2" == "-t" ]]; then
        TAIL_LOG=true
    fi
else
    LOG_FILE="$2"

    LOG_DIR=$(dirname "$LOG_FILE")
    mkdir -p "$LOG_DIR"

    if [[ "$3" == "-t" ]]; then
        TAIL_LOG=true
    fi
fi

echo "开始运行训练任务, 配置文件: $CONFIG_FILE, 日志文件: $LOG_FILE"
nohup llamafactory-cli train "$CONFIG_FILE" > "$LOG_FILE" 2>&1 &

PID=$!
echo "训练任务已启动, PID: $PID"
echo "PID: $PID" >> "$LOG_FILE"

if [ "$TAIL_LOG" = true ]; then
    tail -f "$LOG_FILE"
fi
