#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
LOG_DIR="$SCRIPT_DIR/logs"
RES_DIR="$SCRIPT_DIR/results"

mkdir -p "$LOG_DIR"

PORT=8082
MAX_SAMPLE_COUNT=2000
EVAL_STEP=10
RUN_ONLY=false

usage() {
    echo "用法: $0 [-p PORT] [-c CONFIG_PATH] [-d DATA_PATH] [-r] [-t wait_time]"
    echo "  -r  仅运行数据处理脚本，不启动或关闭服务"
    echo "例子: $0 -p 8080 -c custom/inference/StackChange_DPO.yaml -d ../Datasets/StackChange/DPO/10000.json -t 180"
    exit 1
}

while getopts ":p:c:d:r:t:" opt; do
    case "$opt" in
        p) PORT=$OPTARG ;;
        c) CONFIG_PATH=$OPTARG ;;
        d) DATA_PATH=$OPTARG ;;
        r) RUN_ONLY=true ;;
        t) WAIT_TIME=$OPTARG ;;
        *) usage ;;
    esac
done

if [ -z "$DATA_PATH" ]; then
    echo "错误: 必须提供数据文件路径 (-d)"
    usage
fi

if [ "$RUN_ONLY" = false ] && [ -z "$CONFIG_PATH" ]; then
    echo "错误: 完整流程需要提供模型配置文件 (-c)"
    usage
fi

LOG_FILE=$(basename "$DATA_PATH" | sed 's/\.[^.]*$//')_eval.log
LOG_FILE="$LOG_DIR/$LOG_FILE"

if [ "$RUN_ONLY" = true ]; then
    echo "仅运行数据处理脚本..."
    nohup python custom/test/evaluation.py \
        -p "$PORT" \
        -m "$MAX_SAMPLE_COUNT" \
        -s "$EVAL_STEP" \
        -d "$DATA_PATH" > "$LOG_FILE" 2>&1 &
    echo "数据处理脚本已运行，日志保存在 $LOG_FILE"
    exit 0
fi

MODEL_LOG_FILE=$(basename "$CONFIG_PATH" | sed 's/\.[^.]*$//')_model.log
MODEL_LOG_FILE="$LOG_DIR/$MODEL_LOG_FILE"

echo "启动模型API服务，端口: $PORT 配置文件: $CONFIG_PATH"
API_PORT=$PORT \
nohup llamafactory-cli api "$CONFIG_PATH" > "$MODEL_LOG_FILE" 2>&1 &
API_PID=$!

WAIT_TIME=${WAIT_TIME:-30}
echo "等待模型API服务启动...(等待时间: $WAIT_TIME s)"
for ((i = WAIT_TIME; i > 0; i--)); do
    printf "\r剩余时间: %2d 秒" "$i"
    sleep 1
done
echo -e "\n模型API服务已启动，PID: $API_PID"

RES_FILE=$(basename "$DATA_PATH" | sed 's/\.[^.]*$//')_eval.json
RES_FILE="$RES_DIR/$RES_FILE"

echo "运行数据处理脚本..."
echo "样本数: $MAX_SAMPLE_COUNT 步长: $EVAL_STEP 结果文件: $RES_FILE"
nohup python custom/test/evaluation.py \
    -p "$PORT" \
    -m "$MAX_SAMPLE_COUNT" \
    -s "$EVAL_STEP" \
    -r "$RES_FILE" \
    -d "$DATA_PATH" > "$LOG_FILE" 2>&1 &
CLIENT_PID=$!
echo "数据处理脚本已运行，PID: $CLIENT_PID"
