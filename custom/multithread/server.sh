#!/bin/bash

# 基础端口号
BASE_PORT=8081

# CUDA 设备分配规则（可根据需求修改）
CUDA_DEVICES=("0" "1" "2" "3")  # 假设有4个GPU，循环分配

# 启动的实例数量
NUM_INSTANCES=12

# 检查 llamafactory-cli 是否存在
if ! command -v llamafactory-cli &> /dev/null; then
    echo "错误：找不到 llamafactory-cli 命令，请检查安装路径！"
    exit 1
fi

# 循环启动实例
for i in $(seq 0 $((NUM_INSTANCES - 1)))
do
    PORT=$((BASE_PORT + i)) # 计算端口号
    CUDA_DEVICE=${CUDA_DEVICES[$((i % ${#CUDA_DEVICES[@]}))]} # 循环分配 CUDA 设备

    echo "启动实例 $i, 端口: $PORT, CUDA设备: $CUDA_DEVICE"

    # 启动服务实例，并将日志输出到文件
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE API_PORT=$PORT \
    nohup llamafactory-cli api ../../inference/llama3_3.2_3B_instruct.yaml > "logs/server_$i.log" 2>&1 &

    # 记录实例的 PID，方便后续管理
    echo $! > "pids/server_$i.pid"
done

echo "所有实例已启动"

# 打印 PID 文件位置
echo "实例 PID 文件已保存为 server_*.pid"
