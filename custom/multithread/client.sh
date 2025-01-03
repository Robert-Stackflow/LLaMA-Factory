#!/bin/bash

MAX_WORKERS=12

START_INDEX=0

CHUNK_SIZE=1000

nohup python -m translator.api -m $MAX_WORKERS -s $START_INDEX -c $CHUNK_SIZE > translator/logs/client.log 2>&1 &

echo $! > translator/pids/client.pid

echo "客户端已启动，PID: $!"

# lsof | grep "client.log" | awk '{print $2}' | xargs kill