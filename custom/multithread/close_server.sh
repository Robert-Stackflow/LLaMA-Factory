#!/bin/bash

# 停止所有实例
for pid_file in pids/server_*.pid
do
    if [ -f "$pid_file" ]; then
        PID=$(cat "$pid_file")
        echo "停止实例，PID: $PID"
        kill -9 $PID && rm -f "$pid_file"
    fi
done

echo "所有实例已停止"
