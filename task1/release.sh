#!/bin/bash

PORTS=(8001 8002 8003)

echo "检查并终止占用端口 ${PORTS[*]} 的进程..."

for port in "${PORTS[@]}"; do
    echo "--------------------"
    echo "检查端口: $port"

    # 使用 ss 获取监听该端口的 PID（更现代，比 netstat/lsof 更可靠）
    pid=$(sudo ss -tulnH '( sport = :'$port' )' 2>/dev/null | awk '{print $7}' | cut -d',' -f2)

    # 如果 ss 未返回 PID，尝试用 lsof（兼容旧系统）
    if [[ -z "$pid" ]]; then
        pid=$(sudo lsof -t -i :"$port" 2>/dev/null)
    fi

    if [[ -n "$pid" ]]; then
        echo "端口 $port 被 PID $pid 占用"
        
        # 先发送 SIGTERM（优雅退出）
        echo "发送 SIGTERM 到 PID $pid..."
        sudo kill "$pid" 2>/dev/null
        
        # 等待最多 5 秒看是否退出
        for i in {1..5}; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid 已退出。"
                break
            fi
            sleep 1
        done

        # 如果还在运行，强制 kill
        if kill -0 "$pid" 2>/dev/null; then
            echo "PID $pid 未退出，发送 SIGKILL..."
            sudo kill -9 "$pid" 2>/dev/null
        fi
    else
        echo "端口 $port 未被占用。"
    fi
done

echo "--------------------"
echo "操作完成。"