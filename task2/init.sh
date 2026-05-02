#!/bin/bash

# 存储后台进程的 PID
declare -a pids=()

# 函数：清理后台进程
cleanup() {
    echo "收到终止信号，正在停止所有后台进程..."
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "终止进程 PID: $pid"
            kill "$pid"
            # 等待退出，超时则强制终止
            for i in {1..10}; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                echo "PID $pid 仍未退出，强制终止..."
                kill -9 "$pid"
            fi
        fi
    done
    echo "所有进程已停止。注意qwen需要手动释放"
}

# 捕获终止信号
trap cleanup EXIT INT TERM

echo "启动服务..."

# 启动第一个服务
conda run -n HRI——test --no-capture-output python common/skills/audio_module/chat.py &
pids+=($!)
echo "chat.py 启动，PID: ${pids[-1]}"

# 启动第二个服务
conda run -n HRI——test --no-capture-output python common/skills/audio_module/asr_server_sensesmall.py &
pids+=($!)
echo "asr_server_sensesmall.py 启动，PID: ${pids[-1]}"



echo "所有服务已启动，PID 列表: ${pids[@]}"
echo "按 Ctrl+C 终止所有服务..."

# 等待所有后台进程
wait

. task1/release.sh
echo "注意qwen需要手动释放"
