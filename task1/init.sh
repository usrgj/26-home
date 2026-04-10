#!/bin/bash
# 提前各个部署模型
# 使用方法: ./deploy_models.sh [start|stop|restart|status]

set -e  # 遇到错误立即退出

# ==================== 配置区域 ====================

# 视觉语言模型配置
QWEN_VL="Qwen3.5-VL-4B"  # 模型名称
MODEL1_DIR="./model1"  # 模型1的工作目录
MODEL1_SCRIPT="app.py"  # 启动脚本
MODEL1_PORT=8004
MODEL1_HOST="0.0.0.0"
MODEL1_VENV="venv1"  # 虚拟环境目录

# tts模型配置
MODEL2_NAME="model_service_2"
MODEL2_DIR="./model2"  # 模型2的工作目录
MODEL2_SCRIPT="app.py"  # 启动脚本
MODEL2_PORT=8002
MODEL2_HOST="0.0.0.0"
MODEL2_VENV="venv2"  # 虚拟环境目录

# tts模型配置
MODEL2_NAME="model_service_2"
MODEL2_DIR="./model2"  # 模型2的工作目录
MODEL2_SCRIPT="app.py"  # 启动脚本
MODEL2_PORT=8002
MODEL2_HOST="0.0.0.0"
MODEL2_VENV="venv2"  # 虚拟环境目录

# 日志目录
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# PID文件目录
PID_DIR="./pids"
mkdir -p "$PID_DIR"

# ==================== 颜色输出 ====================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 工具函数 ====================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查进程是否运行
check_process() {
    local pid_file=$1
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0  # 进程运行中
        fi
    fi
    return 1  # 进程未运行
}

# ==================== 启动模型1 ====================

start_model1() {
    log_info "正在启动 $MODEL1_NAME..."
    
    # 检查是否已经运行
    if check_process "$PID_DIR/${MODEL1_NAME}.pid"; then
        log_warning "$MODEL1_NAME 已经在运行中"
        return 0
    fi
    
    # 检查目录是否存在
    if [ ! -d "$MODEL1_DIR" ]; then
        log_error "$MODEL1_DIR 目录不存在"
        return 1
    fi
    
    cd "$MODEL1_DIR"
    
    # 激活虚拟环境
    if [ -d "$MODEL1_VENV" ]; then
        source "$MODEL1_VENV/bin/activate"
        log_info "已激活虚拟环境: $MODEL1_VENV"
    else
        log_warning "虚拟环境 $MODEL1_VENV 不存在，使用系统Python"
    fi
    
    # 启动服务（后台运行）
    nohup python "$MODEL1_SCRIPT" \
        --host "$MODEL1_HOST" \
        --port "$MODEL1_PORT" \
        > "../$LOG_DIR/${MODEL1_NAME}.log" 2>&1 &
    
    local pid=$!
    echo "$pid" > "../$PID_DIR/${MODEL1_NAME}.pid"
    
    # 等待服务启动
    sleep 3
    
    if check_process "../$PID_DIR/${MODEL1_NAME}.pid"; then
        log_success "$MODEL1_NAME 启动成功 (PID: $pid, Port: $MODEL1_PORT)"
    else
        log_error "$MODEL1_NAME 启动失败，请检查日志: $LOG_DIR/${MODEL1_NAME}.log"
        return 1
    fi
    
    cd - > /dev/null
}

# ==================== 启动模型2 ====================

start_model2() {
    log_info "正在启动 $MODEL2_NAME..."
    
    # 检查是否已经运行
    if check_process "$PID_DIR/${MODEL2_NAME}.pid"; then
        log_warning "$MODEL2_NAME 已经在运行中"
        return 0
    fi
    
    # 检查目录是否存在
    if [ ! -d "$MODEL2_DIR" ]; then
        log_error "$MODEL2_DIR 目录不存在"
        return 1
    fi
    
    cd "$MODEL2_DIR"
    
    # 激活虚拟环境
    if [ -d "$MODEL2_VENV" ]; then
        source "$MODEL2_VENV/bin/activate"
        log_info "已激活虚拟环境: $MODEL2_VENV"
    else
        log_warning "虚拟环境 $MODEL2_VENV 不存在，使用系统Python"
    fi
    
    # 启动服务（后台运行）
    nohup python "$MODEL2_SCRIPT" \
        --host "$MODEL2_HOST" \
        --port "$MODEL2_PORT" \
        > "../$LOG_DIR/${MODEL2_NAME}.log" 2>&1 &
    
    local pid=$!
    echo "$pid" > "../$PID_DIR/${MODEL2_NAME}.pid"
    
    # 等待服务启动
    sleep 3
    
    if check_process "../$PID_DIR/${MODEL2_NAME}.pid"; then
        log_success "$MODEL2_NAME 启动成功 (PID: $pid, Port: $MODEL2_PORT)"
    else
        log_error "$MODEL2_NAME 启动失败，请检查日志: $LOG_DIR/${MODEL2_NAME}.log"
        return 1
    fi
    
    cd - > /dev/null
}

# ==================== 停止模型 ====================

stop_model() {
    local model_name=$1
    local pid_file="$PID_DIR/${model_name}.pid"
    
    log_info "正在停止 $model_name..."
    
    if check_process "$pid_file"; then
        local pid=$(cat "$pid_file")
        kill "$pid"
        
        # 等待进程结束
        local count=0
        while ps -p "$pid" > /dev/null 2>&1; do
            sleep 1
            count=$((count + 1))
            if [ $count -gt 10 ]; then
                log_warning "进程未响应，强制终止..."
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
        done
        
        rm -f "$pid_file"
        log_success "$model_name 已停止"
    else
        log_warning "$model_name 未运行"
    fi
}

# ==================== 查看状态 ====================

show_status() {
    echo ""
    echo "==================== 模型服务状态 ===================="
    
    # 模型1状态
    if check_process "$PID_DIR/${MODEL1_NAME}.pid"; then
        local pid=$(cat "$PID_DIR/${MODEL1_NAME}.pid")
        echo -e "${GREEN}✓${NC} $MODEL1_NAME: ${GREEN}运行中${NC} (PID: $pid, Port: $MODEL1_PORT)"
    else
        echo -e "${RED}✗${NC} $MODEL1_NAME: ${RED}已停止${NC}"
    fi
    
    # 模型2状态
    if check_process "$PID_DIR/${MODEL2_NAME}.pid"; then
        local pid=$(cat "$PID_DIR/${MODEL2_NAME}.pid")
        echo -e "${GREEN}✓${NC} $MODEL2_NAME: ${GREEN}运行中${NC} (PID: $pid, Port: $MODEL2_PORT)"
    else
        echo -e "${RED}✗${NC} $MODEL2_NAME: ${RED}已停止${NC}"
    fi
    
    echo "===================================================="
    echo ""
}

# ==================== 主控制逻辑 ====================

case "${1:-}" in
    start)
        log_info "开始部署模型服务..."
        start_model1
        start_model2
        show_status
        log_success "所有模型服务部署完成！"
        ;;
    
    stop)
        log_info "停止所有模型服务..."
        stop_model "$MODEL1_NAME"
        stop_model "$MODEL2_NAME"
        show_status
        ;;
    
    restart)
        log_info "重启所有模型服务..."
        stop_model "$MODEL1_NAME"
        stop_model "$MODEL2_NAME"
        sleep 2
        start_model1
        start_model2
        show_status
        ;;
    
    status)
        show_status
        ;;
    
    *)
        echo "用法: $0 {start|stop|restart|status}"
        echo ""
        echo "命令说明:"
        echo "  start   - 启动所有模型服务"
        echo "  stop    - 停止所有模型服务"
        echo "  restart - 重启所有模型服务"
        echo "  status  - 查看服务运行状态"
        exit 1
        ;;
esac

exit 0