#!/usr/bin/env bash

# 本脚本由 ChatGPT 生成，负责 Task1 的模型部署。

# 统一预部署 Task1 所需的模型与服务。
# 支持启动的服务：
# - vl: 视觉语言模型服务
# - tts: Edge-TTS 服务
# - asr: SenseVoiceSmall ASR 服务
#
# 使用方式：
#   ./task1/init.sh start [all|vl|tts|asr ...]
#   ./task1/init.sh stop [all|vl|tts|asr ...]
#   ./task1/init.sh restart [all|vl|tts|asr ...]
#   ./task1/init.sh status [all|vl|tts|asr ...]

set -euo pipefail

# ==================== 路径与目录 ====================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/logs}"
PID_DIR="${PID_DIR:-$SCRIPT_DIR/pids}"

mkdir -p "$LOG_DIR" "$PID_DIR"

# ==================== 可配置项 ====================

SERVICES=(vl tts asr)
CONDA_PROFILE=""

# 视觉语言模型服务
VLM_NAME="${VLM_NAME:-Qwen3.5-VL-4B}"
VLM_HOST="${VLM_HOST:-0.0.0.0}"
VLM_PORT="${VLM_PORT:-8004}"
VLM_WORKDIR="${VLM_WORKDIR:-$REPO_ROOT}"
VLM_CONDA_ENV="${VLM_CONDA_ENV:-vllm_deploy}"
VLM_MODEL_PATH="${VLM_MODEL_PATH:-/home/blinx/api_servers/models/Qwen3.5-4B-FP8}"
VLM_STARTUP_TIMEOUT="${VLM_STARTUP_TIMEOUT:-120}"
VLM_GPU_MEMORY_UTILIZATION="${VLM_GPU_MEMORY_UTILIZATION:-0.7}"
VLM_TENSOR_PARALLEL_SIZE="${VLM_TENSOR_PARALLEL_SIZE:-1}"
VLM_MAX_MODEL_LEN="${VLM_MAX_MODEL_LEN:-32768}"

# 语音模块共享配置
AUDIO_MODULE_DIR="${AUDIO_MODULE_DIR:-$REPO_ROOT/common/skills/audio_module}"
AUDIO_CONDA_ENV="${AUDIO_CONDA_ENV:-llm}"

# TTS 服务
TTS_NAME="${TTS_NAME:-Edge-TTS}"
TTS_PORT="${TTS_PORT:-8002}"
TTS_WORKDIR="${TTS_WORKDIR:-$AUDIO_MODULE_DIR}"
TTS_SCRIPT="${TTS_SCRIPT:-chat.py}"
TTS_STARTUP_TIMEOUT="${TTS_STARTUP_TIMEOUT:-30}"

# ASR 服务
ASR_NAME="${ASR_NAME:-SenseVoiceSmall-ASR}"
ASR_PORT="${ASR_PORT:-8001}"
ASR_WORKDIR="${ASR_WORKDIR:-$AUDIO_MODULE_DIR}"
ASR_SCRIPT="${ASR_SCRIPT:-asr_server_sensesmall.py}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$ASR_WORKDIR/models/SenseVoiceSmall}"
ASR_STARTUP_TIMEOUT="${ASR_STARTUP_TIMEOUT:-180}"

# ==================== 颜色输出 ====================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ==================== 日志函数 ====================

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

# ==================== 基础工具函数 ====================

# 打印帮助信息。
usage() {
    cat <<EOF
用法: $0 {start|stop|restart|status} [all|vl|tts|asr ...]

命令说明:
  start    启动指定服务，默认启动全部
  stop     停止指定服务，默认停止全部
  restart  重启指定服务，默认重启全部
  status   查看指定服务状态，默认查看全部

服务说明:
  vl       Qwen VL 服务，默认端口 ${VLM_PORT}
  tts      Edge-TTS 服务，默认端口 ${TTS_PORT}
  asr      SenseVoiceSmall ASR 服务，默认端口 ${ASR_PORT}

常用环境变量覆盖:
  VLM_MODEL_PATH     覆盖视觉语言模型路径
  AUDIO_MODULE_DIR   覆盖 common/skills/audio_module 所在目录
  VLM_CONDA_ENV      覆盖 VL 服务 conda 环境名
  AUDIO_CONDA_ENV    覆盖 ASR/TTS 服务 conda 环境名
  ASR_MODEL_DIR      覆盖 SenseVoiceSmall 模型目录
EOF
}

# 检查命令是否存在。
require_command() {
    local command_name=$1
    if ! command -v "$command_name" >/dev/null 2>&1; then
        log_error "未找到命令: $command_name"
        exit 1
    fi
}

# 初始化 conda shell 配置，后续用 exec 保持 PID 正确。
init_conda() {
    if [ -n "$CONDA_PROFILE" ]; then
        return
    fi

    require_command conda

    local conda_base
    conda_base="$(conda info --base 2>/dev/null || true)"
    if [ -z "$conda_base" ]; then
        log_error "无法解析 conda base 目录，请确认 conda 可用"
        exit 1
    fi

    CONDA_PROFILE="$conda_base/etc/profile.d/conda.sh"
    if [ ! -f "$CONDA_PROFILE" ]; then
        log_error "未找到 conda 初始化脚本: $CONDA_PROFILE"
        exit 1
    fi
}

# 读取 PID 文件中的进程号。
read_pid() {
    local pid_file=$1
    [ -f "$pid_file" ] || return 1
    tr -d '[:space:]' < "$pid_file"
}

# 判断指定 PID 是否还在运行。
is_pid_running() {
    local pid=$1
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# 通过 PID 文件检查服务进程。
check_process() {
    local pid_file=$1
    local pid

    pid="$(read_pid "$pid_file" 2>/dev/null || true)"
    [ -n "$pid" ] && is_pid_running "$pid"
}

# 清理过期的 PID 文件。
cleanup_stale_pid() {
    local pid_file=$1
    if [ -f "$pid_file" ] && ! check_process "$pid_file"; then
        rm -f "$pid_file"
    fi
}

# 判断端口是否已经进入监听状态。
port_is_listening() {
    local port=$1

    if command -v ss >/dev/null 2>&1; then
        ss -ltnH | awk '{print $4}' | grep -Eq "[:.]${port}$"
        return $?
    fi

    if command -v netstat >/dev/null 2>&1; then
        netstat -ltn 2>/dev/null | awk 'NR > 2 {print $4}' | grep -Eq "[:.]${port}$"
        return $?
    fi

    return 1
}

# 打印最近的日志片段，便于快速定位启动失败原因。
show_recent_log() {
    local log_file=$1
    if [ -f "$log_file" ]; then
        echo "----- 最近日志: $log_file -----"
        tail -n 20 "$log_file" || true
        echo "--------------------------------"
    fi
}

# ==================== 服务元信息 ====================

# 返回服务显示名称。
service_name() {
    local service=$1
    case "$service" in
        vl)  echo "$VLM_NAME" ;;
        tts) echo "$TTS_NAME" ;;
        asr) echo "$ASR_NAME" ;;
        *)
            log_error "未知服务: $service"
            exit 1
            ;;
    esac
}

# 返回服务监听端口。
service_port() {
    local service=$1
    case "$service" in
        vl)  echo "$VLM_PORT" ;;
        tts) echo "$TTS_PORT" ;;
        asr) echo "$ASR_PORT" ;;
        *)
            log_error "未知服务: $service"
            exit 1
            ;;
    esac
}

# 返回服务启动超时。
service_timeout() {
    local service=$1
    case "$service" in
        vl)  echo "$VLM_STARTUP_TIMEOUT" ;;
        tts) echo "$TTS_STARTUP_TIMEOUT" ;;
        asr) echo "$ASR_STARTUP_TIMEOUT" ;;
        *)
            log_error "未知服务: $service"
            exit 1
            ;;
    esac
}

# 返回服务工作目录。
service_workdir() {
    local service=$1
    case "$service" in
        vl)  echo "$VLM_WORKDIR" ;;
        tts) echo "$TTS_WORKDIR" ;;
        asr) echo "$ASR_WORKDIR" ;;
        *)
            log_error "未知服务: $service"
            exit 1
            ;;
    esac
}

# 返回服务对应的 PID 文件。
service_pid_file() {
    local service=$1
    echo "$PID_DIR/${service}.pid"
}

# 返回服务对应的日志文件。
service_log_file() {
    local service=$1
    echo "$LOG_DIR/${service}.log"
}

# 返回服务对应的脚本路径（若有）。
service_script_path() {
    local service=$1
    case "$service" in
        tts) echo "$TTS_WORKDIR/$TTS_SCRIPT" ;;
        asr) echo "$ASR_WORKDIR/$ASR_SCRIPT" ;;
        *)
            echo ""
            ;;
    esac
}

# 判断服务是否已经对外可用。
service_ready() {
    local service=$1
    local port

    port="$(service_port "$service")"
    case "$service" in
        tts)
            if command -v curl >/dev/null 2>&1; then
                curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null 2>&1
                return $?
            fi
            ;;
    esac

    port_is_listening "$port"
}

# ==================== 启动与停止 ====================

# 在指定 conda 环境内后台启动命令，并用 exec 确保 PID 对应真实进程。
launch_in_conda_env() {
    local workdir=$1
    local env_name=$2
    local log_file=$3
    local pid_file=$4
    shift 4

    init_conda

    local command_string
    local shell_command

    printf -v command_string '%q ' "$@"
    printf -v shell_command 'source %q && conda activate %q && exec %s' \
        "$CONDA_PROFILE" "$env_name" "$command_string"

    (
        cd "$workdir"
        nohup bash -lc "$shell_command" > "$log_file" 2>&1 &
        echo $! > "$pid_file"
    )
}

# 等待服务启动完成。
wait_for_service() {
    local service=$1
    local timeout=$2
    local pid_file
    local deadline

    pid_file="$(service_pid_file "$service")"
    deadline=$((SECONDS + timeout))

    while [ "$SECONDS" -lt "$deadline" ]; do
        if ! check_process "$pid_file"; then
            cleanup_stale_pid "$pid_file"
            return 1
        fi

        if service_ready "$service"; then
            return 0
        fi

        sleep 1
    done

    return 1
}

# 启动视觉语言模型服务。
start_vl() {
    local service="vl"
    local pid_file log_file timeout

    pid_file="$(service_pid_file "$service")"
    log_file="$(service_log_file "$service")"
    timeout="$(service_timeout "$service")"

    if check_process "$pid_file"; then
        log_warning "$(service_name "$service") 已经在运行中"
        return 0
    fi

    cleanup_stale_pid "$pid_file"

    if [ ! -d "$VLM_WORKDIR" ]; then
        log_error "VL 工作目录不存在: $VLM_WORKDIR"
        return 1
    fi

    if [[ "$VLM_MODEL_PATH" == /* || "$VLM_MODEL_PATH" == ./* || "$VLM_MODEL_PATH" == ../* ]] && [ ! -e "$VLM_MODEL_PATH" ]; then
        log_error "VL 模型路径不存在: $VLM_MODEL_PATH"
        return 1
    fi

    log_info "正在启动 $(service_name "$service")..."
    launch_in_conda_env \
        "$VLM_WORKDIR" \
        "$VLM_CONDA_ENV" \
        "$log_file" \
        "$pid_file" \
        env VLLM_USE_V1=0 \
        python3 -m swift.cli.deploy \
        --model "$VLM_MODEL_PATH" \
        --infer_backend vllm \
        --max_new_tokens 4096 \
        --api_key retoo \
        --served_model_name "$VLM_NAME" \
        --host "$VLM_HOST" \
        --port "$VLM_PORT" \
        --vllm_gpu_memory_utilization "$VLM_GPU_MEMORY_UTILIZATION" \
        --vllm_tensor_parallel_size "$VLM_TENSOR_PARALLEL_SIZE" \
        --vllm_max_model_len "$VLM_MAX_MODEL_LEN" \
        --vllm_enforce_eager True \
        --verbose

    if wait_for_service "$service" "$timeout"; then
        log_success "$(service_name "$service") 启动成功 (PID: $(read_pid "$pid_file"), Port: $VLM_PORT)"
        return 0
    fi

    log_error "$(service_name "$service") 启动失败，请检查日志: $log_file"
    show_recent_log "$log_file"
    return 1
}

# 启动 Edge-TTS 服务。
start_tts() {
    local service="tts"
    local pid_file log_file timeout script_path

    pid_file="$(service_pid_file "$service")"
    log_file="$(service_log_file "$service")"
    timeout="$(service_timeout "$service")"
    script_path="$(service_script_path "$service")"

    if check_process "$pid_file"; then
        log_warning "$(service_name "$service") 已经在运行中"
        return 0
    fi

    cleanup_stale_pid "$pid_file"

    if [ ! -d "$TTS_WORKDIR" ]; then
        log_error "TTS 工作目录不存在: $TTS_WORKDIR"
        return 1
    fi

    if [ ! -f "$script_path" ]; then
        log_error "TTS 启动脚本不存在: $script_path"
        return 1
    fi

    log_info "正在启动 $(service_name "$service")..."
    launch_in_conda_env \
        "$TTS_WORKDIR" \
        "$AUDIO_CONDA_ENV" \
        "$log_file" \
        "$pid_file" \
        python "$TTS_SCRIPT"

    if wait_for_service "$service" "$timeout"; then
        log_success "$(service_name "$service") 启动成功 (PID: $(read_pid "$pid_file"), Port: $TTS_PORT)"
        return 0
    fi

    log_error "$(service_name "$service") 启动失败，请检查日志: $log_file"
    show_recent_log "$log_file"
    return 1
}

# 启动 SenseVoiceSmall ASR 服务。
start_asr() {
    local service="asr"
    local pid_file log_file timeout script_path

    pid_file="$(service_pid_file "$service")"
    log_file="$(service_log_file "$service")"
    timeout="$(service_timeout "$service")"
    script_path="$(service_script_path "$service")"

    if check_process "$pid_file"; then
        log_warning "$(service_name "$service") 已经在运行中"
        return 0
    fi

    cleanup_stale_pid "$pid_file"

    if [ ! -d "$ASR_WORKDIR" ]; then
        log_error "ASR 工作目录不存在: $ASR_WORKDIR"
        return 1
    fi

    if [ ! -f "$script_path" ]; then
        log_error "ASR 启动脚本不存在: $script_path"
        return 1
    fi

    if [ ! -d "$ASR_MODEL_DIR" ]; then
        log_error "ASR 模型目录不存在: $ASR_MODEL_DIR"
        return 1
    fi

    log_info "正在启动 $(service_name "$service")..."
    launch_in_conda_env \
        "$ASR_WORKDIR" \
        "$AUDIO_CONDA_ENV" \
        "$log_file" \
        "$pid_file" \
        python "$ASR_SCRIPT"

    if wait_for_service "$service" "$timeout"; then
        log_success "$(service_name "$service") 启动成功 (PID: $(read_pid "$pid_file"), Port: $ASR_PORT)"
        return 0
    fi

    log_error "$(service_name "$service") 启动失败，请检查日志: $log_file"
    show_recent_log "$log_file"
    return 1
}

# 根据服务名分发启动逻辑。
start_service() {
    local service=$1
    case "$service" in
        vl)  start_vl ;;
        tts) start_tts ;;
        asr) start_asr ;;
        *)
            log_error "未知服务: $service"
            return 1
            ;;
    esac
}

# 停止指定服务。
stop_service() {
    local service=$1
    local pid_file pid count

    pid_file="$(service_pid_file "$service")"
    cleanup_stale_pid "$pid_file"

    log_info "正在停止 $(service_name "$service")..."
    if ! check_process "$pid_file"; then
        log_warning "$(service_name "$service") 未运行"
        return 0
    fi

    pid="$(read_pid "$pid_file")"
    kill "$pid" 2>/dev/null || true

    count=0
    while is_pid_running "$pid"; do
        sleep 1
        count=$((count + 1))
        if [ "$count" -ge 10 ]; then
            log_warning "$(service_name "$service") 未在 10 秒内退出，尝试强制终止..."
            kill -9 "$pid" 2>/dev/null || true
            break
        fi
    done

    rm -f "$pid_file"
    log_success "$(service_name "$service") 已停止"
}

# ==================== 状态展示 ====================

# 显示单个服务的运行状态。
print_service_status() {
    local service=$1
    local pid_file pid port status_mark ready_text

    pid_file="$(service_pid_file "$service")"
    port="$(service_port "$service")"

    if check_process "$pid_file"; then
        pid="$(read_pid "$pid_file")"
        if service_ready "$service"; then
            ready_text="端口已就绪"
        else
            ready_text="进程存在，但端口未就绪"
        fi
        status_mark="${GREEN}✓${NC}"
        echo -e "${status_mark} $(service_name "$service"): ${GREEN}运行中${NC} (PID: ${pid}, Port: ${port}, ${ready_text})"
        return
    fi

    if [ -f "$pid_file" ]; then
        status_mark="${YELLOW}!${NC}"
        echo -e "${status_mark} $(service_name "$service"): ${YELLOW}PID 文件已过期${NC} (Port: ${port})"
        return
    fi

    status_mark="${RED}✗${NC}"
    echo -e "${status_mark} $(service_name "$service"): ${RED}已停止${NC} (Port: ${port})"
}

# 显示指定服务的状态汇总。
show_status() {
    local service
    echo ""
    echo "==================== 服务状态 ===================="
    for service in "${TARGET_SERVICES[@]}"; do
        print_service_status "$service"
    done
    echo "日志目录: $LOG_DIR"
    echo "PID 目录: $PID_DIR"
    echo "================================================="
    echo ""
}

# ==================== 参数解析 ====================

declare -a TARGET_SERVICES=()

# 判断数组里是否已经包含某个服务。
contains_service() {
    local needle=$1
    local item
    shift

    for item in "$@"; do
        if [ "$item" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

# 解析服务列表，默认选择全部。
parse_services() {
    local service
    TARGET_SERVICES=()

    if [ "$#" -eq 0 ]; then
        TARGET_SERVICES=("${SERVICES[@]}")
        return
    fi

    for service in "$@"; do
        case "$service" in
            all)
                TARGET_SERVICES=("${SERVICES[@]}")
                return
                ;;
            vl|tts|asr)
                if ! contains_service "$service" "${TARGET_SERVICES[@]}"; then
                    TARGET_SERVICES+=("$service")
                fi
                ;;
            *)
                log_error "未知服务: $service"
                usage
                exit 1
                ;;
        esac
    done
}

# ==================== 主控制逻辑 ====================

ACTION="${1:-}"
shift || true

parse_services "$@"

case "$ACTION" in
    start)
        log_info "开始启动服务: ${TARGET_SERVICES[*]}"
        for service in "${TARGET_SERVICES[@]}"; do
            start_service "$service"
        done
        show_status
        log_success "服务启动流程完成"
        ;;

    stop)
        log_info "开始停止服务: ${TARGET_SERVICES[*]}"
        for service in "${TARGET_SERVICES[@]}"; do
            stop_service "$service"
        done
        show_status
        ;;

    restart)
        log_info "开始重启服务: ${TARGET_SERVICES[*]}"
        for service in "${TARGET_SERVICES[@]}"; do
            stop_service "$service"
        done
        sleep 1
        for service in "${TARGET_SERVICES[@]}"; do
            start_service "$service"
        done
        show_status
        log_success "服务重启流程完成"
        ;;

    status)
        show_status
        ;;

    *)
        usage
        exit 1
        ;;
esac

exit 0
