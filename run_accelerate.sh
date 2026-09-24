#!/bin/bash

# Usage:
#   ./run_accelerate.sh [--cuda "0,1"] [-p PORT | --main_process_port=PORT] script.py [script_args...]
#
# Example:
#   ./run_accelerate.sh --cuda "0,1" -p 29501 train.py --lr 1e-4
#   ./run_accelerate.sh inference.py

set -e  # 遇错退出

# 默认值
CUDA_DEVS=""
MAIN_PROCESS_PORT=""
SCRIPT=""
SCRIPT_ARGS=()

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda)
            CUDA_DEVS="$2"
            shift 2
            ;;
        --cuda=*)
            CUDA_DEVS="${1#*=}"
            shift
            ;;
        -p)
            MAIN_PROCESS_PORT="$2"
            shift 2
            ;;
        --main_process_port)
            MAIN_PROCESS_PORT="$2"
            shift 2
            ;;
        --main_process_port=*)
            MAIN_PROCESS_PORT="${1#*=}"
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            echo "Usage: $0 [--cuda \"0,1\"] [-p PORT | --main_process_port=PORT] script.py [args...]" >&2
            exit 1
            ;;
        *)
            SCRIPT="$1"
            shift
            SCRIPT_ARGS=("$@")
            break
            ;;
    esac
done

# 检查是否提供了脚本名
if [[ -z "$SCRIPT" ]]; then
    echo "Error: No Python script provided." >&2
    echo "Usage: $0 [--cuda \"0,1\"] [-p PORT | --main_process_port=PORT] script.py [args...]" >&2
    exit 1
fi

# 设置 workspaceFolder
WORKSPACE_FOLDER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${WORKSPACE_FOLDER}/configs/accelerate.yaml"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE" >&2
    exit 1
fi

# 构建 accelerate 命令
CMD=("accelerate" "launch" "--config_file" "$CONFIG_FILE")

# 如果用户指定了端口，添加 --main_process_port
if [[ -n "$MAIN_PROCESS_PORT" ]]; then
    CMD+=("--main_process_port" "$MAIN_PROCESS_PORT")
    echo "Using main_process_port: $MAIN_PROCESS_PORT"
fi

# 如果指定了 CUDA_VISIBLE_DEVICES，则设置环境变量
if [[ -n "$CUDA_DEVS" ]]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVS"
    echo "Set CUDA_VISIBLE_DEVICES=$CUDA_DEVS"
fi

# 添加脚本和其参数
CMD+=("$SCRIPT" "${SCRIPT_ARGS[@]}")

echo "Running command:"
echo "${CMD[@]}"
echo "----------------------------------------"

# 执行
exec "${CMD[@]}"