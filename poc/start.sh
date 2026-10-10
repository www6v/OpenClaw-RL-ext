#!/bin/bash
# =============================================================================
# OpenClaw-RL POC — 主入口脚本
# =============================================================================
# 用法:
#   bash OpenClaw-RL-ext/poc/start.sh [命令]
#
# 命令:
#   setup       — 环境检查 + 下载模型
#   rl          — 启动 Binary RL (对接 OpenClaw)
#   combine     — 启动 Hybrid Combine (最优效果)
#   toolcall    — 启动 Tool-call RL (代码调用)
#   terminal    — 启动 Terminal RL (终端命令)
#   status      — 查看训练状态
#   stop        — 停止所有训练
#   eval        — 评估当前 checkpoint
#
# 示例:
#   bash start.sh setup
#   bash start.sh rl
#   bash start.sh combine
# =============================================================================

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POC_ROOT="${SCRIPT_DIR}"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  OpenClaw-RL POC — Qwen3-1.7B 单卡训练        ║${NC}"
    echo -e "${BLUE}║  RTX 3090 (24GB) · FSDP + LoRA                ║${NC}"
    echo -e "${BLUE}║  OpenClaw @ 124.221.28.203                     ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════╝${NC}"
    echo ""
}

usage() {
    banner
    echo "用法: bash start.sh <命令>"
    echo ""
    echo "命令:"
    echo -e "  ${GREEN}setup${NC}       环境检查 + 下载 Qwen3-1.7B 模型"
    echo -e "  ${GREEN}serve${NC}        启动 vLLM 推理服务 (OpenAI 兼容 API)"
    echo -e "  ${GREEN}rl${NC}          启动 Binary RL 训练 (推荐第一步)"
    echo -e "  ${GREEN}combine${NC}     启动 Hybrid Combine 训练 (效果最好)"
    echo -e "  ${GREEN}toolcall${NC}    启动 Tool-call RL 训练"
    echo -e "  ${GREEN}terminal${NC}    启动 Terminal RL 训练 (需 Docker Worker)"
    echo -e "  ${GREEN}status${NC}      查看训练状态和 GPU 使用情况"
    echo -e "  ${GREEN}stop${NC}        停止所有训练进程"
    echo ""
    echo "示例:"
    echo "  bash start.sh setup      # 首次运行，准备环境"
    echo "  bash start.sh serve      # 启动 vLLM 推理服务"
    echo "  bash start.sh rl         # 启动 Binary RL"
    echo "  bash start.sh combine    # 启动 Hybrid Combine"
    echo "  bash start.sh status     # 查看 GPU 和训练状态"
    echo "  bash start.sh stop       # 停止训练"
}

cmd_setup() {
    banner
    echo -e "${GREEN}[Step 1/4] 检查 GPU...${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""

    echo -e "${GREEN}[Step 2/4] 检查 Python 环境...${NC}"
    python3 --version
    pip show torch 2>/dev/null | grep -E "^(Name|Version):" || echo "  ⚠️ torch 未安装"
    pip show ray 2>/dev/null | grep -E "^(Name|Version):" || echo "  ⚠️ ray 未安装"
    echo ""

    echo -e "${GREEN}[Step 3/4] 检查 Slime 框架...${NC}"
    OPENCLAW_RL_ROOT="$(cd "${POC_ROOT}/../.." &>/dev/null && pwd)"
    SLIME_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL/slime"
    if [ -f "${SLIME_ROOT}/train_async.py" ]; then
        echo -e "  ✅ Slime: ${SLIME_ROOT}"
    else
        echo -e "  ❌ Slime 不存在: ${SLIME_ROOT}"
        exit 1
    fi
    if [ -f "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh" ]; then
        echo -e "  ✅ Qwen3-1.7B 配置存在"
    else
        echo -e "  ❌ Qwen3-1.7B 配置不存在"
        exit 1
    fi
    echo ""

    echo -e "${GREEN}[Step 4/4] 检查/下载模型...${NC}"
    MODEL_DIR="${HOME}/models/Qwen3-1.7B"
    if [ -d "${MODEL_DIR}" ] && [ -f "${MODEL_DIR}/config.json" ]; then
        echo -e "  ✅ 模型已存在: ${MODEL_DIR}"
        ls -lh "${MODEL_DIR}/"*.safetensors 2>/dev/null | head -5
    else
        echo -e "  📥 下载 Qwen3-1.7B (~3.5GB)..."
        mkdir -p "${HOME}/models"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='Qwen/Qwen3-1.7B', local_dir='${MODEL_DIR}', local_dir_use_symlinks=False)
print('✅ 下载完成!')
" 2>&1
    fi
    echo ""

    # 创建输出目录
    mkdir -p "${POC_ROOT}/ckpt"
    mkdir -p "${POC_ROOT}/data"

    banner
    echo -e "${GREEN}✅ 环境准备完成!${NC}"
    echo ""
    echo "下一步:"
    echo -e "  ${YELLOW}bash start.sh rl${NC}       启动 Binary RL 训练"
    echo -e "  ${YELLOW}bash start.sh combine${NC}  启动 Hybrid Combine 训练"
}

cmd_rl() {
    banner
    echo -e "${BLUE}启动 Binary RL (GRPO) 训练...${NC}"
    echo -e "  模型: Qwen3-1.7B + LoRA"
    echo -e "  后端: FSDP (单卡)"
    echo -e "  服务: http://0.0.0.0:30000/v1"
    echo ""
    cd "$(cd "${POC_ROOT}/../.." && pwd)/OpenClaw-RL/slime"
    bash "${POC_ROOT}/openclaw-rl/run_qwen3_1.7b_openclaw_rl_1gpu.sh"
}

cmd_combine() {
    banner
    echo -e "${BLUE}启动 Hybrid Combine (GRPO + OPD) 训练...${NC}"
    echo -e "  模型: Qwen3-1.7B + LoRA"
    echo -e "  后端: FSDP (单卡)"
    echo -e "  服务: http://0.0.0.0:30000/v1"
    echo ""
    cd "$(cd "${POC_ROOT}/../.." && pwd)/OpenClaw-RL/slime"
    bash "${POC_ROOT}/openclaw-combine/run_qwen3_1.7b_openclaw_combine_1gpu.sh"
}

cmd_toolcall() {
    banner
    echo -e "${BLUE}启动 Tool-call RL 训练...${NC}"
    echo -e "  模型: Qwen3-1.7B + LoRA"
    echo -e "  任务: 代码解释器 + 数学推理"
    echo ""
    bash "${POC_ROOT}/scripts/quickstart_toolcall.sh"
}

cmd_terminal() {
    banner
    echo -e "${BLUE}启动 Terminal RL 训练...${NC}"
    echo -e "  模型: Qwen3-1.7B + LoRA"
    echo -e "  任务: 终端命令生成"
    echo ""
    if [ -z "${WORKER_URLS}" ]; then
        echo -e "${RED}错误: 需要设置 WORKER_URLS${NC}"
        echo "  export WORKER_URLS=\"http://worker-ip:18081\""
        exit 1
    fi
    bash "${POC_ROOT}/scripts/quickstart_terminal.sh"
}

cmd_status() {
    banner
    echo -e "${BLUE}=== GPU 状态 ===${NC}"
    nvidia-smi
    echo ""

    echo -e "${BLUE}=== Ray 集群 ===${NC}"
    ray status 2>/dev/null || echo "  Ray 未运行"
    echo ""

    echo -e "${BLUE}=== 检查点 ===${NC}"
    for d in "${POC_ROOT}/ckpt"/*/; do
        if [ -d "$d" ]; then
            name=$(basename "$d")
            count=$(find "$d" -name "checkpoint-*" -type d 2>/dev/null | wc -l)
            latest=$(ls -td "$d"checkpoint-* 2>/dev/null | head -1)
            echo "  ${name}: ${count} 个检查点"
            [ -n "${latest}" ] && echo "    最新: $(basename ${latest})"
        fi
    done
    echo ""

    echo -e "${BLUE}=== 训练记录 ===${NC}"
    for f in "${POC_ROOT}"/*/results/*.jsonl; do
        if [ -f "$f" ]; then
            lines=$(wc -l < "$f")
            echo "  $(basename $(dirname $(dirname $f)))/$(basename $f): ${lines} 条记录"
        fi
    done
}

cmd_stop() {
    banner
    echo -e "${YELLOW}停止所有训练进程...${NC}"
    pkill -9 sglang 2>/dev/null && echo "  ✅ SGLang 已停止" || echo "  SGLang 未运行"
    ray stop --force 2>/dev/null && echo "  ✅ Ray 已停止" || echo "  Ray 未运行"
    pkill -f "train_async" 2>/dev/null && echo "  ✅ 训练进程已停止" || echo "  训练进程未运行"
    sleep 2
    echo ""
    echo -e "${GREEN}✅ 已清理${NC}"
}

cmd_serve() {
    banner
    echo -e "${BLUE}启动 vLLM 推理服务...${NC}"
    echo -e "  模型: Qwen3-1.7B"
    echo -e "  端口: ${PORT:-30000}"
    echo -e "  API:  http://0.0.0.0:${PORT:-30000}/v1"
    echo ""
    bash "${POC_ROOT}/scripts/start_vllm_server.sh"
}

# =============================================================================
# 主入口
# =============================================================================
case "${1:-}" in
    setup)      cmd_setup ;;
    serve)      cmd_serve ;;
    rl)         cmd_rl ;;
    combine)    cmd_combine ;;
    toolcall)   cmd_toolcall ;;
    terminal)   cmd_terminal ;;
    status)     cmd_status ;;
    stop)       cmd_stop ;;
    *)          usage ;;
esac
