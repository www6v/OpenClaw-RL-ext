#!/bin/bash
# =============================================================================
# OpenClaw-RL POC: Qwen3-1.7B vLLM 推理服务
# =============================================================================
# 用 vLLM 替代 sglang 作为推理后端 (vLLM 支持 RTX 3090 SM86)
# 提供 OpenAI 兼容 API: http://0.0.0.0:30000/v1
#
# 用法:
#   bash OpenClaw-RL-ext/poc/scripts/start_vllm_server.sh
# =============================================================================

set -e

MODEL_PATH=${HF_CKPT:-${HOME}/models/Qwen3-1.7B}
if [ ! -f "${MODEL_PATH}/config.json" ] && [ -f "/data/models/models/Qwen3-1.7B/config.json" ]; then
    MODEL_PATH="/data/models/models/Qwen3-1.7B"
fi

PORT=${PORT:-30000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.75}

# CUDA 库路径
TORCH_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"

echo "=============================================="
echo "  OpenClaw-RL POC — vLLM 推理服务"
echo "=============================================="
echo "  模型:        ${MODEL_PATH}"
echo "  端口:        ${PORT}"
echo "  最大长度:    ${MAX_MODEL_LEN}"
echo "  GPU 利用率:  ${GPU_MEM_UTIL}"
echo "  API:         http://0.0.0.0:${PORT}/v1"
echo "=============================================="
echo ""

# 检查模型
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "❌ 模型不存在: ${MODEL_PATH}"
    echo "   运行: bash OpenClaw-RL-ext/poc/scripts/setup_env.sh"
    exit 1
fi

# 启动 vLLM
exec /usr/bin/python3.13 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port "${PORT}" \
    --host 0.0.0.0 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
