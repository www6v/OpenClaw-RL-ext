#!/bin/bash
# =============================================================================
# OpenClaw-RL + Slime + vLLM 完整 RL 训练
# =============================================================================
# 架构:
#   vLLM (:30000) ← Adapter (:30001) ← OpenClaw API Server (:30002) ← OpenClaw/用户
#   Slime Trainer (FSDP + LoRA) ← 从 API Server 收集训练样本
#
# 用法:
#   bash OpenClaw-RL-ext/poc/scripts/run_slime_vllm_training.sh
# =============================================================================

set -ex

# ---- 配置 ----
MODEL_PATH=${HF_CKPT:-${HOME}/models/Qwen3-1.7B}
if [ ! -f "${MODEL_PATH}/config.json" ] && [ -f "/data/models/models/Qwen3-1.7B/config.json" ]; then
    MODEL_PATH="/data/models/models/Qwen3-1.7B"
fi

VLLM_PORT=${VLLM_PORT:-30010}
ADAPTER_PORT=${ADAPTER_PORT:-30001}
PROXY_PORT=${PROXY_PORT:-30000}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POC_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
OPENCLAW_RL_ROOT="$(cd -- "${POC_ROOT}/../.." &>/dev/null && pwd)"
SLIME_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL/slime"
REPO_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL"

# CUDA 库路径
TORCH_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"

# 模型配置
source "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh"

SAVE_CKPT="${POC_ROOT}/ckpt/qwen3-1.7b-vllm-rl"
mkdir -p "${SAVE_CKPT}"
mkdir -p "${POC_ROOT}/openclaw-rl/results"

echo "=============================================="
echo "  OpenClaw-RL + Slime + vLLM RL 训练"
echo "=============================================="
echo "  模型:       ${MODEL_PATH}"
echo "  vLLM:       :${VLLM_PORT}"
echo "  Adapter:    :${ADAPTER_PORT}"
echo "  API Proxy:  :${PROXY_PORT}"
echo "  Checkpoint: ${SAVE_CKPT}"
echo "=============================================="

# ---- Step 1: 清理 ----
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "vllm_adapter" 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 2

# ---- Step 2: 启动 vLLM ----
echo "[1/4] 启动 vLLM..."
nohup /usr/bin/python3.13 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port ${VLLM_PORT} --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.50 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${POC_ROOT}/vllm.log" 2>&1 &

echo -n "  等待 vLLM 就绪"
for i in $(seq 1 60); do
    if curl -s http://localhost:${VLLM_PORT}/health 2>/dev/null | grep -qi ok; then
        echo " ✅"
        break
    fi
    echo -n "."
    sleep 5
done

# ---- Step 3: 启动 Adapter ----
echo "[2/4] 启动 vLLM Adapter..."
nohup /usr/bin/python3.13 "${SCRIPT_DIR}/vllm_adapter.py" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --port ${ADAPTER_PORT} \
    > "${POC_ROOT}/adapter.log" 2>&1 &

for i in $(seq 1 10); do
    if curl -s http://localhost:${ADAPTER_PORT}/health 2>/dev/null | grep -qi ok; then
        echo "  ✅ Adapter 就绪"
        break
    fi
    sleep 2
done

# ---- Step 4: 启动 Slime 训练 ----
echo "[3/4] 启动 Slime 训练..."

# 环境变量
export SGLANG_API_KEY=""
export SERVED_MODEL_NAME="qwen3-1.7b"
export HOST="0.0.0.0"
export PORT="${PROXY_PORT}"
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="${POC_ROOT}/openclaw-rl/results/qwen3_1.7b_vllm_record.jsonl"
export TP=1
export CONTEXT_LENGTH=16384
export MEM_FRACTION_STATIC=0.50
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="qwen25"
export PRM_M=1

# Slime 参数
CKPT_ARGS=(
   --hf-checkpoint "${MODEL_PATH}"
   --ref-load "${MODEL_PATH}"
   --save "${SAVE_CKPT}"
   --save-interval 50
   --rotary-base 1000000
)

ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   --rollout-function-path openclaw_rollout.generate_rollout_openclaw
   --num-rollout 100000000
   --rollout-batch-size 4
   --n-samples-per-prompt 1
   --rollout-max-response-len 2048
   --rollout-max-context-len ${CONTEXT_LENGTH}
   --rollout-temperature 0.6
   --reward-key score
   --num-steps-per-rollout 1
)

PERF_ARGS=(
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
   --gradient-checkpointing
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --disable-rewards-normalization
   --use-kl-loss
   --kl-loss-coef 0.0
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

LORA_ARGS=(
   --use-lora
   --lora-rank 32
   --lora-alpha 64
   --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
)

# 推理引擎: 指向 adapter (external mode)
EXTERNAL_ARGS=(
   --rollout-external
   --rollout-external-engine-addrs "127.0.0.1:${ADAPTER_PORT}"
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 0
   --sglang-mem-fraction-static 0.0
   --sglang-context-length ${CONTEXT_LENGTH}
   --sglang-reasoning-parser "${REASONING_PARSER}"
)

CUSTOM_ARGS=(
   --custom-generate-function-path openclaw_api_server.generate
   --custom-rm-path openclaw_api_server.reward_func
)

# 启动 Ray
export MASTER_ADDR="127.0.0.1"
ray start --head --node-ip-address "${MASTER_ADDR}" \
   --num-gpus 1 \
   --disable-usage-stats \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=8265

echo "[4/4] 提交 Slime 训练任务..."

# PYTHONPATH: openclaw-rl 目录 + Slime
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${REPO_ROOT}/openclaw-rl:${SLIME_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"LD_LIBRARY_PATH\": \"${LD_LIBRARY_PATH}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 "${SLIME_ROOT}/train_async.py" \
   --train-backend fsdp \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 0 \
   --num-gpus-per-node 1 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${CUSTOM_ARGS[@]} \
   ${LORA_ARGS[@]} \
   ${EXTERNAL_ARGS[@]}
