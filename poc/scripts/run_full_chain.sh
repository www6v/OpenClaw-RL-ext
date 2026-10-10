#!/bin/bash
# =============================================================================
# OpenClaw-RL POC: 一键启动 vLLM + 训练
# =============================================================================
# 架构:
#   vLLM (:30000) → Adapter (:30001) → OpenClaw API Proxy (:30002)
#   OpenClaw (124.221.28.203) → :30002
#
# 用法:
#   bash OpenClaw-RL-ext/poc/scripts/run_full_chain.sh
# =============================================================================

set -e

MODEL_PATH=${HF_CKPT:-${HOME}/models/Qwen3-1.7B}
if [ ! -f "${MODEL_PATH}/config.json" ] && [ -f "/data/models/models/Qwen3-1.7B/config.json" ]; then
    MODEL_PATH="/data/models/models/Qwen3-1.7B"
fi

VLLM_PORT=${VLLM_PORT:-30000}
ADAPTER_PORT=${ADAPTER_PORT:-30001}
PROXY_PORT=${PROXY_PORT:-30002}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POC_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

TORCH_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"
export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"

echo "=============================================="
echo "  OpenClaw-RL — 一键启动 vLLM 完整链路"
echo "=============================================="
echo "  模型:       ${MODEL_PATH}"
echo "  vLLM:       :${VLLM_PORT}"
echo "  Adapter:    :${ADAPTER_PORT}"
echo "  API Proxy:  :${PROXY_PORT}"
echo "  OpenClaw:   124.221.28.203 → :${PROXY_PORT}"
echo "=============================================="

# ---- Step 1: 清理 ----
echo ""
echo "[1/4] 清理旧进程..."
pkill -f "vllm.entrypoints" 2>/dev/null || true
pkill -f "vllm_adapter" 2>/dev/null || true
pkill -f "verify_server" 2>/dev/null || true
sleep 2

# ---- Step 2: 启动 vLLM ----
echo "[2/4] 启动 vLLM 推理服务..."
nohup /usr/bin/python3.13 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_PATH}" \
    --port ${VLLM_PORT} --host 0.0.0.0 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.65 \
    > "${POC_ROOT}/vllm.log" 2>&1 &
echo "  PID: $!"

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
echo "[3/4] 启动 vLLM Adapter..."
nohup /usr/bin/python3.13 "${SCRIPT_DIR}/vllm_adapter.py" \
    --vllm-url "http://localhost:${VLLM_PORT}" \
    --port ${ADAPTER_PORT} \
    > "${POC_ROOT}/adapter.log" 2>&1 &
echo "  PID: $!"

echo -n "  等待 Adapter 就绪"
for i in $(seq 1 15); do
    if curl -s http://localhost:${ADAPTER_PORT}/health 2>/dev/null | grep -qi ok; then
        echo " ✅"
        break
    fi
    echo -n "."
    sleep 2
done

# ---- Step 4: 启动 API Proxy ----
echo "[4/4] 启动 OpenClaw API Proxy..."
nohup /usr/bin/python3.13 "${SCRIPT_DIR}/../scripts/verify_server.py" \
    > "${POC_ROOT}/proxy.log" 2>&1 &
echo "  PID: $!"

echo -n "  等待 Proxy 就绪"
for i in $(seq 1 15); do
    if curl -s http://localhost:${PROXY_PORT}/health 2>/dev/null | grep -qi ok; then
        echo " ✅"
        break
    fi
    echo -n "."
    sleep 2
done

# ---- 验证 ----
echo ""
echo "=============================================="
echo "  验证完整链路"
echo "=============================================="

# 通过 Adapter 测试推理
RESULT=$(curl -s http://localhost:${ADAPTER_PORT}/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":15}' 2>/dev/null)
if echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:40])" 2>/dev/null; then
    echo "  ✅ vLLM → Adapter → 推理正常"
else
    echo "  ❌ 推理失败"
fi

# Slime 兼容端点
curl -s http://localhost:${ADAPTER_PORT}/health_generate 2>/dev/null | grep -qi ok && echo "  ✅ /health_generate OK (Slime 兼容)"
curl -s http://localhost:${ADAPTER_PORT}/get_server_info 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  ✅ /get_server_info OK (model={d.get(\"model_path\",\"?\")[:30]})')" 2>/dev/null

echo ""
echo "=============================================="
echo "  ✅ 全链路就绪!"
echo "=============================================="
echo ""
echo "  OpenClaw (124.221.28.203) 配置:"
echo "  ─────────────────────────────────"

# 获取本机 IP
LOCAL_IP=$(hostname -I | awk '{print $1}')
PUBLIC_IP=$(curl -s ifconfig.me 2>/dev/null || curl -s ip.sb 2>/dev/null || echo "未知")

echo "  内网: http://${LOCAL_IP}:${PROXY_PORT}/v1"
echo "  公网: http://${PUBLIC_IP}:${PROXY_PORT}/v1"
echo ""
echo "  如果从公网无法访问，需要:"
echo "    1. 配置路由器端口转发: ${PROXY_PORT} → ${LOCAL_IP}:${PROXY_PORT}"
echo "    2. 或 SSH 隧道: ssh -R ${PROXY_PORT}:${LOCAL_IP}:${PROXY_PORT} root@124.221.28.203"
echo ""
echo "  在 OpenClaw (124.221.28.203) 的 openclaw.json 中:"
echo "  {"
echo "    \"models\": {"
echo "      \"providers\": {"
echo "        \"openclaw-rl\": {"
echo "          \"baseUrl\": \"http://<训练机IP>:${PROXY_PORT}/v1\","
echo "          \"apiKey\": \"\","
echo "          \"api\": \"openai-completions\","
echo "          \"models\": [{"
echo "            \"id\": \"qwen3-1.7b\","
echo "            \"name\": \"Qwen3 1.7B (RL)\","
echo "            \"reasoning\": true,"
echo "            \"input\": [\"text\"],"
echo "            \"cost\": {\"input\":0,\"output\":0,\"cacheRead\":0,\"cacheWrite\":0},"
echo "            \"contextWindow\": 8192,"
echo "            \"maxTokens\": 4096"
echo "          }]"
echo "        }"
echo "      }"
echo "    }"
echo "  }"
echo ""
echo "  查看日志:"
echo "    tail -f ${POC_ROOT}/vllm.log"
echo "    tail -f ${POC_ROOT}/adapter.log"
echo ""
echo "  停止: pkill -f 'vllm\\|vllm_adapter\\|verify_server'"
echo "=============================================="
