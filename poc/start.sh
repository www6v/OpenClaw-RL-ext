#!/bin/bash
# =============================================================================
# OpenClaw-RL POC — 主入口脚本
# =============================================================================
# 封装 deploy.md 的 7 步流程，一键启动完整 RL 训练循环:
#   OpenClaw (124.221.28.203) → SSH 隧道 → Slime API Server (:30000)
#       → sglang Router → vLLM Adapter (:30001) → vLLM (:30010) → Qwen3-1.7B
#
# 用法:
#   bash start.sh <命令>
#
# 命令:
#   setup       — Step 0: 重建 sgl-kernel 兼容层 (ASM shim)
#   vllm        — Step 1: 启动 vLLM 推理服务 (port 30010)
#   adapter     — Step 2: 启动 vLLM Adapter (port 30001)
#   ray         — Step 3: 启动 Ray 集群
#   train       — Step 4: 启动 Slime RL 训练
#   register    — Step 5: 注册 Adapter 为 Router Worker (自动找 router 端口)
#   tunnel      — Step 6: 建立 SSH 隧道到 OpenClaw (124.221.28.203)
#   all         — 顺序执行 setup → vllm → adapter → ray → train
#   status      — 查看所有服务状态
#   test        — 端到端链路测试 (本地 + 远程)
#   stop        — 停止所有服务
# =============================================================================

set -e

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POC_ROOT="${SCRIPT_DIR}"
OPENCLAW_RL_ROOT="$(cd "${POC_ROOT}/../.." &>/dev/null && pwd)"
SLIME_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL/slime"
REPO_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL"

# ---- 配置 ----
MODEL_PATH=${HF_CKPT:-${HOME}/models/Qwen3-1.7B}
[ -f "/data/models/models/Qwen3-1.7B/config.json" ] && MODEL_PATH="/data/models/models/Qwen3-1.7B"

VLLM_PORT=${VLLM_PORT:-30010}
ADAPTER_PORT=${ADAPTER_PORT:-30001}
API_PORT=${API_PORT:-30000}
RAY_PORT=${RAY_PORT:-6380}
OPENCLAW_HOST=${OPENCLAW_HOST:-124.221.28.203}
OPENCLAW_USER=${OPENCLAW_USER:-root}

TORCH_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib"
NVIDIA_LIBS="/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"
COMPAT_SHIM="/tmp/libsgl_compat_all.so"

# ---- 颜色 ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  OpenClaw-RL POC — Qwen3-1.7B + LoRA + vLLM       ║${NC}"
    echo -e "${BLUE}║  RTX 3090 (24GB) · Slime FSDP · OpenClaw @ ${OPENCLAW_HOST} ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════╝${NC}"
    echo ""
}

ok()   { echo -e "  ${GREEN}✅${NC} $1"; }
fail() { echo -e "  ${RED}❌${NC} $1"; }
info() { echo -e "  ${YELLOW}→${NC} $1"; }

# =============================================================================
# Step 0: 重建 sgl-kernel 兼容层
# =============================================================================
cmd_setup() {
    banner
    echo -e "${BLUE}[Step 0] 重建 sgl-kernel 兼容层${NC}"
    echo ""
    if [ -f "${POC_ROOT}/scripts/rebuild_compat.sh" ]; then
        bash "${POC_ROOT}/scripts/rebuild_compat.sh"
        ok "兼容层已构建: ${COMPAT_SHIM}"
    else
        fail "找不到 rebuild_compat.sh"
        exit 1
    fi
}

# =============================================================================
# Step 1: 启动 vLLM 推理服务
# =============================================================================
cmd_vllm() {
    banner
    echo -e "${BLUE}[Step 1] 启动 vLLM 推理服务 (port ${VLLM_PORT})${NC}"
    echo ""

    if curl -s http://localhost:${VLLM_PORT}/health 2>/dev/null | grep -qi ok; then
        ok "vLLM 已在运行"
        return 0
    fi

    # 清理残留
    pkill -f "vllm.entrypoints" 2>/dev/null || true
    sleep 2

    export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"

    info "启动 vLLM (model=${MODEL_PATH}, port=${VLLM_PORT})..."
    setsid /usr/bin/python3.13 -m vllm.entrypoints.openai.api_server \
        --model "${MODEL_PATH}" \
        --port ${VLLM_PORT} --host 0.0.0.0 \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.50 \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > "${POC_ROOT}/vllm.log" 2>&1 &
    disown
    local pid=$!
    info "vLLM PID=${pid}"

    echo -n "  等待 vLLM 就绪"
    for i in $(seq 1 60); do
        if curl -s http://localhost:${VLLM_PORT}/health 2>/dev/null | grep -qi ok; then
            echo ""
            ok "vLLM 就绪 ($((i*5))s)"
            return 0
        fi
        echo -n "."
        sleep 5
    done
    echo ""
    fail "vLLM 启动超时"
    exit 1
}

# =============================================================================
# Step 2: 启动 vLLM Adapter
# =============================================================================
cmd_adapter() {
    banner
    echo -e "${BLUE}[Step 2] 启动 vLLM Adapter (port ${ADAPTER_PORT})${NC}"
    echo ""

    if curl -s http://localhost:${ADAPTER_PORT}/health 2>/dev/null | grep -q '"ok"'; then
        ok "Adapter 已在运行"
        return 0
    fi

    pkill -f "vllm_adapter.py" 2>/dev/null || true
    sleep 1

    info "启动 Adapter (vLLM=${VLLM_PORT}, adapter=${ADAPTER_PORT})..."
    setsid /usr/bin/python3.13 "${POC_ROOT}/scripts/vllm_adapter.py" \
        --vllm-url "http://localhost:${VLLM_PORT}" \
        --port ${ADAPTER_PORT} \
        > "${POC_ROOT}/adapter.log" 2>&1 &
    disown
    info "Adapter PID=$!"

    sleep 3
    if curl -s http://localhost:${ADAPTER_PORT}/health 2>/dev/null | grep -q '"ok"'; then
        ok "Adapter 就绪"
    else
        fail "Adapter 启动失败，检查 ${POC_ROOT}/adapter.log"
        exit 1
    fi
}

# =============================================================================
# Step 3: 启动 Ray 集群
# =============================================================================
cmd_ray() {
    banner
    echo -e "${BLUE}[Step 3] 启动 Ray 集群${NC}"
    echo ""

    if ray status 2>/dev/null | grep -q "Active"; then
        ok "Ray 已在运行"
        return 0
    fi

    ray stop 2>/dev/null || true
    sleep 2

    export LD_PRELOAD="${COMPAT_SHIM}"
    export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
    export RAY_AUTH_MODE=disabled

    info "启动 Ray (port=${RAY_PORT}, 1 GPU)..."
    ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 \
        --disable-usage-stats --port ${RAY_PORT} 2>&1 | tail -2
    ok "Ray 已启动"
}

# =============================================================================
# Step 4: 启动 Slime RL 训练
# =============================================================================
cmd_train() {
    banner
    echo -e "${BLUE}[Step 4] 启动 Slime RL 训练${NC}"
    echo ""

    if pgrep -f "train_async" >/dev/null 2>&1; then
        ok "训练已在运行 (PID=$(pgrep -f 'train_async' | head -1))"
        return 0
    fi

    # 环境变量
    export LD_PRELOAD="${COMPAT_SHIM}"
    export LD_LIBRARY_PATH="${TORCH_LIBS}:${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
    export PYTHONPATH="${REPO_ROOT}/openclaw-rl:${SLIME_ROOT}"
    export RAY_AUTH_MODE=disabled
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export SGLANG_API_KEY=""
    export SERVED_MODEL_NAME="qwen3-1.7b"
    export HOST="0.0.0.0"
    export PORT="${API_PORT}"
    export OPENCLAW_RECORD_ENABLED=1
    export OPENCLAW_RECORD_FILE="${POC_ROOT}/openclaw-rl/results/record.jsonl"
    export TP=1 CONTEXT_LENGTH=16384 MEM_FRACTION_STATIC=0.50
    export REASONING_PARSER="qwen3"
    export TOOL_CALL_PARSER="qwen25"
    export PRM_M=1
    export OPENCLAW_SGLANG_ROUTER_IP="127.0.0.1"
    export OPENCLAW_SGLANG_ROUTER_PORT="${ADAPTER_PORT}"
    export NO_PROXY="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8"
    export no_proxy="${NO_PROXY}"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

    mkdir -p "${POC_ROOT}/openclaw-rl/results"
    mkdir -p "${POC_ROOT}/ckpt/qwen3-1.7b-vllm-rl"

    cd "${SLIME_ROOT}"
    > "${POC_ROOT}/train.log"

    info "提交训练任务..."
    /usr/bin/python3.13 train_async.py \
       --train-backend fsdp --attn-implementation sdpa \
       --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
       --rollout-num-gpus 0 --num-gpus-per-node 1 \
       --hf-checkpoint "${MODEL_PATH}" --ref-load "${MODEL_PATH}" \
       --save "${POC_ROOT}/ckpt/qwen3-1.7b-vllm-rl" \
       --save-interval 50 --disable-rollout-global-dataset \
       --rollout-function-path openclaw_rollout.generate_rollout_openclaw \
       --num-rollout 100000000 --rollout-batch-size 4 --n-samples-per-prompt 1 \
       --rollout-max-response-len 2048 --rollout-max-context-len 16384 \
       --rollout-temperature 0.6 --reward-key score --num-steps-per-rollout 1 \
       --use-dynamic-batch-size --max-tokens-per-gpu 4096 --gradient-checkpointing \
       --advantage-estimator grpo --disable-rewards-normalization \
       --use-kl-loss --kl-loss-coef 0.0 --kl-loss-type low_var_kl \
       --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 \
       --optimizer adam --lr 1e-5 --lr-decay-style constant --weight-decay 0.1 \
       --adam-beta1 0.9 --adam-beta2 0.98 \
       --use-lora --lora-rank 32 --lora-alpha 64 \
       --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
       --rollout-external --rollout-external-engine-addrs "127.0.0.1:${ADAPTER_PORT}" \
       --rollout-num-gpus-per-engine 0 \
       --sglang-mem-fraction-static 0.0 --sglang-context-length 16384 \
       --sglang-reasoning-parser qwen3 \
       --custom-generate-function-path openclaw_api_server.generate \
       --custom-rm-path openclaw_api_server.reward_func \
       > "${POC_ROOT}/train.log" 2>&1 &
    disown
    info "训练 PID=$!"

    echo -n "  等待训练初始化"
    for i in $(seq 1 60); do
        if grep -q "fired up and ready" "${POC_ROOT}/train.log" 2>/dev/null; then
            echo ""
            ok "训练就绪"
            # 自动注册 router worker
            cmd_register
            return 0
        fi
        if grep -qi "error\|traceback\|exception" "${POC_ROOT}/train.log" 2>/dev/null | grep -v "SyntaxWarning"; then
            echo ""
            fail "训练启动失败，检查 ${POC_ROOT}/train.log"
            exit 1
        fi
        echo -n "."
        sleep 5
    done
    echo ""
    info "训练仍在初始化，用 'bash start.sh status' 查看进度"
}

# =============================================================================
# Step 5: 注册 Adapter 为 Router Worker
# =============================================================================
cmd_register() {
    banner
    echo -e "${BLUE}[Step 5] 注册 Adapter 为 Router Worker${NC}"
    echo ""

    # 从训练日志找 router 端口
    local router_port=$(grep -oP "Router launched at .*:\K\d+" "${POC_ROOT}/train.log" 2>/dev/null | tail -1)
    local router_ip=$(grep -oP "Router launched at \K[^:]+" "${POC_ROOT}/train.log" 2>/dev/null | tail -1)

    if [ -z "${router_port}" ]; then
        fail "找不到 Router 端口，训练可能还没启动完成"
        echo "  稍后手动运行: bash start.sh register"
        return 1
    fi

    info "Router: ${router_ip}:${router_port}"

    # 注册 adapter 为 worker
    curl -s -X POST "http://${router_ip}:${router_port}/workers" \
        -H "Content-Type: application/json" \
        -d "{\"url\":\"http://127.0.0.1:${ADAPTER_PORT}\",\"worker_type\":\"regular\"}" > /dev/null

    sleep 3

    # 验证
    local worker_count=$(curl -s "http://${router_ip}:${router_port}/workers" 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('total',0))" 2>/dev/null)
    if [ "${worker_count}" = "1" ]; then
        ok "Adapter 已注册为 worker"
    else
        info "Worker 注册中 (total=${worker_count:-?})，稍后重试"
    fi
}

# =============================================================================
# Step 6: 建立 SSH 隧道到 OpenClaw
# =============================================================================
cmd_tunnel() {
    banner
    echo -e "${BLUE}[Step 6] SSH 隧道到 OpenClaw (${OPENCLAW_HOST})${NC}"
    echo ""

    # 检查是否已有隧道
    if pgrep -f "ssh.*-R.*30000.*${OPENCLAW_HOST}" >/dev/null 2>&1; then
        ok "隧道已存在"
        return 0
    fi

    # 清理旧隧道
    pkill -f "ssh.*-R.*3000" 2>/dev/null || true
    sleep 1

    info "建立隧道: ${OPENCLAW_HOST}:30000,30001 → localhost:30000,30001"
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -f -N \
        -R 30000:localhost:30000 \
        -R 30001:localhost:30001 \
        ${OPENCLAW_USER}@${OPENCLAW_HOST}
    ok "隧道已建立"
}

# =============================================================================
# 一键启动所有
# =============================================================================
cmd_all() {
    banner
    echo -e "${BLUE}一键启动完整训练循环${NC}"
    echo ""
    cmd_setup
    cmd_vllm
    cmd_adapter
    cmd_ray
    cmd_train
    cmd_tunnel
    echo ""
    echo -e "${GREEN}=============================================${NC}"
    echo -e "${GREEN}  完整 RL 训练循环已启动!${NC}"
    echo -e "${GREEN}=============================================${NC}"
    echo ""
    echo "  OpenClaw (${OPENCLAW_HOST}) 现在可以通过隧道访问:"
    echo "    http://localhost:${API_PORT}/v1"
    echo ""
    echo "  监控: bash start.sh status"
    echo "  测试: bash start.sh test"
    echo "  停止: bash start.sh stop"
}

# =============================================================================
# 状态查看
# =============================================================================
cmd_status() {
    banner
    echo -e "${BLUE}=== 服务状态 ===${NC}"
    echo ""

    # vLLM
    if curl -s http://localhost:${VLLM_PORT}/health 2>/dev/null | grep -qi ok; then
        ok "vLLM (:${VLLM_PORT})"
    else
        fail "vLLM (:${VLLM_PORT})"
    fi

    # Adapter
    if curl -s http://localhost:${ADAPTER_PORT}/health 2>/dev/null | grep -q '"ok"'; then
        ok "Adapter (:${ADAPTER_PORT})"
    else
        fail "Adapter (:${ADAPTER_PORT})"
    fi

    # Ray
    if ray status 2>/dev/null | grep -q "Active"; then
        ok "Ray (:${RAY_PORT})"
    else
        fail "Ray"
    fi

    # Training
    if pgrep -f "train_async" >/dev/null 2>&1; then
        ok "Slime Training (PID=$(pgrep -f 'train_async' | head -1))"
    else
        fail "Slime Training"
    fi

    # API Server (Slime 内部的 OpenClaw API Server)
    if ss -tln 2>/dev/null | grep -q ":${API_PORT} "; then
        ok "Slime API Server (:${API_PORT})"
    else
        info "Slime API Server 未监听"
    fi

    # SSH 隧道
    if pgrep -f "ssh.*-R.*${OPENCLAW_HOST}" >/dev/null 2>&1; then
        ok "SSH 隧道 → ${OPENCLAW_HOST}"
    else
        fail "SSH 隧道"
    fi

    echo ""
    echo -e "${BLUE}=== GPU ===${NC}"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null || echo "  无 GPU 进程"
    echo ""

    echo -e "${BLUE}=== 训练进度 ===${NC}"
    if [ -f "${POC_ROOT}/train.log" ]; then
        grep "submitted sample\|train_step\|waiting for API\|Timer" "${POC_ROOT}/train.log" 2>/dev/null | \
            sed 's/\x1b\[[0-9;]*m//g' | tail -5 || echo "  日志为空"
    fi
}

# =============================================================================
# 端到端测试
# =============================================================================
cmd_test() {
    banner
    echo -e "${BLUE}端到端链路测试${NC}"
    echo ""

    # 本地测试
    echo -e "${YELLOW}本地测试 (localhost:${API_PORT})${NC}"
    local result=$(curl -s http://localhost:${API_PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -H "X-Session-Id: test-local" \
        -H "X-Turn-Type: main" \
        -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":15}' 2>/dev/null)

    if echo "$result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:60])" 2>/dev/null; then
        ok "本地链路通"
    else
        fail "本地链路: $(echo "$result" | head -c 100)"
    fi

    echo ""

    # 远程测试
    echo -e "${YELLOW}远程测试 (${OPENCLAW_USER}@${OPENCLAW_HOST})${NC}"
    local remote_result=$(ssh -o ConnectTimeout=5 ${OPENCLAW_USER}@${OPENCLAW_HOST} \
        "curl -s http://localhost:${API_PORT}/v1/chat/completions \
            -H 'Content-Type: application/json' \
            -H 'X-Session-Id: test-remote' \
            -H 'X-Turn-Type: main' \
            -d '{\"model\":\"qwen3-1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":15}'" 2>/dev/null)

    if echo "$remote_result" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['choices'][0]['message']['content'][:60])" 2>/dev/null; then
        ok "远程链路通"
    else
        fail "远程链路: $(echo "$remote_result" | head -c 100)"
        echo "  检查: bash start.sh tunnel"
    fi

    echo ""
    echo -e "${YELLOW}训练样本收集${NC}"
    grep "submitted sample" "${POC_ROOT}/train.log" 2>/dev/null | tail -3 | \
        sed 's/\x1b\[[0-9;]*m//g' || echo "  尚无样本"
}

# =============================================================================
# 停止所有服务
# =============================================================================
cmd_stop() {
    banner
    echo -e "${YELLOW}停止所有服务${NC}"
    echo ""

    pkill -f "train_async" 2>/dev/null && ok "Slime Training 已停止" || info "Training 未运行"
    pkill -f "vllm.entrypoints" 2>/dev/null && ok "vLLM 已停止" || info "vLLM 未运行"
    pkill -f "vllm_adapter" 2>/dev/null && ok "Adapter 已停止" || info "Adapter 未运行"
    ray stop 2>/dev/null && ok "Ray 已停止" || info "Ray 未运行"
    pkill -f "ssh.*-R.*3000" 2>/dev/null && ok "SSH 隧道已关闭" || info "无活跃隧道"

    sleep 2
    echo ""
    ok "全部已清理"
}

# =============================================================================
# 使用帮助
# =============================================================================
usage() {
    banner
    echo "用法: bash start.sh <命令>"
    echo ""
    echo "部署步骤:"
    echo -e "  ${GREEN}setup${NC}       Step 0: 重建 sgl-kernel 兼容层"
    echo -e "  ${GREEN}vllm${NC}        Step 1: 启动 vLLM 推理 (port ${VLLM_PORT})"
    echo -e "  ${GREEN}adapter${NC}     Step 2: 启动 vLLM Adapter (port ${ADAPTER_PORT})"
    echo -e "  ${GREEN}ray${NC}         Step 3: 启动 Ray 集群"
    echo -e "  ${GREEN}train${NC}       Step 4: 启动 Slime RL 训练"
    echo -e "  ${GREEN}register${NC}    Step 5: 注册 Adapter 为 Router Worker"
    echo -e "  ${GREEN}tunnel${NC}      Step 6: SSH 隧道到 OpenClaw"
    echo ""
    echo "快捷命令:"
    echo -e "  ${GREEN}all${NC}         一键执行 setup → vllm → adapter → ray → train → tunnel"
    echo -e "  ${GREEN}status${NC}      查看所有服务状态"
    echo -e "  ${GREEN}test${NC}        端到端链路测试"
    echo -e "  ${GREEN}stop${NC}        停止所有服务"
    echo ""
    echo "示例:"
    echo "  bash start.sh all          # 从零启动"
    echo "  bash start.sh status       # 查看状态"
    echo "  bash start.sh test         # 测试链路"
    echo "  bash start.sh stop         # 停止"
    echo ""
    echo "详细流程: 见 poc/deploy.md"
}

# =============================================================================
# 主入口
# =============================================================================
case "${1:-}" in
    setup)      cmd_setup ;;
    vllm)       cmd_vllm ;;
    adapter)    cmd_adapter ;;
    ray)        cmd_ray ;;
    train)      cmd_train ;;
    register)   cmd_register ;;
    tunnel)     cmd_tunnel ;;
    all)        cmd_all ;;
    status)     cmd_status ;;
    test)       cmd_test ;;
    stop)       cmd_stop ;;
    *)          usage ;;
esac
