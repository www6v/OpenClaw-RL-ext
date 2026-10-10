# OpenClaw-RL POC — 部署指南

> 在 **1x RTX 3090 (24GB)** 上运行 OpenClaw-RL，使用 **Qwen3-1.7B + LoRA + vLLM + Slime**
> OpenClaw 部署在 `124.221.28.203`

---

## 从零启动（机器重启后）

### 架构概览

```
OpenClaw (124.221.28.203)
    ↓ SSH 反向隧道
Slime OpenClaw API Server (:30000)  ← 收集对话 + PRM 打分
    ↓ HTTP
sglang Router (:????)                ← 负载均衡 (自动分配端口)
    ↓ HTTP
vLLM Adapter (:30001)                ← 协议转换 + 模型名重写 + Slime 兼容端点
    ↓ HTTP
vLLM Server (:30010)                 ← Qwen3-1.7B 推理, 16K context, ~13GB VRAM
    ↓
Slime RL Trainer (FSDP + LoRA)       ← 从 API Server 取样本, 更新权重, ~4.3GB VRAM
```

**GPU 分配**: vLLM 13GB + FSDP 4.3GB = 17.3GB / 24GB

### Step 0: 重建 sgl-kernel 兼容层

Slime 的 SGLangEngine 会 import sglang/sgl_kernel。RTX 3090 是 SM86，但 sgl-kernel 只为 SM90/SM100 编译了 CUDA 内核。需要通过二进制 patch + ASM shim 桥接。

```bash
bash OpenClaw-RL-ext/poc/scripts/rebuild_compat.sh
```

这个脚本做 3 件事:
1. 二进制 patch `sgl_kernel/sm90/common_ops.abi3.so` (修复 `c10_cuda_check` 符号)
2. Patch `sgl_kernel/load_utils.py` (让 SM86 使用 SM90 内核)
3. 构建 `/tmp/libsgl_compat_all.so` (6 个 ASM 符号桥接)

### Step 1: 启动 vLLM 推理服务

```bash
export LD_LIBRARY_PATH="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib:/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"

python3.13 -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen3-1.7B \
    --port 30010 --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.50 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes &
```

等待 `Application startup complete` (约 3 分钟)。

验证: `curl http://localhost:30010/health` → `{"status":"ok"}`

### Step 2: 启动 vLLM Adapter

```bash
python3.13 OpenClaw-RL-ext/poc/scripts/vllm_adapter.py \
    --vllm-url http://localhost:30010 \
    --port 30001 &
```

Adapter 提供 Slime 需要但 vLLM 没有的端点:

| 端点 | 作用 |
|---|---|
| `/health_generate` | Slime 健康检查 (转发到 vLLM `/health`) |
| `/get_server_info` | 返回模型配置 (model_path, tp_size 等) |
| `/flush_cache` | KV cache 清空 (返回 ok) |
| `/init_weights_update_group` | 权重同步初始化 (no-op, vLLM 不支持在线更新) |
| `/update_weights_from_distributed` | 权重推送 (no-op) |
| `/release_weights_update_group` | 释放同步组 (no-op) |
| `/v1/chat/completions` | 推理请求 (自动重写模型名) |

验证: `curl http://localhost:30001/health` → `{"status":"ok"}`

### Step 3: 启动 Ray 集群

```bash
export LD_PRELOAD="/tmp/libsgl_compat_all.so"
export LD_LIBRARY_PATH="/home/wangwei/.local/lib/python3.13/site-packages/torch/lib:/home/wangwei/.local/lib/python3.13/site-packages/nvidia/cublas/lib"
export RAY_AUTH_MODE=disabled

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 \
    --disable-usage-stats --port 6380
```

注意: `LD_PRELOAD` 让所有 Ray worker 加载 ASM compat shim，解决 sgl_kernel ABI 不兼容。

### Step 4: 启动 Slime RL 训练

```bash
export PYTHONPATH="/home/wangwei/wei/workspacePy/agentic-rl/OpenClaw-RL/openclaw-rl:/home/wangwei/wei/workspacePy/agentic-rl/OpenClaw-RL/slime"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export SGLANG_API_KEY=""
export SERVED_MODEL_NAME="qwen3-1.7b"
export HOST="0.0.0.0"
export PORT="30000"
export OPENCLAW_RECORD_ENABLED=1
export OPENCLAW_RECORD_FILE="OpenClaw-RL-ext/poc/openclaw-rl/results/record.jsonl"
export TP=1 CONTEXT_LENGTH=16384 MEM_FRACTION_STATIC=0.50
export REASONING_PARSER="qwen3"
export TOOL_CALL_PARSER="qwen25"
export PRM_M=1
export OPENCLAW_SGLANG_ROUTER_IP="127.0.0.1"
export OPENCLAW_SGLANG_ROUTER_PORT="30001"

MODEL_PATH=~/models/Qwen3-1.7B
mkdir -p OpenClaw-RL-ext/poc/openclaw-rl/results
mkdir -p OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-vllm-rl

cd OpenClaw-RL/slime

python3.13 train_async.py \
   --train-backend fsdp --attn-implementation sdpa \
   --actor-num-nodes 1 --actor-num-gpus-per-node 1 \
   --rollout-num-gpus 0 --num-gpus-per-node 1 \
   --hf-checkpoint ${MODEL_PATH} --ref-load ${MODEL_PATH} \
   --save ../../OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-vllm-rl \
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
   --rollout-external --rollout-external-engine-addrs "127.0.0.1:30001" \
   --rollout-num-gpus-per-engine 0 \
   --sglang-mem-fraction-static 0.0 --sglang-context-length 16384 \
   --sglang-reasoning-parser qwen3 \
   --custom-generate-function-path openclaw_api_server.generate \
   --custom-rm-path openclaw_api_server.reward_func
```

等日志出现:
```
✅ Uvicorn running on http://0.0.0.0:30000
✅ [OpenClaw] your model is fired up and ready to roll
✅ proxy 0.0.0.0:30000 -> SGLang 192.168.3.4:<PORT>
```

记下日志里 `Router launched at` 的端口号 (如 `192.168.3.4:4178`)。

### Step 5: 注册 Adapter 为 Router Worker

```bash
# 替换 <ROUTER_PORT> 为上一步日志里的端口
curl -X POST "http://192.168.3.4:<ROUTER_PORT>/workers" \
    -H "Content-Type: application/json" \
    -d '{"url":"http://127.0.0.1:30001","worker_type":"regular"}'
```

等几秒后验证:
```bash
curl -s "http://192.168.3.4:<ROUTER_PORT>/workers" | python3 -m json.tool
# 应看到 is_healthy: true
```

### Step 6: 建立 SSH 隧道到 OpenClaw

```bash
ssh -f -N -o ServerAliveInterval=30 \
    -R 30000:localhost:30000 \
    -R 30001:localhost:30001 \
    root@124.221.28.203
```

### Step 7: OpenClaw 已配置 (一次性)

`124.221.28.203` 上的 `/root/.openclaw/openclaw.json` 已添加:

```json
{
  "models": {
    "providers": {
      "openclaw-rl": {
        "baseUrl": "http://localhost:30000/v1",
        "apiKey": "",
        "api": "openai-completions",
        "models": [{
          "id": "qwen3-1.7b",
          "name": "Qwen3 1.7B (RL)",
          "reasoning": true,
          "contextWindow": 16384,
          "maxTokens": 4096
        }]
      }
    }
  }
}
```

main agent 的 `model.primary` 已设为 `openclaw-rl/qwen3-1.7b`。

### 验证完整链路

```bash
# 从 OpenClaw 服务器发起请求
ssh root@124.221.28.203 'curl -s http://localhost:30000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -H "X-Session-Id: test" \
    -H "X-Turn-Type: main" \
    -d "{\"model\":\"qwen3-1.7b\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":20}"'
```

应看到 Qwen3 的推理回答。

训练日志中应出现:
```
[OpenClaw] submitted sample session=test score=0.0
```

---

## 目录结构

```
OpenClaw-RL-ext/
├── OpenClaw-RL-Analysis.md              ← 项目深度分析 (1100+ 行)
├── troubleshooting/
│   └── README.md                        ← 踩坑记录 (12 个问题)
└── poc/
    ├── start.sh                         ← 主入口 (setup/serve/rl/...)
    │
    ├── scripts/
    │   ├── rebuild_compat.sh            ← ⭐ Step 0: 重建兼容层
    │   ├── vllm_adapter.py              ← ⭐ vLLM ↔ Slime 兼容层
    │   ├── start_vllm_server.sh         ← Step 1: 启动 vLLM
    │   ├── run_slime_vllm_training.sh   ← Step 4: 一键启动训练
    │   ├── setup_env.sh                 ← 环境检查 + 模型下载
    │   └── run_full_chain.sh            ← 一键启动全链路
    │
    └── docs/
        ├── VERIFICATION_REPORT.md       ← 验证报告 (含架构图)
        └── openclaw_config.json         ← OpenClaw 配置模板
```

---

## Slime 源码改动 (6 个文件)

所有改动仅为让 `--rollout-external` 模式对接 vLLM Adapter:

| 文件 | 改动 |
|---|---|
| `slime/backends/sglang_utils/arguments.py` | `getattr` 默认值，避免 sglang 参数缺失时报错 |
| `slime/backends/sglang_utils/sglang_engine.py` | external 模式跳过 `_compute_server_args` |
| `slime/ray/rollout.py` | external 模式: num_engines 从 addrs 计算; engine 不分配 GPU; env_vars 传 LD_PRELOAD |
| `slime/ray/placement_group.py` | `import os`; RolloutManager runtime_env 传 PYTHONPATH + 清代理变量 |
| `slime/ray/actor_group.py` | env_vars 传 PYTHONPATH/LD_PRELOAD + 清代理变量 |
| `slime/backends/fsdp_utils/actor.py` | `update_weights()` 在 external 模式直接 return (vLLM 不支持在线权重推送) |

---

## 为什么不直接用 sglang

| 问题 | 原因 |
|---|---|
| sgl-kernel 不支持 RTX 3090 | 只为 SM90/SM100 编译 CUDA 内核 |
| 二进制 patch 能加载但运行崩溃 | `sgl_kernel::rmsnorm` 算子在 SM86 上 CUDA 后端不可用 |
| ABI 不兼容 | PyTorch 2.13 与 sgl-kernel 0.3.21 的 C++ 符号签名不匹配 |

**解决方案**: vLLM 0.28.0 原生支持 SM86 (RTX 3090)，通过 `vllm_adapter.py` 兼容层无缝接入 Slime。

---

## 常用操作

### 停止所有服务

```bash
pkill -f "vllm.entrypoints" 2>/dev/null
pkill -f "vllm_adapter" 2>/dev/null
pkill -f "train_async" 2>/dev/null
ray stop 2>/dev/null
pkill -f "ssh.*-R.*3000" 2>/dev/null
```

### 查看训练状态

```bash
# 训练日志
tail -f /tmp/slime_train.log

# 收集的样本
grep "submitted sample" /tmp/slime_train.log | tail -5

# GPU 使用
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# Checkpoint
ls OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-vllm-rl/
```

### 导出训练好的模型

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("~/models/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-vllm-rl/checkpoint-xxx")
model = model.merge_and_unload()
model.save_pretrained("~/models/Qwen3-1.7B-rl-merged")
```

---

## 故障排查

| 问题 | 解决 |
|---|---|
| `ModuleNotFoundError: No module named 'openclaw_rollout'` | PYTHONPATH 没传给 Ray worker → 检查 placement_group.py 的 runtime_env |
| `ZeroDivisionError: integer division or modulo by zero` | `--rollout-num-gpus 0` 时 num_engines 计算错误 → 已修复 |
| `AttributeError: 'sglang_data_parallel_size'` | sglang 参数缺失 → 已用 getattr 加默认值 |
| `Unknown scheme for proxy URL URL('socks://...')` | SOCKS 代理干扰 localhost → 设置 `NO_PROXY` 或 `trust_env=False` |
| Adapter `/flush_cache` 返回 404 | Slime `_wait_server_healthy` 死循环 → 已添加该端点 |
| `TypeError: _new_process_group_helper() got unexpected keyword` | PyTorch 2.13 与 Slime 权重同步不兼容 → external 模式跳过 |
| vLLM 启动报 `Engine core initialization failed` | 旧 vLLM 进程残留 → `kill -9` 所有 vllm 相关 PID |
| OpenClaw `upstream provider timeout` | API 转发到 sglang router 但 router 没注册 worker → Step 5 |
| OpenClaw `Context overflow` | system prompt 太大 → contextWindow 设到 16384+，tools 设为 minimal |

详细踩坑记录: `OpenClaw-RL-ext/troubleshooting/README.md`

---

*最后更新: 2026-10-10 — 完整 RL 训练循环跑通，从零启动流程文档化*
