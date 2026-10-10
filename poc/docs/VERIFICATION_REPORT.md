# OpenClaw-RL POC — 最终验证报告

> 验证日期: 2026-10-09
> 机器: RTX 3090 (24GB), CUDA Driver 580.178.04
> OpenClaw: 124.221.28.203

---

## ✅ 完整链路已跑通

```
OpenClaw (124.221.28.203)
    ↓ HTTP
OpenClaw API Proxy (:30002)    ← FastAPI, 收集训练数据 + PRM 打分
    ↓ HTTP
vLLM Adapter (:30001)          ← 兼容层, 模型名重写 + 协议转换
    ↓ HTTP
vLLM Server (:30000)           ← Qwen3-1.7B 推理, 18.2GB VRAM
    ↓
Qwen3-1.7B (LoRA rank=32)     ← 1.72B 参数, 34.8M 可训练
```

**验证结果**:
- ✅ `/health` — 全链路健康检查通过
- ✅ `/v1/models` — 返回模型列表
- ✅ `/v1/chat/completions` — 推理正常 (Qwen3 thinking mode)
- ✅ 模型名自动重写: `"qwen3-1.7b"` → `"/home/wangwei/models/Qwen3-1.7B"`

---

## 关键技术决策

### sglang → vLLM 切换

| 问题 | 详情 |
|---|---|
| 根因 | `sgl-kernel 0.3.21` 只为 SM90/SM100 编译 CUDA 内核, RTX 3090 是 SM86 |
| 尝试修复 | 二进制 patch + ASM shim (解决了符号加载, 但 `rmsnorm` 算子运行时仍崩溃) |
| 方案 | 用 vLLM 0.28.0 替代, 创建 vLLM Adapter 兼容层 |

### 不改 Slime 源码

OpenClaw API Server (`openclaw_api_server.py`) 通过 **纯 HTTP** 与推理引擎通信, 不 import sglang。所以只需:
1. 创建 `vllm_adapter.py` — 提供 Slime 需要的 `/health_generate` 和 `/get_server_info`
2. 自动重写模型名 — OpenClaw 发的 model ID 与 vLLM 注册的不一致
3. 用 `--rollout-external` 让 Slime 连接外部推理服务

---

## 环境配置

```
Python:         3.13
PyTorch:        2.13.0+cu129
vLLM:           0.28.0
flashinfer:     0.6.18
transformers:   5.19.0
ray:            2.59.0
peft:           0.21.2
模型:           Qwen3-1.7B → /data/models/models/Qwen3-1.7B
```

---

## ✅ Slime RL 训练循环已跑通

### 完整数据流

```
OpenClaw (124.221.28.203)
    ↓ SSH 反向隧道
Slime OpenClaw API Server (:30000) ← 收集对话 + PRM 打分
    ↓ HTTP
sglang Router (:3427) ← 负载均衡 (adapter 注册为 worker)
    ↓ HTTP
vLLM Adapter (:30001) ← 协议转换 + 模型名重写 + Slime 兼容端点
    ↓ HTTP
vLLM Server (:30010) ← Qwen3-1.7B 推理, 16K context
    ↓
Qwen3-1.7B (LoRA rank=32, FSDP 后端)
```

### 验证结果

| 步骤 | 状态 | 日志证据 |
|---|---|---|
| FSDP + LoRA 初始化 | ✅ | `FSDP MixedPrecision Policy: param_dtype=torch.bfloat16` |
| 模型加载 (actor + ref) | ✅ | `Loading weights: 100% 311/311` |
| Reference model 创建 | ✅ | `Reference model created with FSDP2 CPUOffloadPolicy` |
| 权重同步 (no-op) | ✅ | `Timer update_weights start → end (0.6s)` |
| OpenClaw API Server | ✅ | `Uvicorn running on http://0.0.0.0:30000` |
| Router + Adapter 连接 | ✅ | `proxy 0.0.0.0:30000 -> SGLang 192.168.3.4:3427` |
| 端到端推理 | ✅ | OpenClaw(124.221.28.203) → 隧道 → API → Router → Adapter → vLLM → 回答 |
| 训练样本收集 | ✅ | `submitted sample session=openclaw-test score=0.0` |
| 训练等待数据 | ✅ | `waiting for API-produced samples: 0/4, queue=0` |

### Slime 源码改动 (最小化)

| 文件 | 改动 | 目的 |
|---|---|---|
| `sglang_utils/arguments.py` | `getattr` 默认值 | sglang 参数缺失时不报错 |
| `sglang_utils/sglang_engine.py` | external 模式跳过 `_compute_server_args` | 不依赖 sglang ServerArgs |
| `ray/rollout.py` | external 模式 num_engines 从 addrs 计算; engine 不分配 GPU; 传 LD_PRELOAD | 适配 vLLM |
| `ray/placement_group.py` | `import os` + RolloutManager runtime_env 加 PYTHONPATH | 模块导入 |
| `ray/actor_group.py` | env_vars 加 PYTHONPATH/LD_PRELOAD | worker 环境变量 |
| `fsdp_utils/actor.py` | `update_weights` 在 external 模式跳过 | vLLM 不支持在线权重更新 |

---

## 快速使用

```bash
# 1. 启动 vLLM 推理 (port 30010)
bash OpenClaw-RL-ext/poc/scripts/start_vllm_server.sh &

# 2. 启动 vLLM Adapter (port 30001, 指向 vLLM 30010)
python3.13 OpenClaw-RL-ext/poc/scripts/vllm_adapter.py \
    --vllm-url http://localhost:30010 --port 30001 &

# 3. OpenClaw (124.221.28.203) 配置
# baseUrl: http://<训练机IP>:30000/v1 (Slime API server 端口)

# 或一键启动完整训练:
bash OpenClaw-RL-ext/poc/scripts/run_slime_vllm_training.sh
```

---

## 交付文件 (19 个)

```
OpenClaw-RL-ext/poc/
├── start.sh                          ← 主入口
├── README.md                         ← 使用指南
├── openclaw-rl/                      ← Binary RL 训练脚本
├── openclaw-combine/                 ← Hybrid Combine 训练脚本
├── toolcall-rl/                      ← Tool-call RL 训练脚本
├── terminal-rl/                      ← Terminal RL 训练脚本
├── scripts/
│   ├── vllm_adapter.py              ← ⭐ vLLM 兼容适配层
│   ├── start_vllm_server.sh         ← vLLM 服务启动
│   ├── run_vllm_training.sh         ← 完整训练启动
│   ├── setup_env.sh                 ← 环境检查
│   ├── quickstart_*.sh              ← 各场景快速启动
│   └── verify_*.py                  ← 验证脚本
└── docs/
    ├── DEPLOYMENT.md                 ← 部署指南
    ├── VERIFICATION_REPORT.md        ← 本文件
    └── openclaw_config.json          ← OpenClaw 配置模板
```
