# OpenClaw-RL-ext

[OpenClaw-RL](https://arxiv.org/abs/2603.10165) 的扩展仓库 — 把原本需要 8×H100 才能跑的全异步 RL Agent 训练框架，压到 **单张 RTX 3090 (24GB) + Qwen3-1.7B + LoRA** 上跑通。

> 上游: 通过日常对话训练个性化 AI Agent 的全异步强化学习框架，支持 Terminal / GUI / SWE / Tool-call 等通用 Agent 场景。

本仓库做三件事:

1. **单卡 POC** — 为 Track 1 (个性化) 和 Track 2 (通用 Agent) 的 4 个训练场景各提供一套能在单张消费级 GPU 上启动的脚本;
2. **vLLM 兼容层** — sgl-kernel 只为 SM90/SM100 编译内核，RTX 3090 (SM86) 跑不了 sglang，所以用 vLLM 0.28 替代，并用一个 FastAPI adapter 补齐 Slime 期望的 `/health_generate`、`/get_server_info` 等端点;
3. **踩坑记录 + 深度分析** — 把部署过程中遇到的 Python 版本错位、依赖地狱、ABI 不兼容等问题一一记录下来，并附上一份对上游项目架构的完整分析。

部署流程见 [`poc/deploy.md`](poc/deploy.md)。

---

## 仓库结构

```
OpenClaw-RL-ext/
├── README.md                     ← 本文件
├── OpenClaw-RL-Analysis.md       ← 上游 OpenClaw-RL 架构深度分析 (中文)
│
├── poc/                          ← 单卡 POC (1× RTX 3090, Qwen3-1.7B + LoRA)
│   ├── deploy.md                 ← 从零启动的完整部署流程
│   ├── start.sh                  ← 主入口 (setup/serve/rl/...)
│   │
│   ├── scripts/
│   │   ├── rebuild_compat.sh     ← 重建 sgl-kernel 兼容层 (ASM shim)
│   │   ├── vllm_adapter.py       ← Slime ↔ vLLM 兼容层
│   │   ├── start_vllm_server.sh  ← vLLM 启动
│   │   ├── run_slime_vllm_training.sh ← Slime 训练启动
│   │   ├── run_full_chain.sh     ← 一键全链路
│   │   └── setup_env.sh          ← 环境检查 + 模型下载
│   │
│   └── docs/
│       ├── openclaw_config.json  ← OpenClaw provider 配置模板
│       └── VERIFICATION_REPORT.md
│
└── troubleshooting/
    └── README.md                 ← 踩坑记录
```

---

## vLLM Adapter — 为什么不用 sglang

上游 OpenClaw-RL 假设推理后端是 sglang。但 `sgl-kernel` 只为 SM90 (H100) 和 SM100 (Blackwell) 编译 CUDA 内核，RTX 3090 是 SM86 (Ampere)，`import sgl_kernel` 直接 `undefined symbol`。

我们试过:
- 修改 `load_utils.py` 把 SM86 路由到 sm90 内核 → 符号能加载
- 二进制 patch + ASM shim 修 `c10_cuda_check_implementation` 等 6 个 ABI 不兼容符号 → 加载成功
- 但运行时 `sgl_kernel::rmsnorm` 等 CUDA kernel 实现本身不兼容 → **无解**

最终方案: **vLLM 0.28.0** 原生支持 SM86。[`poc/scripts/vllm_adapter.py`](poc/scripts/vllm_adapter.py) 是个 FastAPI 代理，在 vLLM 前面加一层:

| Slime 期望 | Adapter 行为 |
|---|---|
| `GET /health_generate` | 转发到 vLLM 的 `GET /health` |
| `GET /get_server_info` | 返回 `{model_path, tp_size, ...}` |
| `GET/POST /flush_cache` | 返回 `{"status":"ok"}` (no-op) |
| `POST /init_weights_update_group` | 返回 ok (no-op, vLLM 不支持在线权重推送) |
| `POST /update_weights_from_distributed` | 返回 ok (no-op) |
| `POST /release_weights_update_group` | 返回 ok (no-op) |
| `POST /v1/chat/completions` | 自动重写 model 名后透传到 vLLM |

---

## Slime 源码改动

所有改动仅为让 `--rollout-external` 模式对接 vLLM Adapter，不改核心训练逻辑:

| 文件 | 改动 |
|---|---|
| `slime/backends/sglang_utils/arguments.py` | `getattr` 默认值，避免 sglang 参数缺失时报错 |
| `slime/backends/sglang_utils/sglang_engine.py` | external 模式跳过 `_compute_server_args` |
| `slime/ray/rollout.py` | external 模式: num_engines 从 addrs 计算; engine 不分配 GPU; env_vars 传 LD_PRELOAD + 清代理 |
| `slime/ray/placement_group.py` | RolloutManager runtime_env 传 PYTHONPATH + 清代理 |
| `slime/ray/actor_group.py` | FSDP actor env_vars 传 PYTHONPATH/LD_PRELOAD + 清代理 |
| `slime/backends/fsdp_utils/actor.py` | `update_weights()` 在 external 模式直接 return |

---

## 踩坑必读

[`troubleshooting/README.md`](troubleshooting/README.md) 记录了 12 个真实遇到的问题:

| # | 问题 | 根因 |
|---|---|---|
| 1 | `pip list` 装好了但 `import` 失败 | 系统 `python3` 是 3.10，`pip` 装到 3.13 |
| 2 | `/home` 分区满 | `~/.cache` 被 pip/go/uv/poetry 塞了 15GB |
| 3 | `outlines_core` 编译失败 | 缺 `libssl-dev`，无 sudo |
| 4 | `sgl_kernel` 加载失败 | SM86 没有对应 CUDA 内核，**核心问题** |
| 5 | PyTorch / sgl-kernel ABI 不兼容 | 6 个 C++ 符号签名对不上 |
| 6 | transformers 版本地狱 | sglang/transformers/flashinfer 互相锁版本 |
| 7 | `PreTrainedConfig` 找不到 | transformers 5.x 改名 |
| 8 | `libtorchaudio.so` 加载失败 | torchaudio CUDA 版本与 torch 不匹配 |
| 9 | 公网 IP 访问不到本机服务 | NAT 后无端口转发 |
| 10 | OpenClaw "Invalid model" | agent 用的是 agent 名，不是 provider/model |
| 11 | OpenClaw "Context overflow" | gateway 注入的系统 prompt 太大 |
| 12 | OpenClaw "upstream provider timeout" | 首次请求要编译 CUDA graph |

---

## 上游架构要点

详细分析见 [`OpenClaw-RL-Analysis.md`](OpenClaw-RL-Analysis.md)。要点:

**双轨路线**
- Track 1 — 个性化 Agent: 通过日常对话收集反馈，持续优化 (本 POC 重点)
  - `openclaw-rl` (Binary RL/GRPO)、`openclaw-opd` (On-Policy Distillation)、`openclaw-combine` (Hybrid，推荐)、`openclaw-tinker` (云端 LoRA)、`openclaw-fireworks` (远程全参数)、`openclaw-test` (GSM8K 评估)
- Track 2 — 通用 Agent: 大规模环境并行
  - `terminal-rl`、`gui-rl`、`swe-rl`、`toolcall-rl`

**全异步 4 组件**: `Agent Serving → Rollout → PRM/Judge → Training`，互不阻塞，模型边服务边训练。

**三种学习范式**

| 范式 | 信号 | 来源 |
|---|---|---|
| Binary RL (GRPO) | 标量 reward (+1/-1/0) | PRM 对 next-state 打分 |
| OPD | Token 级 teacher-student gap | Hindsight hint 增强 teacher |
| **Hybrid (Combine)** | 两者 | **效果最好** |

**Combine 的关键创新** — Overlap-Guided Hint Selection: 对每个候选 hint，计算 student old policy top-k 词汇集与 hint-conditioned teacher top-k 词汇集的交集，选 overlap 最大的那个 hint，OPD 仅在 top-k 词汇子集上应用。

---

## 最终可用栈

```
推理引擎:  vLLM 0.28.0 (替代 sglang)
模型:      Qwen3-1.7B + LoRA rank=32
训练框架:  Slime (FSDP 后端，单卡)
兼容层:    vllm_adapter.py + ASM compat shim
GPU:       1× RTX 3090 (24GB), vLLM ~13GB + FSDP ~4.3GB
```

---

## 参考

- 论文: [arXiv:2603.10165](https://arxiv.org/abs/2603.10165)
- 上游代码: OpenClaw-RL (Apache 2.0)
- 训练后端: [Slime](https://github.com/SlimeAI/Slime) (本地 GPU) / [Tinker](https://thinkingmachines.ai/tinker/) (云端) / [Fireworks](https://fireworks.ai) (远程)
- 推理后端: [vLLM](https://github.com/vllm-project/vllm) (替代 sglang)
- 数据集: [SETA](https://github.com/camel-ai/seta) (terminal)、[ReTool-SFT](https://huggingface.co/datasets) + DAPO-Math-17k (tool-call)、SWE-Bench/SWE-Gym (SWE)、OSWorld (GUI)
