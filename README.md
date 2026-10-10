# OpenClaw-RL-ext

[OpenClaw-RL](https://arxiv.org/abs/2603.10165) 的扩展仓库 — 把原本需要 8×H100 才能跑的全异步 RL Agent 训练框架，压到 **单张 RTX 3090 (24GB) + Qwen3-1.7B + LoRA** 上跑通。

> 上游: 通过日常对话训练个性化 AI Agent 的全异步强化学习框架,支持 Terminal / GUI / SWE / Tool-call 等通用 Agent 场景。

本仓库做三件事:

1. **单卡 POC** — 为 Track 1 (个性化) 和 Track 2 (通用 Agent) 的 4 个训练场景各提供一套能在单张消费级 GPU 上启动的脚本;
2. **vLLM 兼容层** — sgl-kernel 只为 SM90/SM100 编译内核,RTX 3090 (SM86) 跑不了 sglang,所以用 vLLM 0.28 替代,并用一个 FastAPI adapter 补齐 Slime 期望的 `/health_generate`、`/get_server_info` 等端点;
3. **踩坑记录 + 深度分析** — 把部署过程中遇到的 Python 版本错位、依赖地狱、NAT 穿透、OpenClaw gateway 对接等问题一一记录下来,并附上一份对上游项目架构的完整分析。

---

## 仓库结构

```
OpenClaw-RL-ext/
├── README.md                     ← 本文件
├── OpenClaw-RL-Analysis.md       ← 上游 OpenClaw-RL 架构深度分析 (中文)
│
├── poc/                          ← 单卡 POC (1× RTX 3090, Qwen3-1.7B + LoRA)
│   ├── README.md                 ← POC 自己的快速开始
│   ├── start.sh                  ← (已弃用) 老的一键启动
│   ├── adapter.log / vllm.log    ← 运行日志
│   │
│   ├── openclaw-rl/              ← Binary RL (GRPO) 单卡脚本
│   ├── openclaw-combine/         ← Hybrid GRPO + Top-K OPD 单卡脚本
│   ├── toolcall-rl/              ← Tool-call RL (代码解释器 + 数学)单卡脚本
│   ├── terminal-rl/              ← Terminal RL (Shell 沙箱)单卡脚本
│   │
│   ├── scripts/                  ← 辅助脚本
│   │   ├── setup_env.sh          ← 环境检查 + 模型下载
│   │   ├── quickstart_{rl,combine,toolcall,terminal}.sh
│   │   ├── start_vllm_server.sh
│   │   ├── run_slime_vllm_training.sh
│   │   ├── run_full_chain.sh
│   │   └── vllm_adapter.py       ← Slime ↔ vLLM 兼容层
│   │
│   ├── docs/
│   │   ├── openclaw_config.json  ← OpenClaw provider 配置模板
│   │   └── VERIFICATION_REPORT.md
│   │
│   └── ckpt/                     ← 训练检查点 (运行后自动生成)
│
└── troubleshooting/
    └── README.md                 ← 12 个踩坑条目,从 Python 版本到 OpenClaw timeout
```

---

## 单卡 POC 快速开始

完整步骤见 [`poc/README.md`](poc/README.md),核心三步:

```bash
# 1. 环境检查 + 下载 Qwen3-1.7B (~3.5GB)
bash OpenClaw-RL-ext/poc/scripts/setup_env.sh

# 2. 启动训练 (Binary RL,最基础的路线)
cd OpenClaw-RL/slime
bash ../OpenClaw-RL-ext/poc/openclaw-rl/run_qwen3_1.7b_openclaw_rl_1gpu.sh
# API 服务: http://0.0.0.0:30000/v1

# 3. (可选) 切到效果最好的 Hybrid Combine
bash ../OpenClaw-RL-ext/poc/openclaw-combine/run_qwen3_1.7b_openclaw_combine_1gpu.sh
```

每个场景都有 `quickstart_*.sh` 一键脚本:

| 场景 | 命令 | 学什么 |
|---|---|---|
| Binary RL (GRPO) | `poc/scripts/quickstart_rl.sh` | 用下一轮对话当 reward 信号,对接 OpenClaw |
| Hybrid Combine | `poc/scripts/quickstart_combine.sh` | GRPO + Top-K OPD 混合,效果最好 |
| Tool-call RL | `poc/scripts/quickstart_toolcall.sh` | 学调用代码解释器解数学题 |
| Terminal RL | `poc/scripts/quickstart_terminal.sh` | 学生成 shell 命令 (需要额外的 Docker worker) |

### 单卡显存分配 (24GB)

```
Binary RL:
  Actor (FSDP+LoRA)    ≈  6 GB  ████░░░░░░░░
  SGLang/vLLM Rollout  ≈  5 GB  ███░░░░░░░░░
  PRM Judge            ≈  4 GB  ██░░░░░░░░░░
  系统开销              ≈  1 GB  █░░░░░░░░░░░
  空闲                  ≈  8 GB

Hybrid Combine (多一个 Teacher):
  Actor (FSDP+LoRA)    ≈  6 GB  ████░░░░░░░░
  SGLang/vLLM Rollout  ≈  5 GB  ███░░░░░░░░░
  PRM Judge            ≈  4 GB  ██░░░░░░░░░░
  Teacher (OPD)        ≈  4 GB  ██░░░░░░░░░░
  系统开销              ≈  1 GB  █░░░░░░░░░░░
  空闲                  ≈  4 GB
```

OOM 时优先调:`CONTEXT_LENGTH=8192`、`MAX_TOKENS_PER_GPU=2048`、`MEM_FRACTION_STATIC=0.65`。

---

## vLLM Adapter — 为什么不用 sglang

上游 OpenClaw-RL 假设推理后端是 sglang。但 `sgl-kernel` 只为 SM90 (H100) 和 SM100 (Blackwell) 编译 CUDA 内核,RTX 3090 是 SM86 (Ampere),`import sgl_kernel` 直接 `undefined symbol`。

我们试过:
- 修改 `load_utils.py` 把 SM86 路由到 sm90 内核 → 符号能加载
- 二进制 patch + ASM shim 修 `c10_cuda_check_implementation` 等 6 个 ABI 不兼容符号 → 加载成功
- 但运行时 `sgl_kernel::rmsnorm` 等 CUDA kernel 实现本身不兼容 → **无解**

最终方案:**vLLM 0.28.0** 原生支持 SM86。[`poc/scripts/vllm_adapter.py`](poc/scripts/vllm_adapter.py) 是个 FastAPI 代理,在 vLLM 前面加一层:

| Slime 期望 | Adapter 行为 |
|---|---|
| `GET /health_generate` | 转发到 vLLM 的 `GET /health` |
| `GET /get_server_info` | 返回 Slime 需要的 `{model_type, ...}` 结构 |
| `POST /v1/chat/completions` | 直接透传到 vLLM |

启动:`python poc/scripts/vllm_adapter.py --vllm-url http://localhost:30000 --port 30001`

---

## 对接 OpenClaw

POC 跑起来之后,需要让远程的 OpenClaw gateway (例如 `124.221.28.203`) 能调到本机的 vLLM。本机在 NAT 后面,用 SSH 反向隧道:

```bash
# 本机 → OpenClaw 服务器,把远程的 30000/30001 转发回本机
ssh -f -N -R 30001:localhost:30001 -R 30000:localhost:30000 root@124.221.28.203
```

然后把 [`poc/docs/openclaw_config.json`](poc/docs/openclaw_config.json) 里的 `models.providers` 段合并到 OpenClaw 的 `openclaw.json`,把 agent 的 `model.primary` 改成 `openclaw-rl-poc/<模型路径>`。

详见 [`troubleshooting/README.md`](troubleshooting/README.md) 第 9–12 条。

---

## 踩坑必读

[`troubleshooting/README.md`](troubleshooting/README.md) 记录了 12 个真实遇到的问题,按顺序覆盖:

| # | 问题 | 根因 |
|---|---|---|
| 1 | `pip list` 装好了但 `import` 失败 | 系统 `python3` 是 3.10,`pip` 装到 3.13 |
| 2 | `/home` 分区满 | `~/.cache` 被 pip/go/uv/poetry 塞了 15GB |
| 3 | `outlines_core` 编译失败 | 缺 `libssl-dev`,无 sudo |
| 4 | `sgl_kernel` 加载失败 | SM86 没有对应 CUDA 内核,**核心问题** |
| 5 | PyTorch / sgl-kernel ABI 不兼容 | 6 个 C++ 符号签名对不上 |
| 6 | transformers 版本地狱 | sglang/transformers/flashinfer 互相锁版本 |
| 7 | `PreTrainedConfig` 找不到 | transformers 5.x 改名 |
| 8 | `libtorchaudio.so` 加载失败 | torchaudio CUDA 版本与 torch 不匹配 |
| 9 | 公网 IP 访问不到本机服务 | NAT 后无端口转发 |
| 10 | OpenClaw "Invalid model" | agent 用的是 agent 名,不是 provider/model |
| 11 | OpenClaw "Context overflow" | gateway 注入的系统 prompt 太大 |
| 12 | OpenClaw "upstream provider timeout" | 首次请求要编译 CUDA graph |

---

## 上游架构要点

详细分析见 [`OpenClaw-RL-Analysis.md`](OpenClaw-RL-Analysis.md)。要点:

**双轨路线**
- Track 1 — 个性化 Agent:通过日常对话收集反馈,持续优化 (本 POC 重点)
  - `openclaw-rl` (Binary RL/GRPO)、`openclaw-opd` (On-Policy Distillation)、`openclaw-combine` (Hybrid,推荐)、`openclaw-tinker` (云端 LoRA)、`openclaw-fireworks` (远程全参数)、`openclaw-test` (GSM8K 评估)
- Track 2 — 通用 Agent:大规模环境并行
  - `terminal-rl`、`gui-rl`、`swe-rl`、`toolcall-rl`

**全异步 4 组件**:`Agent Serving → Rollout → PRM/Judge → Training`,互不阻塞,模型边服务边训练。

**三种学习范式**
| 范式 | 信号 | 来源 |
|---|---|---|
| Binary RL (GRPO) | 标量 reward (+1/-1/0) | PRM 对 next-state 打分 |
| OPD | Token 级 teacher-student gap | Hindsight hint 增强 teacher |
| **Hybrid (Combine)** | 两者 | **效果最好** |

**Combine 的关键创新** — Overlap-Guided Hint Selection:对每个候选 hint,计算 student old policy top-k 词汇集与 hint-conditioned teacher top-k 词汇集的交集,选 overlap 最大的那个 hint,OPD 仅在 top-k 词汇子集上应用。

---

## 最终可用栈

```
推理引擎:  vLLM 0.28.0 (替代 sglang)
模型:      Qwen3-1.7B + LoRA rank=32
训练框架:  Slime (FSDP 后端,单卡)
兼容层:    vllm_adapter.py
网络:      SSH 反向隧道 → OpenClaw gateway
GPU:       1× RTX 3090 (24GB),vLLM 占 ~20GB @ 32K context
```

---

## 参考

- 论文:[arXiv:2603.10165](https://arxiv.org/abs/2603.10165)
- 上游代码:OpenClaw-RL (Apache 2.0)
- 训练后端:[Slime](https://github.com/SlimeAI/Slime) (本地 GPU) / [Tinker](https://thinkingmachines.ai/tinker/) (云端) / [Fireworks](https://fireworks.ai) (远程)
- 推理后端:[vLLM](https://github.com/vllm-project/vllm) (替代 sglang)
- 数据集:[SETA](https://github.com/camel-ai/seta) (terminal)、[ReTool-SFT](https://huggingface.co/datasets) + DAPO-Math-17k (tool-call)、SWE-Bench/SWE-Gym (SWE)、OSWorld (GUI)
