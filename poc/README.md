# OpenClaw-RL POC — Qwen3-1.7B 单卡 RTX 3090 验证

> 在 **1x RTX 3090 (24GB)** 上运行 OpenClaw-RL，使用 **Qwen3-1.7B + LoRA**
> OpenClaw 部署在 `124.221.28.203`

---

## 快速开始

### Step 0: 环境检查

```bash
bash OpenClaw-RL-ext/poc/scripts/setup_env.sh
```

这会自动：
- 检查 GPU 显存 (需 ≥ 24GB)
- 检查 Python 依赖
- 下载 Qwen3-1.7B 模型 (~3.5GB)
- 创建输出目录

### Step 1: 启动 Binary RL 训练 (路线第一步)

```bash
cd OpenClaw-RL/slime
bash ../OpenClaw-RL-ext/poc/openclaw-rl/run_qwen3_1.7b_openclaw_rl_1gpu.sh
```

或一键启动：
```bash
bash OpenClaw-RL-ext/poc/scripts/quickstart_rl.sh
```

启动后模型服务在 `http://0.0.0.0:30000/v1`

### Step 2: 配置 OpenClaw (124.221.28.203)

在 OpenClaw 的 `openclaw.json` 中添加 provider：

```json
{
  "models": {
    "providers": {
      "openclaw-rl-poc": {
        "baseUrl": "http://<训练机IP>:30000/v1",
        "apiKey": "",
        "api": "openai-completions",
        "models": [
          {
            "id": "qwen3-1.7b",
            "name": "Qwen3 1.7B (POC)",
            "reasoning": true,
            "input": ["text"],
            "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 },
            "contextWindow": 16384,
            "maxTokens": 4096
          }
        ]
      }
    }
  }
}
```

把 `<训练机IP>` 替换为你跑训练的机器的 IP。

### Step 3: 开始对话 → 自动训练

在 OpenClaw 中正常聊天。RL server 会自动：
1. 收集对话轨迹
2. PRM 打分 (基于下一轮对话反馈)
3. 后台更新 LoRA 权重

### Step 4: 切换到 Hybrid Combine (路线第二步，效果最好)

```bash
cd OpenClaw-RL/slime
bash ../OpenClaw-RL-ext/poc/openclaw-combine/run_qwen3_1.7b_openclaw_combine_1gpu.sh
```

或一键启动：
```bash
bash OpenClaw-RL-ext/poc/scripts/quickstart_combine.sh
```

---

## 目录结构

```
OpenClaw-RL-ext/poc/
├── README.md                          ← 本文件
│
├── openclaw-rl/                       ← Binary RL (GRPO) 单卡脚本
│   └── run_qwen3_1.7b_openclaw_rl_1gpu.sh
│
├── openclaw-combine/                  ← Hybrid GRPO+OPD 单卡脚本
│   └── run_qwen3_1.7b_openclaw_combine_1gpu.sh
│
├── toolcall-rl/                       ← Tool-call RL 单卡脚本
│   └── run_qwen3_1.7b_retool_rl_1gpu.sh
│
├── terminal-rl/                       ← Terminal RL 单卡脚本
│   └── run_qwen3_1.7b_terminal_rl_1gpu.sh
│
├── scripts/                           ← 辅助脚本
│   ├── setup_env.sh                   ← 环境检查 + 模型下载
│   ├── quickstart_rl.sh               ← 一键启动 Binary RL
│   ├── quickstart_combine.sh          ← 一键启动 Hybrid Combine
│   ├── quickstart_toolcall.sh         ← 一键启动 Tool-call RL
│   └── quickstart_terminal.sh         ← 一键启动 Terminal RL
│
├── docs/                              ← 文档
│   └── openclaw_config.json           ← OpenClaw 配置模板
│
└── ckpt/                              ← 训练检查点 (运行后自动生成)
    ├── qwen3-1.7b-openclaw-rl/
    ├── qwen3-1.7b-openclaw-combine/
    ├── qwen3-1.7b-retool-rl/
    └── qwen3-1.7b-terminal-rl/
```

---

## 技术细节

### 为什么能跑在单卡上？

| 对比 | 原始 (Qwen3.5-4B) | POC (Qwen3-1.7B) |
|---|---|---|
| 训练后端 | Megatron (TP=4) | **FSDP (TP=1)** |
| 训练方式 | 全参数 | **LoRA (rank=32)** |
| GPU 总数 | 8 | **1** |
| Actor GPU | 4 | **1** |
| Rollout GPU | 2 | **1** |
| PRM GPU | 2 | **1** |
| 模型大小 | ~8GB (bf16) | **~3.4GB (bf16)** |
| 总显存需求 | ~48GB+ | **~16-20GB** |

### 显存分配 (单卡 24GB)

```
Binary RL 模式:
┌────────────────────────────────────────────┐
│ Actor (FSDP+LoRA)  ≈ 6 GB  ████░░░░░░░░░ │
│ SGLang Rollout     ≈ 5 GB  ███░░░░░░░░░░░ │
│ PRM Judge          ≈ 4 GB  ██░░░░░░░░░░░░ │
│ 系统开销            ≈ 1 GB  █░░░░░░░░░░░░░ │
│ 剩余               ≈ 8 GB  ░░░░░░░░░░░░░░ │
└────────────────────────────────────────────┘

Hybrid Combine 模式 (多一个 Teacher):
┌────────────────────────────────────────────┐
│ Actor (FSDP+LoRA)  ≈ 6 GB  ████░░░░░░░░░ │
│ SGLang Rollout     ≈ 5 GB  ███░░░░░░░░░░░ │
│ PRM Judge          ≈ 4 GB  ██░░░░░░░░░░░░ │
│ Teacher (OPD)      ≈ 4 GB  ██░░░░░░░░░░░░ │
│ 系统开销            ≈ 1 GB  █░░░░░░░░░░░░░ │
│ 剩余               ≈ 4 GB  ░░░░░░░░░░░░░░ │
└────────────────────────────────────────────┘
```

### 关键参数说明

| 参数 | 值 | 说明 |
|---|---|---|
| `--train-backend fsdp` | FSDP | 替代 Megatron，支持单卡训练 |
| `--use-lora` | LoRA rank=32 | 只训练 ~1% 参数，大幅节省显存 |
| `--gradient-checkpointing` | 开启 | 用计算换显存 |
| `--max-tokens-per-gpu 4096` | 4096 | 降低 batch 大小适配单卡 |
| `--rollout-max-response-len 4096` | 4096 | 限制最大生成长度 |
| `--sglang-mem-fraction-static 0.75` | 75% | SGLang KV cache 比例 |
| `--context-length 16384` | 16K | 上下文窗口 (降低以节省显存) |

---

## 环境变量参考

### 通用

| 变量 | 默认值 | 说明 |
|---|---|---|
| `HF_CKPT` | `~/models/Qwen3-1.7B` | 模型路径 |
| `PORT` | `30000` | API 服务端口 |
| `USE_WANDB` | `0` | 是否启用 W&B 日志 |
| `WANDB_KEY` | — | W&B API key |

### Binary RL 专用

| 变量 | 默认值 | 说明 |
|---|---|---|
| `PRM_M` | `3` | PRM 投票次数 |
| `PRM_TEMPERATURE` | `0.6` | PRM 采样温度 |

### Hybrid Combine 专用

| 变量 | 默认值 | 说明 |
|---|---|---|
| `OPENCLAW_COMBINE_W_RL` | `1.0` | GRPO 损失权重 |
| `OPENCLAW_COMBINE_W_OPD` | `1.0` | OPD 蒸馏损失权重 |
| `TRAIN_EPOCHS` | `2` | 每个样本重复训练次数 |

---

## 常见问题

### Q: 显存不够 (OOM) 怎么办？

```bash
# 降低 context length
export CONTEXT_LENGTH=8192

# 降低 max tokens per gpu
export MAX_TOKENS_PER_GPU=2048

# 降低 SGLang 内存比例
export MEM_FRACTION_STATIC=0.65

# 关闭 PRM (省 ~4GB, 但失去 reward 信号)
# 修改脚本: 把 PRM_GPUS=1 改为 PRM_GPUS=0
```

### Q: 训练太慢？

```bash
# 增大 batch size (如果显存允许)
# 修改脚本中 --rollout-batch-size 从 8 改为 16

# 增加 PRM 投票数以获得更稳定的 reward
export PRM_M=5
```

### Q: 如何查看训练进度？

```bash
# Ray Dashboard
# 浏览器打开 http://localhost:8265

# 训练记录
cat OpenClaw-RL-ext/poc/openclaw-rl/results/qwen3_1.7b_record.jsonl

# 检查点
ls OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-openclaw-rl/
```

### Q: 如何导出训练好的模型？

LoRA 权重在 checkpoint 目录中。可以合并到基础模型：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("~/models/Qwen3-1.7B")
model = PeftModel.from_pretrained(base, "OpenClaw-RL-ext/poc/ckpt/qwen3-1.7b-openclaw-rl/checkpoint-xxx")
model = model.merge_and_unload()
model.save_pretrained("~/models/Qwen3-1.7B-rl-merged")
```

---

## 路线图

- [x] Binary RL (GRPO) 单卡脚本 — `openclaw-rl/`
- [x] Hybrid Combine 单卡脚本 — `openclaw-combine/`
- [x] Tool-call RL 单卡脚本 — `toolcall-rl/`
- [x] Terminal RL 单卡脚本 — `terminal-rl/`
- [x] 环境准备脚本
- [x] OpenClaw 配置模板

---

## 全部场景快速启动

| 场景 | 一键启动 | 说明 |
|---|---|---|
| **Binary RL** | `bash poc/scripts/quickstart_rl.sh` | 核心场景，对接 OpenClaw |
| **Hybrid Combine** | `bash poc/scripts/quickstart_combine.sh` | 效果最好 |
| **Tool-call RL** | `bash poc/scripts/quickstart_toolcall.sh` | 学代码调用解数学题 |
| **Terminal RL** | `bash poc/scripts/quickstart_terminal.sh` | 学终端命令生成 (需 Docker Worker) |
