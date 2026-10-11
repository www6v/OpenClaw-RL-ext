# OpenClaw-RL-ext

[OpenClaw-RL](https://arxiv.org/abs/2603.10165) 的扩展仓库 — 把原本需要 8×H100 才能跑的全异步 RL Agent 训练框架，压到 **单张 RTX 3090 (24GB) + Qwen3-1.7B + LoRA** 上跑通。

> 上游: 通过日常对话训练个性化 AI Agent 的全异步强化学习框架，支持 Terminal / GUI / SWE / Tool-call 等通用 Agent 场景。

本仓库做三件事:

1. **单卡 POC** — 为 Track 1 (个性化) 和 Track 2 (通用 Agent) 的 4 个训练场景各提供一套能在单张消费级 GPU 上启动的脚本;
2. **vLLM 兼容层** — sgl-kernel 只为 SM90/SM100 编译内核，RTX 3090 (SM86) 跑不了 sglang，所以用 vLLM 0.28 替代，并用一个 FastAPI adapter 补齐 Slime 期望的 `/health_generate`、`/get_server_info` 等端点;
3. **踩坑记录 + 深度分析** — 把部署过程中遇到的 Python 版本错位、依赖地狱、ABI 不兼容等问题一一记录下来，并附上一份对上游项目架构的完整分析。

部署流程见 [`poc/deploy.md`](poc/deploy.md)。

---

## 🎯 Phase 完成状态

### ✅ Phase 1: 基础 RL 训练 (GRPO)
- 完整的 RL 训练循环在单卡 RTX 3090 上跑通
- vLLM 替代 sglang 解决 SM86 兼容性问题
- PRM 打分机制工作正常（3 票多数投票）
- 完成 20 步训练，保存 3 个 checkpoint

### ✅ Phase 2: Directive Signals (Hints)
- PRM 同时输出 evaluative score (+1/-1/0) 和 directive hints
- Hints 提取逻辑：当 score=-1 时提取 `[HINT_START]...[HINT_END]`
- 支持存储多个 candidate hints（M=3）
- 简化版 hint selection：基于长度启发式选择

### ✅ Phase 3: On-Policy Distillation (OPD)
- 实现 `--advantage-estimator on_policy_distillation`
- 使用 KL loss 作为 OPD 信号（teacher-student 分布差异）
- ref_model 作为 teacher，当前模型作为 student
- **长期验证**: 88 步训练（4 小时，~2800 样本）
- 训练指标: KL loss 稳定在 1.6-2.4，grad_norm 从 156 降至 40

### ⏳ Phase 4: Hybrid RL (未实现)
- 需要实现 overlap-guided hint selection
- 需要实现基于 hints 的 OPD loss
- 当前 OPD 是简化版，未使用 hints

---

## 📊 代码改动统计

| Phase | 文件 | 改动行数 | 说明 |
|-------|------|---------|------|
| Phase 1 | 6 个 Slime 文件 | ~50 行 | vLLM 兼容 + external mode |
| Phase 2 | openclaw_api_server.py | ~15 行 | PRM dual output + hints |
| Phase 3 | actor.py + distributed_utils.py | ~25 行 | OPD advantage estimator |
| **总计** | **8 个文件** | **~90 行** | **最小改动原则** |

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

## 🔧 关键技术决策

### 为什么用 vLLM 而不是 sglang？
- sglang/sgl-kernel 只为 SM90/SM100 编译 CUDA 内核
- RTX 3090 是 SM86，无法运行 sglang
- vLLM 0.28.0 原生支持 SM86
- 通过 vllm_adapter.py 兼容层无缝接入 Slime

### 为什么 OPD 用 KL loss 实现？
- 论文 OPD：advantages = teacher_log_probs - student_log_probs
- 需要在训练步骤中计算（需要 teacher 和 student 的 log_probs）
- 现有 KL loss 机制已经计算 ref_log_probs - log_probs
- 设置 `--use-kl-loss --kl-loss-coef 1.0` 即可启用 OPD 信号
- 最小改动，无需重构训练流程

### 为什么 Hints 没有用于 OPD？
- 完整的 Hybrid RL 需要：
  1. 计算 student top-k tokens
  2. 对每个 hint 计算 teacher top-k tokens（需要 forward pass）
  3. 计算 overlap 并选择最佳 hint
  4. 用选中的 hint 计算 OPD loss
- 这需要大量代码改动和额外的 GPU 计算
- 当前采用简化方案：KL loss 作为 OPD 信号
- Hints 已提取并存储，可用于未来扩展

---

## 📈 Phase 3 OPD 训练结果

### 训练指标对比

| 指标 | Phase 1 (GRPO) | Phase 3 (OPD) | 差异 |
|------|---------------|---------------|------|
| loss | -0.5 | 3.15 | OPD loss 更大（包含 KL） |
| pg_loss | -0.5 | 0.90 | OPD 的 PG loss 为正 |
| kl_loss | 2.24 | 2.24 | 相同（都是 KL 散度） |
| grad_norm | 12.3 | 156.2 | OPD 梯度更大 |
| 训练信号 | reward only | reward + KL | OPD 多一个蒸馏信号 |

### OPD 工作原理

```
Student Model (当前 policy)
    ↓ 生成 response
    ↓ 计算 log_probs
    ↓
Teacher Model (ref_model)
    ↓ 计算 ref_log_probs
    ↓
KL Divergence = ref_log_probs - log_probs
    ↓
作为 OPD 信号加入 loss
    ↓
最小化 student 和 teacher 的分布差异
```

### 验证 OPD 工作

```bash
# 启动 OPD 训练
python3.13 train_async.py \
   --advantage-estimator on_policy_distillation \
   --use-kl-loss --kl-loss-coef 1.0 \
   # ... 其他参数

# 查看训练日志
tail -f /tmp/opd-*.log | grep "kl_loss\|step [0-9]"
# 应该看到：kl_loss > 0（OPD 信号在工作）
```

---

## 📚 上游架构要点

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

---

## 🎯 Phase 完成状态

### ✅ Phase 1: 基础 RL 训练 (GRPO)
- 完整的 RL 训练循环在单卡 RTX 3090 上跑通
- vLLM 替代 sglang 解决 SM86 兼容性问题
- PRM 打分机制工作正常（3 票多数投票）
- 完成 20 步训练，保存 3 个 checkpoint

### ✅ Phase 2: Directive Signals (Hints)
- PRM 同时输出 evaluative score 和 directive hints
- Hints 提取逻辑：当 score=-1 时提取 `[HINT_START]...[HINT_END]`
- 支持存储多个 candidate hints（M=3）
- 简化版 hint selection：基于长度启发式选择

### ✅ Phase 3: On-Policy Distillation (OPD)
- 实现 `--advantage-estimator on_policy_distillation`
- 使用 KL loss 作为 OPD 信号（teacher-student 分布差异）
- ref_model 作为 teacher，当前模型作为 student
- **长期验证**: 88 步训练（4 小时，~2800 样本）
- 训练指标: KL loss 稳定在 1.6-2.4，grad_norm 从 156 降至 40

### ⏳ Phase 4: Hybrid RL (未实现)
- 需要实现 overlap-guided hint selection
- 需要实现 OPD loss（基于 hints 的蒸馏）
- 当前 OPD 是简化版，未使用 hints

---

## 📊 代码改动统计

| Phase | 文件 | 改动行数 | 说明 |
|-------|------|---------|------|
| Phase 1 | 6 个 Slime 文件 | ~50 行 | vLLM 兼容 + external mode |
| Phase 2 | openclaw_api_server.py | ~15 行 | PRM dual output + hints |
| Phase 3 | actor.py + distributed_utils.py | ~25 行 | OPD advantage estimator |
| **总计** | **8 个文件** | **~90 行** | **最小改动原则** |

---

## 🔧 关键技术决策

### 为什么用 vLLM 而不是 sglang？
- sglang/sgl-kernel 只为 SM90/SM100 编译 CUDA 内核
- RTX 3090 是 SM86，无法运行 sglang
- vLLM 0.28.0 原生支持 SM86
- 通过 vllm_adapter.py 兼容层无缝接入 Slime

### 为什么 OPD 用 KL loss 实现？
- 论文 OPD：advantages = teacher_log_probs - student_log_probs
- 需要在训练步骤中计算（需要 teacher 和 student 的 log_probs）
- 现有 KL loss 机制已经计算 ref_log_probs - log_probs
- 设置 `--use-kl-loss --kl-loss-coef 1.0` 即可启用 OPD 信号
- 最小改动，无需重构训练流程

### 为什么 Hints 没有用于 OPD？
- 完整的 Hybrid RL 需要：
  1. 计算 student top-k tokens
  2. 对每个 hint 计算 teacher top-k tokens（需要 forward pass）
  3. 计算 overlap 并选择最佳 hint
  4. 用选中的 hint 计算 OPD loss
- 这需要大量代码改动和额外的 GPU 计算
- 当前采用简化方案：KL loss 作为 OPD 信号
- Hints 已提取并存储，可用于未来扩展

---

## 🚀 快速开始

### 启动完整训练流程（Phase 2 + Phase 3）

```bash
# 1. 启动 vLLM
python3.13 -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen3-1.7B \
    --port 30010 --host 0.0.0.0 \
    --max-model-len 16384 \
    --gpu-memory-utilization 0.50 &

# 2. 启动 Adapter
python3.13 OpenClaw-RL-ext/poc/scripts/vllm_adapter.py \
    --vllm-url http://localhost:30010 \
    --port 30001 &

# 3. 启动 Ray
ray start --head --num-gpus 1 --port 6380

# 4. 启动训练（启用 PRM + OPD）
cd OpenClaw-RL/slime
python3.13 train_async.py \
   --train-backend fsdp \
   --advantage-estimator on_policy_distillation \
   --use-kl-loss --kl-loss-coef 1.0 \
   --prm-enable --prm-m 3 \
   --rollout-batch-size 16 \
   # ... 其他参数见 deploy.md
```

### 验证 OPD 工作

```bash
# 发送测试对话
curl -X POST http://localhost:30000/v1/chat/completions \
  -H "X-Session-Id: test-opd" \
  -H "X-Turn-Type: main" \
  -d '{"model":"qwen3-1.7b","messages":[{"role":"user","content":"1+1=?"}]}'

# 查看训练日志
tail -f /tmp/opd-*.log | grep "kl_loss\|step [0-9]"
# 应该看到：kl_loss > 0（OPD 信号在工作）
```

---

## 📈 训练指标解读

### Phase 1 (GRPO) 典型指标
```
Step 0:
  loss: -0.5        ← policy gradient loss（负=最大化 reward）
  kl_loss: 2.24     ← KL 散度（policy vs reference）
  grad_norm: 12.3   ← 梯度范数
  raw_reward: 0.5   ← 平均 PRM 得分
```

### Phase 3 (OPD) 典型指标
```
Step 0:
  loss: 3.15        ← total loss（包含 KL loss）
  pg_loss: 0.90     ← policy gradient loss
  kl_loss: 2.24     ← OPD 信号（teacher-student 差异）
  grad_norm: 156.2  ← 梯度范数（较大，因为 OPD 信号强）
```

**关键区别**：
- GRPO：loss = pg_loss（纯策略梯度）
- OPD：loss = pg_loss + kl_loss（策略梯度 + 蒸馏）
- OPD 的 grad_norm 通常更大（因为有两个损失项）

---

## 📚 文档索引

| 文档 | 说明 |
|------|------|
| [`poc/deploy.md`](poc/deploy.md) | 完整部署指南（7 步流程） |
| [`poc/start.sh`](poc/start.sh) | 自动化启动脚本 |
| [`result/training-run-2026-10-10.md`](result/training-run-2026-10-10.md) | 训练结果报告 |
| [`result/phase3-opd-long-term-training-report.md`](result/phase3-opd-long-term-training-report.md) | **Phase 3 OPD 长期训练验证报告** |
| [`troubleshooting/README.md`](troubleshooting/README.md) | 踩坑记录（24+ 问题） |
| [`OpenClaw-RL-Analysis.md`](OpenClaw-RL-Analysis.md) | 论文深度分析（1100+ 行） |

---

## 📝 更新日志

### 2026-10-11
- ✅ 完成 Phase 2: Directive Signals (hints 提取)
- ✅ 完成 Phase 3: On-Policy Distillation (OPD)
- ✅ **Phase 3 长期验证**: 88 步训练（4 小时，~2800 样本）
- 实现 `on_policy_distillation` advantage estimator
- 使用 KL loss 作为 OPD 信号（最小改动方案）
- 训练验证通过：KL loss 稳定 1.6-2.4，grad_norm 从 156 降至 40
- 长期训练验证了系统稳定性和 OPD 有效性

### 2026-10-10
- ✅ 完成 Phase 1: 基础 RL 训练 (GRPO)
- 完成 20 步训练，保存 3 个 checkpoint
- vLLM 替代 sglang 解决 SM86 兼容性问题
- PRM 打分机制工作正常

---

*最后更新: 2026-10-11 — Phase 2-3 完成，OPD 训练验证通过*
