# Phase 3: On-Policy Distillation (OPD) 实现说明

## 概述

本文档说明如何在单卡 RTX 3090 环境下实现 Phase 3 的 On-Policy Distillation (OPD)。

## 方案对比

### 方案 A：使用 Megatron Backend（推荐但不可行）

Megatron backend 已经内置了完整的 OPD 支持：

```bash
python train_async.py \
    --train-backend megatron \
    --advantage-estimator on_policy_distillation \
    --distill-topk 4 \
    --distill-subset-mode overlap \
    ...
```

**优点**：
- 无需修改代码
- 支持论文中的所有 OPD 特性（overlap-guided selection, top-K 等）

**缺点**：
- ❌ RTX 3090 (SM86) 不兼容
- ❌ sgl-kernel 只支持 SM90+

### 方案 B：FSDP Backend + 最小代码改动（当前实现）

在 FSDP backend 中添加最小化的 OPD 支持。

**代码改动**：`slime/backends/fsdp_utils/actor.py`（约 20 行）

```python
elif self.args.advantage_estimator == "on_policy_distillation":
    # 使用 ref_model 作为 teacher
    # 计算 advantages = teacher_log_probs - student_log_probs
```

**优点**：
- ✅ 在 RTX 3090 上可用
- ✅ 代码改动最小化
- ✅ 复用现有的 ref_model

**缺点**：
- 需要修改 ~20 行代码
- 暂不支持 overlap-guided hint selection（可后续添加）

### 方案 C：Custom Loss Function（理论上可行）

使用 `--custom-loss-function-path` 实现 OPD loss。

**优点**：
- 不修改核心代码

**缺点**：
- 需要创建新的 loss 函数文件
- 需要理解 Slime 的内部数据结构
- 实现复杂度不低于方案 B

## 当前实现（方案 B）

### 代码改动

文件：`slime/backends/fsdp_utils/actor.py`

改动内容：
```python
# 在 _train_core 方法中添加
elif self.args.advantage_estimator == "on_policy_distillation":
    if self.ref_model is None:
        raise ValueError("on_policy_distillation requires ref_model")
    
    # 使用 ref_model 计算 teacher log_probs
    packed_batches_for_teacher, _ = self._packed_data(rollout_data)
    self._compute_log_prob("ref", packed_batches_for_teacher, store_prefix="teacher_")
    
    teacher_log_probs = [batch["teacher_cur_log_probs"] for batch in packed_batches_for_teacher]
    student_log_probs = rollout_data.get("rollout_log_probs")
    
    # OPD advantages: teacher - student
    rollout_data["advantages"] = rollout_data["returns"] = [
        (t_log_prob - s_log_prob).detach()
        for t_log_prob, s_log_prob in zip(teacher_log_probs, student_log_probs)
    ]
```

### 使用配置

```bash
python train_async.py \
    --train-backend fsdp \
    --advantage-estimator on_policy_distillation \
    --ref-load /home/wangwei/models/Qwen3-1.7B \
    --use-kl-loss \
    --kl-loss-coef 0.01 \
    ...
```

### 工作原理

1. **Teacher Model**：使用 `ref_model`（reference model）作为 teacher
2. **Advantage 计算**：`advantages = teacher_log_probs - student_log_probs`
3. **训练目标**：最小化 student 和 teacher 分布的 KL 散度

这与论文的 OPD 思想一致：
- 论文使用 PRM teacher 生成 hints
- 我们简化为使用 ref_model 作为 teacher
- 核心思想相同：通过 teacher-student 差异指导学习

## 后续优化

### 1. 集成 Phase 2 的 Hints

当前实现未使用 Phase 2 提取的 hints。可以扩展为：

```python
# 在 prompt 中附加 hint
if "hint" in rollout_data:
    hint_augmented_prompt = prompt + hint
    teacher_log_probs = compute_teacher_log_probs(hint_augmented_prompt)
```

### 2. Overlap-Guided Hint Selection

实现论文中的 overlap-guided selection：

```python
# 对每个 hint 计算 overlap
for hint in candidate_hints:
    student_topk = get_student_topk(prompt, k=4)
    teacher_topk = get_teacher_topk(prompt + hint, k=4)
    overlap = compute_overlap(student_topk, teacher_topk)
    
# 选择 overlap 最大的 hint
best_hint = max(candidate_hints, key=compute_overlap)
```

### 3. Top-K OPD Loss

只在 top-K tokens 上计算 OPD loss，减少计算量：

```python
# 论文中的 top-K OPD
topk_indices = get_topk(teacher_dist, k=4)
opd_loss = compute_opd_loss_on_subset(topk_indices)
```

## 总结

| 方案 | 代码改动 | RTX 3090 兼容 | 完整 OPD 特性 |
|------|---------|--------------|--------------|
| A: Megatron | 0 行 | ❌ | ✅ |
| B: FSDP (当前) | ~20 行 | ✅ | ⚠️ 基础版 |
| C: Custom Loss | 新文件 | ✅ | ⚠️ 取决于实现 |

**当前选择**：方案 B，在代码改动最小化的前提下，在 RTX 3090 上实现 OPD 的核心功能。

## 参考

- 论文 Section 3.3: On-Policy Distillation
- 论文 Algorithm 1: OpenClaw-RL Training Pipeline
- Megatron backend OPD 实现：`slime/backends/megatron_utils/loss.py`
