# Phase 3: On-Policy Distillation (OPD) Loss 实现

## 实现概述

基于论文 "OpenClaw-RL: Training Personalized AI Agents Through Conversations" 的 On-Policy Distillation (OPD) 方法，在 FSDP backend 中实现了最小化的 OPD loss。

## 核心思想

OPD 利用 teacher model（这里使用 ref_model）的分布来指导 student model 的训练：

```
advantages = teacher_log_probs - student_log_probs
```

这与 GRPO 的 reward-based advantages 不同，OPD 使用 token 级别的 teacher-student 分布差异作为学习信号。

## 实现细节

### 1. 代码改动

**文件**: `slime/slime/backends/fsdp_utils/actor.py`

**改动 1**: 在 `train()` 方法中添加 teacher forward pass
```python
# Phase 3: For OPD, compute teacher log_probs before training
if self.args.advantage_estimator == "on_policy_distillation":
    if self.ref_model is None:
        raise ValueError("on_policy_distillation requires ref_model")
    with timer("teacher_forward"):
        self._compute_teacher_log_probs(rollout_data)
```

**改动 2**: 新增 `_compute_teacher_log_probs()` 方法
```python
def _compute_teacher_log_probs(self, rollout_data: dict) -> None:
    """Compute teacher (ref_model) log_probs for OPD loss."""
    # 对每个样本，用 ref_model 计算 student 生成的 tokens 的 log_probs
    # 返回 teacher_log_probs 列表
```

**改动 3**: 在 `_train_core()` 中添加 OPD advantage 计算
```python
elif self.args.advantage_estimator == "on_policy_distillation":
    # OPD advantages: teacher - student (per-token)
    rollout_data["advantages"] = rollout_data["returns"] = [
        (t_log_prob - s_log_prob).detach()
        for t_log_prob, s_log_prob in zip(teacher_log_probs, student_log_probs)
    ]
```

### 2. 使用方法

在训练脚本中，将 `--advantage-estimator grpo` 改为：
```bash
--advantage-estimator on_policy_distillation
```

### 3. 工作原理

1. **Student 生成**: Student model 生成回复，产生 `rollout_log_probs`
2. **Teacher 评估**: Ref model（作为 teacher）对 student 生成的 tokens 计算 log_probs
3. **Advantage 计算**: `advantages = teacher_log_probs - student_log_probs`
   - 正值：teacher 认为这个 token 比 student 更可能 → student 应该学习
   - 负值：teacher 认为这个 token 不如 student → student 应该避免
4. **策略更新**: 使用这些 advantages 更新 student policy

## 与 Phase 2 Hints 的结合

当前实现使用 ref_model 作为 teacher，这是一个简化的 OPD。完整的论文方法包括：

1. **Hint-augmented teacher**: Teacher 在看到 hint 后生成分布
2. **Overlap-guided selection**: 选择与 student top-k 重叠最大的 hint
3. **Top-k OPD**: 只在 top-k tokens 上计算 OPD loss

当前简化版本已经可以提供 distillation 信号。完整的 hint-augmented OPD 需要：
- 在 `_compute_teacher_log_probs` 中，将 hint 附加到 prompt
- 实现 overlap-guided hint selection
- 这些可以作为后续优化

## 性能影响

- **额外计算**: 每个 training step 需要额外的 ref_model forward pass
- **内存**: 需要存储 teacher_log_probs（与 rollout_log_probs 相同大小）
- **时间**: 约增加 30-50% 的训练时间（取决于 ref_model 大小）

## 测试建议

1. **小规模测试**: 先用小 batch size 验证 OPD 能正常工作
2. **对比实验**: 
   - GRPO only (baseline)
   - OPD only
   - Hybrid (GRPO + OPD) - 需要修改 loss 计算
3. **指标监控**:
   - `teacher_log_probs` vs `student_log_probs` 的差异
   - KL divergence 变化
   - 训练稳定性

## 后续优化

1. **Hint-augmented OPD**: 使用 Phase 2 提取的 hints 增强 teacher
2. **Overlap-guided selection**: 实现论文中的 hint 选择算法
3. **Top-k OPD**: 只在 top-k tokens 上计算 loss，减少计算量
4. **Hybrid loss**: GRPO + OPD 的加权组合

## 配置示例

```bash
# 纯 OPD 训练
python3 train_async.py \
  --advantage-estimator on_policy_distillation \
  --ref-load /path/to/ref/model \
  ...

# 未来的 Hybrid 训练（需要额外实现）
python3 train_async.py \
  --advantage-estimator hybrid \
  --hybrid-alpha 0.5 \  # GRPO 权重
  --hybrid-beta 0.5 \   # OPD 权重
  ...
```

## 总结

本次实现遵循最小改动原则：
- ✅ 仅修改 1 个文件（actor.py）
- ✅ 新增 ~50 行代码
- ✅ 不改变现有 GRPO 逻辑
- ✅ 通过配置参数启用
- ✅ 可回退到 GRPO

这为后续的 hint-augmented OPD 和 overlap-guided selection 奠定了基础。
