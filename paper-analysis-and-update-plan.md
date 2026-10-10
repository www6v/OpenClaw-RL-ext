# OpenClaw-RL 论文分析与方案更新

> 基于论文 arXiv:2603.10165v2 的完整分析

## 一、论文核心洞察

### 1.1 Next-State Signal（下一状态信号）

**关键发现**: 每次 agent 交互后，用户的回复、工具输出、GUI 状态变化等构成"下一状态信号"，这是一个实时的、在线的学习信号源。

**两种互补信号类型**:

| 信号类型 | 来源 | 粒度 | 频率 | 信息量 |
|---|---|---|---|---|
| **Evaluative（评估性）** | PRM 打分 | 标量 (scalar) | 每个 scored turn | 低（仅 +1/-1/0） |
| **Directive（指导性）** | PRM 提取 hints | token 级 | 仅含修正的 turns | 高（具体改进方向） |

**核心洞察**: 这两种信号是互补的：
- 评估信号频繁但信息少
- 指导信号稀疏但信息丰富
- 两者结合 = Hybrid RL Objective

### 1.2 Hybrid RL Objective（混合 RL 目标）

**公式**:
```
L_hybrid = w_RL * L_GRPO + w_OPD * L_OPD
```

其中：
- `L_GRPO`: 标准 GRPO clipped surrogate loss（由评估信号驱动）
- `L_OPD`: On-Policy Distillation loss（由指导信号驱动）
- 默认权重: `w_RL = w_OPD = 1`

### 1.3 Overlap-Guided Hint Selection（重叠引导的 Hint 选择）

**问题**: Teacher-student distribution mismatch 导致训练不稳定

**解决方案**: 选择与 student 的 top-k tokens 重叠最大的 hint

**算法**:
1. PRM 生成 M=3 个候选 hints
2. 对每个 hint h，计算 teacher 分布的 top-k vocab
3. 计算与学生 top-k vocab 的重叠度 O[h,i]
4. 选择最大化序列重叠的 hint: `h* = argmax_h Σ_i O[h,i]`

**关键参数**: k=4（论文 ablation 显示 k≥4 后增益很小）

### 1.4 Top-k OPD Loss with Log-Prob-Difference Clip

**OPD 损失公式**:
```
A_v = Δ_v * w_v
w_v = softmax_{v∈S_i}(ℓ_old(v))
Δ_v = clip(ℓ_T,h*(v) - ℓ_old(v), -C, +C)
ρ_v = exp(ℓ_cur(v) - ℓ_old(v))

L_OPD = Σ_{v∈S_i} max(-A_v*ρ_v, -A_v*clip(ρ_v, 1-ε_lo, 1+ε_hi))
```

**关键参数**:
- ε_lo = 0.2, ε_hi = 0.28 (asymmetric PPO clipping)
- C = 1 (personal agent) 或 C = 2 (general agent)
- S_i = S_i^q (student top-k) by default

---

## 二、论文实验设置

### 2.1 Personal Agent Setup（与我们最相关）

| 参数 | 论文设置 | 我们的设置 | 差异 |
|---|---|---|---|
| Model | Qwen3-4B-Thinking-2507 | Qwen3-1.7B | 我们用更小的模型 |
| PRM | Qwen3-4B-Thinking-2507 | Qwen3-1.7B (same) | ✅ 一致 |
| Learning Rate | 1e-5 | 1e-5 | ✅ 一致 |
| KL Coefficient | 1.0 | 0.0 |  我们禁用 KL |
| Clip Ratios | 0.2 / 0.28 | 0.2 / 0.28 | ✅ 一致 |
| **Batch Size** | **16 samples** | **4 samples** | ❌ 我们太小 |
| **Training Trigger** | **每 16 个样本** | **每 4 个样本** | ❌ 我们太频繁 |
| Max Response Length | 8192 | 2048 | ❌ 我们太短 |
| Context Length | 16384 | 16384 | ✅ 一致 |
| Temperature | 1.0 (sampling) | 0.6 (PRM) | 用途不同 |

### 2.2 关键发现

**Table 3**: Hybrid RL 在 joint optimization 下平均只需 **10.3 sessions** 达到优化效果，而 GRPO 单独需要 14.1 sessions。

**Figure 2**: 优化前后对比显示模型明显向用户偏好偏移：
- Student: 从 AI-like 格式 → 更自然的解答
- TA: 从太简短 → 更详细具体
- Teacher: 从冷淡 → 更友好耐心

**Figure 5**: 在 terminal/GUI/SWE/tool-call 场景下，Hybrid RL 都能稳定提升。

**Figure 7**: Token-level 选择比 sequence-level 更稳定，Random hint selection 会导致训练不稳定。

---

## 三、我们的方案需要更新的地方

### 3.1 紧急修复（影响训练效果）

#### ❌ 问题 1: Batch Size 太小
- **当前**: `--rollout-batch-size 4`
- **论文**: 16 samples per training trigger
- **影响**: 4 个样本的 GRPO advantage 估计方差太大，训练不稳定
- **修复**: 改为 `--rollout-batch-size 16`

#### ❌ 问题 2: 训练触发太频繁
- **当前**: 每 4 个样本触发一次训练
- **论文**: 每 16 个样本触发一次
- **影响**: 频繁训练导致 policy 还没探索够就更新，样本利用率低
- **修复**: 等待 16 个样本后再触发

#### ❌ 问题 3: 没有 Directive Signal
- **当前**: 只使用 PRM 的 evaluative score (+1/-1/0)
- **论文**: PRM 还提取 [HINT_START]...[HINT_END] 作为 directive signal
- **影响**: 损失了最丰富的训练信号（token 级指导）
- **修复**: 实现完整的 OPD pipeline

#### ❌ 问题 4: 没有 Overlap-Guided Hint Selection
- **当前**: 无
- **论文**: 生成 M=3 个 hints，选择与学生 top-k 重叠最大的
- **影响**: 随机选择 hint 会导致训练不稳定（Figure 7）
- **修复**: 实现 overlap-guided selection

#### ❌ 问题 5: Response Length 太短
- **当前**: `--rollout-max-response-len 2048`
- **论文**: 8192 tokens
- **影响**: 模型无法生成长回答，限制表达能力
- **修复**: 改为 8192（需要更多 GPU 内存）

### 3.2 架构改进

#### ✅ 需要新增组件

```
当前架构:
  OpenClaw → API Server → Router → Adapter → vLLM
                                    ↓
                                  PRM (scoring only)

论文架构:
  OpenClaw → API Server → Router → Adapter → vLLM
                                    ↓
                            PRM Server (separate)
                              ├─ Evaluative scoring
                              └─ Directive hint extraction
                                    ↓
                            Training Engine
                              ├─ GRPO loss (evaluative)
                              └─ OPD loss (directive)
```

**关键区别**: 论文的 PRM 是独立 server，异步运行，不阻塞推理。

### 3.3 实现优先级

**Phase 1: 基础修复（1-2 天）**
1. 增加 batch size 到 16
2. 修改训练触发逻辑（每 16 样本）
3. 增加 response length 到 8192

**Phase 2: Directive Signal（3-5 天）**
4. 修改 PRM prompt 同时输出 score + hint
5. 解析 [HINT_START]...[HINT_END]
6. 实现 teacher distribution 计算

**Phase 3: Hybrid RL（5-7 天）**
7. 实现 overlap-guided hint selection
8. 实现 OPD loss 计算
9. 实现 hybrid objective 训练

**Phase 4: 优化（7-10 天）**
10. Token-level vs sequence-level ablation
11. Log-prob-difference clip 实现
12. 多用户 joint optimization

---

## 四、更新后的完整方案

### 4.1 训练流程（Algorithm 1 from paper）

```
Algorithm: OpenClaw-RL Online Hybrid Optimization Pipeline

Given: 
  - deployed clients
  - policy π_θ served by inference API
  - PRM server
  - hint budget M=3
  - top-k width K=4
  - clipping constants C=1, ε_lo=0.2, ε_hi=0.28
  - weights w_RL=1, w_OPD=1

Initialize: buffer B = 

while policy server is running:
  1. Client queries π_θ with state s_t
  2. Receives action a_t ~ π_θ(·|s_t)
  3. Executes a_t, streams next state s_{t+1} to RL server
  
  4. PRM asynchronously extracts:
     - majority vote r_t ∈ {+1, -1, 0}
     - candidate hints H_t = {h_{t,k}}_{k=1}^M (if s_{t+1} contains correction)
  
  5. Add to buffer B:
     - hybrid if H_t ≠ ∅ and r_t ∈ {±1}
     - directive-only if H_t ≠ ∅
     - evaluative-only if r_t ∈ {±1}
  
  6. if |B| ≥ 16 (batch size):
     a. Freeze rollout policy as π_old
     b. Compute GRPO advantage A^grpo
     c. For each sample in B:
        - Compute student support S_i^q = top-K{π_old(·|s_i, y_<i)}
        - For each hint h in H_t:
          · Set s_i^h = s_t ⊕ h
          · Compute S_{i,h}^p = top-K{π_T(·|s_i^h)}
          · Compute overlap O[h,i] = |S_i^q ∩ S_{i,h}^p|
        - Select h* = argmax_h Σ_i O[h,i]
        - Compute OPD loss with clipped log-prob difference
     d. Compute hybrid loss: L = w_RL * L_GRPO + w_OPD * L_OPD
     e. Update θ, push weights to serving engine
     f. Clear buffer B
```

### 4.2 PRM Prompt 设计

#### Evaluative Signal Prompt（已有，需微调）
```
System: You are a process reward model (PRM) evaluating an AI assistant.
Task: Decide whether the assistant's output successfully fulfilled the user's intent.

Scoring:
- \boxed{1} (good): Next state shows task progressed as expected
- \boxed{-1} (bad): Next state signals wrong/incomplete/unwanted output
- \boxed{0} (neutral): Ambiguous next state

Input:
- Assistant response at turn t
- Next state at turn t+1 (with role)

Output: \boxed{score}
```

#### Directive Signal Prompt（新增）
```
System: You are a process reward model used for hindsight hint extraction.
Task: Decide whether the next state reveals useful hindsight information.

Output format (strict):
- MUST include exactly one decision: \boxed{1} or \boxed{-1}
- If \boxed{1}: provide hint wrapped in [HINT_START]...[HINT_END]
- If \boxed{-1}: do NOT provide hint block

Hint rules:
1. ONE short sentence inside tags
2. Must NOT quote/paraphrase assistant's response
3. Must NOT reproduce next state content
4. Must NOT contain step-by-step reasoning
5. Must NOT include specific numbers/code from next state
6. Focus on SINGLE conceptual nudge/missing step
7. Keep short, abstract, content-free

Input:
- Assistant response at turn t
- Next state at turn t+1 (with role)

Output: \boxed{decision}
[HINT_START]hint text[HINT_END] (if decision=1)
```

### 4.3 更新后的训练脚本参数

```bash
python3 train_async.py \
  --train-backend fsdp \
  --attn-implementation sdpa \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --rollout-num-gpus 0 \
  --num-gpus-per-node 1 \
  --hf-checkpoint ${MODEL_PATH} \
  --ref-load ${MODEL_PATH} \
  --save ${CKPT_PATH} \
  --save-interval 10 \
  --disable-rollout-global-dataset \
  --rollout-function-path openclaw_rollout.generate_rollout_openclaw \
  --num-rollout 100000000 \
  --rollout-batch-size 16 \              # ← 改为 16
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 8192 \      # ← 改为 8192
  --rollout-max-context-len 16384 \
  --rollout-temperature 1.0 \            # ← 改为 1.0 (paper uses 1.0 for sampling)
  --reward-key score \
  --num-steps-per-rollout 1 \
  --use-dynamic-batch-size \
  --max-tokens-per-gpu 8192 \            # ← 改为 8192
  --gradient-checkpointing \
  --advantage-estimator grpo \
  --disable-rewards-normalization \
  --use-kl-loss \
  --kl-loss-coef 1.0 \                   # ← 改为 1.0 (paper uses β_KL=0.01, but we use coefficient)
  --kl-loss-type low_var_kl \
  --entropy-coef 0.0 \
  --eps-clip 0.2 \
  --eps-clip-high 0.28 \
  --optimizer adam \
  --lr 1e-5 \
  --lr-decay-style constant \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --use-lora \
  --lora-rank 32 \
  --lora-alpha 64 \
  --lora-target-modules "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj" \
  --rollout-external \
  --rollout-external-engine-addrs "127.0.0.1:30001" \
  --rollout-num-gpus-per-engine 0 \
  --prm-enable \
  --prm-num-gpus 0 \
  --prm-router-ip 127.0.0.1 \
  --prm-router-port 30001 \
  --prm-m 3 \                            # ← M=3 hints per sample
  --prm-temperature 0.6 \
  --prm-max-new-tokens 256 \
  --sglang-mem-fraction-static 0.0 \
  --sglang-context-length 16384 \
  --sglang-reasoning-parser qwen3 \
  --custom-generate-function-path openclaw_api_server.generate \
  --custom-rm-path openclaw_api_server.reward_func \
  --hybrid-rl-enable \                   # ← 新增：启用 Hybrid RL
  --hybrid-w-rl 1.0 \                    # ← 新增：w_RL
  --hybrid-w-opd 1.0 \                   # ← 新增：w_OPD
  --opd-k 4 \                            # ← 新增：top-k width
  --opd-clip-c 1.0 \                     # ← 新增：log-prob clip constant
  --opd-hint-budget 3                    # ← 新增：M hints per sample
```

### 4.4 需要的代码修改

#### 4.4.1 `openclaw_api_server.py` 修改

**当前**: PRM 只返回 score
**需要**: PRM 同时返回 score + hint

```python
# 当前实现
async def _query_prm_once(self, judge_prompt, vote_id):
    # ... calls PRM endpoint ...
    return _parse_prm_score(str(raw)), str(raw)

# 需要改为
async def _query_prm_once(self, judge_prompt, vote_id):
    # ... calls PRM endpoint ...
    score = _parse_prm_score(str(raw))
    hint = _parse_hint(str(raw))  # 新增：解析 [HINT_START]...[HINT_END]
    return score, hint, str(raw)
```

#### 4.4.2 新增 `overlap_guided_selection.py`

```python
def compute_overlap(student_top_k, teacher_top_k):
    """计算两个 top-k vocab 的重叠度"""
    return len(set(student_top_k) & set(teacher_top_k))

def select_best_hint(hints, student_response, teacher_model):
    """Overlap-guided hint selection"""
    student_top_k = get_top_k_tokens(student_response, k=4)
    
    best_hint = None
    best_overlap = -1
    
    for hint in hints:
        augmented_prompt = student_response + hint
        teacher_top_k = teacher_model.get_top_k(augmented_prompt, k=4)
        overlap = compute_overlap(student_top_k, teacher_top_k)
        
        if overlap > best_overlap:
            best_overlap = overlap
            best_hint = hint
    
    return best_hint, best_overlap
```

#### 4.4.3 新增 `opd_loss.py`

```python
def compute_opd_loss(
    student_logits, teacher_logits, 
    hint, support_set, clip_c=1.0,
    eps_lo=0.2, eps_hi=0.28
):
    """Top-k OPD loss with log-prob-difference clip"""
    log_prob_old = F.log_softmax(student_logits, dim=-1)
    log_prob_teacher = F.log_softmax(teacher_logits, dim=-1)
    
    # Compute advantage for each token in support set
    advantages = []
    for v in support_set:
        delta_v = torch.clamp(
            log_prob_teacher[v] - log_prob_old[v],
            -clip_c, clip_c
        )
        w_v = torch.softmax(log_prob_old[support_set], dim=0)[v]
        A_v = delta_v * w_v
        advantages.append(A_v)
    
    # Compute clipped surrogate loss
    rho = torch.exp(log_prob_old - log_prob_old.detach())
    clipped_rho = torch.clamp(rho, 1 - eps_lo, 1 + eps_hi)
    
    loss = torch.max(-A * rho, -A * clipped_rho).sum()
    return loss
```

---

## 五、实施计划

### Phase 1: 基础参数调整（今天）
- [x] 删除合成数据的 checkpoints
- [ ] 修改 batch size 为 16
- [ ] 修改训练触发逻辑（每 16 样本）
- [ ] 修改 response length 为 8192
- [ ] 重启训练

### Phase 2: Directive Signal 实现（1-2 天）
- [ ] 修改 PRM prompt 输出 hint
- [ ] 实现 hint 解析逻辑
- [ ] 测试 PRM 同时输出 score + hint

### Phase 3: Hybrid RL 实现（3-5 天）
- [ ] 实现 overlap-guided hint selection
- [ ] 实现 OPD loss 计算
- [ ] 实现 hybrid objective
- [ ] 集成到训练循环

### Phase 4: 真实对话测试（5-7 天）
- [ ] 使用真实 OpenClaw 对话
- [ ] 监控反馈分布
- [ ] 对比 Hybrid vs GRPO-only

---

## 六、预期改进

基于论文结果，实施完整方案后预期：

1. **训练效率**: 从 14.1 sessions → 10.3 sessions (27% 提升)
2. **训练稳定性**: overlap-guided selection 避免随机 hint 的不稳定
3. **模型质量**: hybrid objective 同时利用评估和指导信号
4. **泛化能力**: 更好的跨用户、跨场景泛化

---

## 七、当前状态

- ✅ 基础架构搭建完成（vLLM + Adapter + Slime）
- ✅ GRPO-only 训练验证通过
- ✅ PRM evaluative scoring 工作正常
- ✅ Checkpoint 保存/恢复机制验证
- ❌ Directive signal 未实现
- ❌ Overlap-guided selection 未实现
- ❌ OPD loss 未实现
- ❌ Hybrid objective 未实现

**下一步**: 按 Phase 1-4 逐步实施论文中的完整方法。
