# OpenClaw-RL POC 踩坑记录

> 环境: RTX 3090 (24GB) · CUDA Driver 580.178.04 · CUDA 13.0 · Python 3.13
> 目标: 单机跑通 Qwen3-1.7B + LoRA + OpenClaw RL 训练

---

## 1. Python / pip 不一致

**现象**: `pip list` 显示包已安装，但 `python3 -c "import xxx"` 报 ModuleNotFoundError。

**原因**: `pip` 在 `/home/wangwei/.local/bin/pip` (用户级)，安装到 Python 3.13 的 site-packages。但默认 `python3` 指向 `/usr/bin/python3` (Python 3.10)。

**解决**: 始终用 `/usr/bin/python3.13` 而非 `python3`。

```bash
# 检查
which python3       # → /usr/bin/python3 (3.10)
which pip           # → ~/.local/bin/pip (装到 3.13)
python3.13 --version  # → 3.13 ✅
```

---

## 2. /home 分区磁盘满了

**现象**: `pip install` 报 `OSError: [Errno 28] 设备上没有空间`

**原因**: `/home` 分区 59GB 被各种缓存塞满。

**解决**: 清理缓存释放 ~15GB。

```bash
rm -rf ~/.cache/pip          # 8.1GB
rm -rf ~/.cache/go-build     # 3.4GB
rm -rf ~/.cache/uv           # 1.9GB
rm -rf ~/.cache/pypoetry     # 1.1GB
rm -rf ~/.cache/mozilla      # 601MB
rm -rf ~/.cache/pnpm         # 455MB
```

---

## 3. sglang 装不上 — outlines_core 编译失败

**现象**: `pip install sglang` 报 `Failed building wheel for outlines_core`

**原因**: `outlines_core` 是 Rust 写的，需要 `libssl-dev` (OpenSSL 开发库)。系统只有 `libssl3` (运行时) 没有开发头文件。没有 sudo 权限装不了。

**解决**: 装 Rust 后从源码编译 outlines_core，或者跳过 — 我们最终没走通 sglang (见下一条)。

```bash
# 装 Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source $HOME/.cargo/env
```

---

## 4. sgl-kernel 不支持 RTX 3090 (SM86) ← 核心问题

**现象**: `import sgl_kernel` 报 `undefined symbol` 或 `Could not load any common_ops library`

**原因**: sgl-kernel 0.3.21 只为 SM90 (H100) 和 SM100 (Blackwell) 编译了 CUDA 内核。RTX 3090 是 SM86 (Ampere)，没有对应内核。`load_utils.py` 把所有非 SM90 的 GPU 都路由到 sm100 目录，但 sm100 内核用了更新的 PyTorch ABI。

**尝试的修复**:
1. 修改 `load_utils.py`，让 SM86 使用 sm90 内核 ✅ (符号能加载了)
2. 二进制 patch `.so` 修复 `c10_cuda_check_implementation` 参数类型不匹配 (`int` vs `unsigned int`) ✅
3. 创建 ASM shim 桥接 6 个 ABI 不兼容的符号 ✅
4. 但运行时 `sgl_kernel::rmsnorm` 算子无法在 CUDA 后端执行 ❌ — 算子注册了但 CUDA kernel 实现不兼容

**最终方案**: 放弃 sglang，改用 **vLLM 0.28.0** (原生支持 SM86)。

---

## 5. PyTorch / sgl-kernel ABI 不兼容 (详细)

**现象**: sgl-kernel `.so` 加载时报多个 `undefined symbol`

**不匹配的符号**:

| sgl-kernel 需要 | PyTorch 提供 | 差异 |
|---|---|---|
| `c10_cuda_check_implementation(..., int, bool)` | `c10_cuda_check_implementation(..., unsigned int, bool)` | 第 4 参数 `int` → `uint` |
| `TensorBase::const_data_ptr<int, 0>()` | `TensorBase::const_data_ptr<int>()` | 多了一个模板参数 |
| `getCurrentCUDABlasHandle()` | `getCurrentCUDABlasHandle(bool)` | 多了个 bool 参数 |
| `MessageLogger(const char*, int, int)` | `MessageLogger(SourceLocation, int, bool)` | 构造函数签名变了 |
| `materialize_cow_storage(StorageImpl&)` | `materialize_cow(StorageImpl*)` | 函数名和参数类型都变了 |
| `Library::_def(...vector<at::Tag>...)` | `Library::_def(...vector<torch::headeronly::Tag>...)` | Tag 类型命名空间变了 |

**修复 (ASM shim)**: 用汇编创建符号别名 + 参数转换:

```asm
# const_data_ptr<int, 0> → const_data_ptr<int>
.globl _ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v
_ZNK2at10TensorBase14const_data_ptrIiLi0EEEPKT_v:
    jmp _ZNK2at10TensorBase14const_data_ptrIiEEPKT_v

# getCurrentCUDABlasHandle() → getCurrentCUDABlasHandle(false)
.globl _ZN2at4cuda24getCurrentCUDABlasHandleEv
_ZN2at4cuda24getCurrentCUDABlasHandleEv:
    xor %esi, %esi   # false = 0
    jmp _ZN2at4cuda24getCurrentCUDABlasHandleEb
```

编译: `as -o shim.o shim.s && gcc -shared -o libsgl_compat.so shim.o -L.../torch/lib -lc10 -ltorch_cpu`

**结论**: 符号级修复能解决加载问题，但 CUDA kernel 实现的运行时不兼容无法通过 shim 解决。sglang 在 RTX 3090 上不可行。

---

## 6. transformers 版本地狱

**现象**: sglang 和 transformers 互相版本冲突

| sglang 需要 | transformers 需要 | 实际 |
|---|---|---|
| `tokenizers==0.22.2` | `tokenizers>=0.23.1` | 只能二选一 |
| `transformers==5.12.1` | `huggingface-hub<2.0` | huggingface-hub 升级后冲突 |
| `flashinfer>=0.6.18` | - | 0.6.16 太旧 |

**连锁反应**:
- 升级 transformers → huggingface-hub 版本冲突 → 降级 huggingface-hub
- 升级 flashinfer → 拉入 torch 2.14 → 与 sgl-kernel ABI 不兼容 → 重装 torch 2.13
- 装 torchvision → 版本与 torch 不匹配 → `torchvision::nms` 算子注册失败
- torchaudio → CUDA 13.0 vs cu129 不匹配 → 创建 mock 模块绕过

**解决**: 最终组合:
```
torch 2.13.0+cu129
transformers 5.19.0
tokenizers 0.23.3
flashinfer 0.6.18
huggingface-hub 1.33.0 (后降级到 0.36.2)
torchaudio → mock (不需要音频)
```

---

## 7. PreTrainedConfig 找不到

**现象**: sglang 启动报 `ImportError: cannot import name 'PreTrainedConfig' from 'transformers.configuration_utils'`

**原因**: transformers 5.x 把 `PreTrainedConfig` 改名为 `PretrainedConfig` (小写 t)，但 sglang 0.5.21 还用旧名。

**解决**: 在 `transformers/configuration_utils.py` 末尾加一行别名:
```python
PreTrainedConfig = PretrainedConfig
```

---

## 8. vLLM 的 torchaudio 问题

**现象**: sglang 启动报 `OSError: Could not load libtorchaudio.so`

**原因**: torchaudio 版本与 torch 不匹配 (CUDA 13.0 vs cu129)。

**解决**: 卸载 torchaudio，创建 mock 模块:
```bash
rm -rf ~/.local/lib/python3.13/site-packages/torchaudio*
mkdir -p ~/.local/lib/python3.13/site-packages/torchaudio
echo '__version__ = "0.0.0"' > ~/.local/lib/python3.13/site-packages/torchaudio/__init__.py
```

---

## 9. 本机服务公网不可达

**现象**: 124.221.28.203 无法访问本机的 30000-30002 端口

**原因**: 本机在 NAT 后面 (内网 IP 192.168.3.4)，公网 IPv4 (114.86.39.128) 没有端口转发。

**解决**: SSH 反向隧道:
```bash
# 从本机连到 124.221.28.203, 把远程 localhost:30000-30002 转发到本机
ssh -f -N -R 30001:localhost:30001 -R 30000:localhost:30000 root@124.221.28.203
```

---

## 10. OpenClaw 配置对接

**现象**: OpenClaw gateway 返回 "Invalid model"

**原因**: OpenClaw 用 agent 名调模型 (`"model": "openclaw"`)，不直接用 provider/model 格式。

**解决**: 在 `/root/.openclaw/openclaw.json` 中:
1. 添加 provider `openclaw-rl`，baseUrl 指向 `http://localhost:30001/v1` (SSH 隧道 → Adapter)
2. 修改 agent 的 `model.primary` 为 `openclaw-rl/qwen3-1.7b`
3. 重启 OpenClaw gateway

---

## 11. OpenClaw "Context overflow"

**现象**: OpenClaw 调用 vLLM 返回 "Context overflow: prompt too large for the model"

**原因**: OpenClaw gateway 会注入大量系统 prompt (agent 配置、工具定义、对话历史)，即使 vLLM 设了 32K context 也不够。

**现状**: 直接通过隧道的推理完全正常。Context overflow 是 OpenClaw gateway 层面的限制，需要进一步调整 OpenClaw 的 agent 配置 (减少工具/历史) 或使用更大 context 的模型。

---

## 最终可用组合

```
推理引擎:  vLLM 0.28.0 (替代 sglang, 原生支持 SM86)
模型:      Qwen3-1.7B + LoRA rank=32
训练框架:  Slime (FSDP 后端)
兼容层:    vllm_adapter.py (补 /health_generate, /get_server_info, 模型名重写)
网络:      SSH 反向隧道 (本机 → 124.221.28.203)
OpenClaw:  已配置 openclaw-rl provider, main agent 指向 vLLM
GPU:       1x RTX 3090 (24GB), vLLM 占 ~20GB (32K context)
```

## 12. OpenClaw "upstream provider timeout"

**现象**: OpenClaw gateway 调用 vLLM 返回 `upstream provider timeout`

**原因**: OpenClaw gateway 默认超时短，vLLM 首次请求需要 CUDA graph 编译较慢。

**解决**: 
1. vLLM 启动参数加 `--enable-auto-tool-choice --tool-call-parser hermes`
2. Adapter 超时从 120s 加到 300s
3. 首次请求后后续请求会快很多 (CUDA graph 已编译)

**最终验证**: OpenClaw(124.221.28.203) → SSH隧道 → Adapter(:30001) → vLLM(:30000) → Qwen3-1.7B ✅

---

## 总结：从问题到解决的链路

```
问题 1: Python 版本不一致
  → 统一用 /usr/bin/python3.13

问题 2: 磁盘满了
  → 清理 ~/.cache 释放 15GB

问题 3-6: sglang/sgl-kernel 不支持 RTX 3090
  → 放弃 sglang, 改用 vLLM

问题 7-8: transformers/torchaudio 版本冲突
  → 精确匹配版本 + mock torchaudio

问题 9: 本机服务公网不可达
  → SSH 反向隧道

问题 10-12: OpenClaw 配置 + context + timeout
  → 改 openclaw.json + 加大 context + 加 tool-call-parser
```

---

## Phase 2: Directive Signal 实现

### 问题 1: PRM 需要同时输出 score 和 hint

**现象**: 论文要求 PRM 输出 evaluative score + directive hint，但当前实现只输出 score。

**根因**: `_build_prm_judge_prompt` 只包含评分指令，缺少 hint 提取指令。

**解决方案**: 
- 创建 `_build_prm_dual_prompt` 包含双重任务指令
- 添加 `_parse_prm_dual_output` 解析 score + hint
- 修改输出格式要求: `\boxed{score}` + 可选 `[HINT_START]hint[HINT_END]`

**状态**: ✅ 测试通过，待集成到 openclaw_api_server.py

### 问题 2: Hint 质量需要严格控制

**现象**: 随机 hint 会导致训练不稳定（论文 Figure 7）。

**解决方案**: 
- Hint 必须是 ONE short sentence
- Hint 不能直接引用 next state 内容
- Hint 不能包含具体数字/代码
- 只在 score=-1 且存在有意义纠正时提取

**状态**: ✅ 已在 prompt 规则中体现

---

---

## Phase 2 实施：2026-10-10

### 问题 13: `args.prompt_data` is None TypeError

**现象**: RolloutManager 初始化时报错 `TypeError: expected string or bytes-like object, got 'NoneType'`

**原因**: 手动构造的训练命令缺少 `--disable-rollout-global-dataset` 参数。没有这个参数，Slime 会尝试加载 prompt 数据集，但 `args.prompt_data` 是 None。

**解决**: 确保训练命令包含 `--disable-rollout-global-dataset`。OpenClaw 的 rollout 从 API 请求收集数据，不需要静态数据集。

**状态**: ✅ 已修复

---

### 问题 14: `--rotary-base` 参数不识别

**现象**: 训练启动失败 `error: unrecognized arguments: --rotary-base 1000000`

**原因**: 当前版本的 Slime 不支持 `--rotary-base` 参数。这个参数可能是旧版本或 Megatron 后端的参数。

**解决**: 从训练命令中删除 `--rotary-base`。Qwen3 模型的 rotary embedding base 已经在模型配置中定义，不需要额外指定。

**状态**: ✅ 已修复

---

### 问题 15: Ray Dashboard 无法连接

**现象**: `ray job submit` 报 `ConnectionError: Failed to connect to Ray at address: http://127.0.0.1:8265`

**原因**: Ray Dashboard 进程启动了但没有绑定到 8265 端口。可能是代理干扰或启动延迟。

**解决**: 绕过 `ray job submit`，直接运行 `train_async.py`。脚本会自动连接到现有的 Ray 集群（通过 `RAY_ADDRESS` 环境变量）。

```bash
export RAY_ADDRESS="192.168.3.4:6380"
export RAY_AUTH_MODE=disabled
/usr/bin/python3.13 train_async.py ...
```

**状态**: ✅ 已修复

---

### 问题 16: SGLang Router 没有注册的 Worker（核心问题）

**现象**: API 请求返回 `503 Service Unavailable: No available workers (all circuits open or unhealthy)`

**原因**: 在 external mode 下，`SGLangEngine._init_external()` 方法只做了健康检查，但**没有向 Router 注册 worker**。Router 启动时 `worker_urls=[]`，没有可用的后端。

**根因**: 代码 bug — `_init_normal()` 有 worker 注册逻辑，但 `_init_external()` 遗漏了。

**解决**: 在 `slime/backends/sglang_utils/sglang_engine.py` 的 `_init_external()` 方法中添加 worker 注册：

```python
def _init_external(self, expect_server_args, external_engine_need_check_fields):
    # ... health check code ...
    
    # Register external engine with the router so it receives requests
    if self.node_rank == 0 and self.router_ip and self.router_port:
        try:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_slime_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                payload = {
                    "url": f"http://{self.server_host}:{self.server_port}",
                    "worker_type": self.worker_type,
                }
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()
            logger.info(f"External engine registered with router at {self.router_ip}:{self.router_port}")
        except Exception as e:
            logger.warning(f"Failed to register external engine with router: {e}")
```

**验证**: Router 现在显示已注册的 worker：
```json
{"workers":[{"url":"http://127.0.0.1:30001","is_healthy":true}],"total":1}
```

**状态**: ✅ 已修复（最小改动原则）

---

### 问题 17: 所有样本的 Reward 为 0

**现象**: 训练日志显示 `score=0.0 exclude=True`，`rollout/zero_std/count_0.0: 16`，`train/loss: 0.0`

**原因**: 训练命令没有启用 PRM（Process Reward Model）。没有 `--prm-enable` 参数，`openclaw_api_server.py` 的 `_prm_enabled` 为 False，PRM 评分被跳过，所有样本的 reward 默认为 0.0。

**影响**: GRPO 算法需要 reward 方差来计算 advantage。所有 reward 都是 0 时，advantage 为 0，loss 为 0，梯度为 0，模型不学习任何东西。

**解决**: 需要在训练命令中添加 PRM 参数：
```bash
--prm-enable \
--prm-num-gpus 0 \
--prm-router-ip 127.0.0.1 \
--prm-router-port 30001 \
--prm-m 3 \
--prm-temperature 0.6 \
--prm-max-new-tokens 256
```

**状态**: ❌ 待修复（需要重启训练）

---

### 当前状态总结

**已完成**:
- ✅ 训练循环完整运行
- ✅ 数据收集工作正常（16 samples per rollout）
- ✅ 完整的请求链路：Client → :30000 (API Server) → :3849 (Router) → :30001 (Adapter) → :30010 (vLLM)
- ✅ External engine 注册修复
- ✅ 所有启动错误已解决

**待完成**:
- ❌ 启用 PRM 获得真实的 reward 信号（+1/-1/0）
- ❌ Phase 2: Directive signal（hint 提取）
- ❌ Phase 3: Hybrid RL（overlap-guided selection + OPD loss）
- ❌ Phase 4: 真实对话测试

**下一步**: 重启训练并启用 PRM，然后逐步实施 Phase 2-4。


---

## Phase 2 关键修复：PRM 评分链路打通 (2026-10-11 凌晨)

### 问题 18: `_build_prm_judge_prompt` 未定义

**现象**: `NameError: name '_build_prm_judge_prompt' is not defined. Did you mean: '_build_prm_dual_prompt'?`

**原因**: 前一轮修改把函数从 `_build_prm_judge_prompt` 改名为 `_build_prm_dual_prompt`（支持双重输出），但 `_prm_evaluate` 中的调用没更新。

**修复**: `sed -i 's/_build_prm_judge_prompt/_build_prm_dual_prompt/g' openclaw_api_server.py`

**状态**: ✅ 已修复

---

### 问题 19: `_query_prm_once` 返回值数量不匹配

**现象**: `ValueError: too many values to unpack (expected 2)`

**原因**: `_query_prm_once` 被改为返回 3 个值 `(score, raw_text, hint)`，但：
- 类型注解还是 `tuple[int | None, str]`（2 个值）
- 错误分支 `return None, ""` 只返回 2 个值
- 调用处 `for s, text in results:` 按 2 个值解包

**修复** (3 处):
1. 类型注解: `tuple[int | None, str]` → `tuple[int | None, str, str | None]`
2. 错误分支: `return None, ""` → `return None, "", None`
3. 解包: `for s, text in results:` → `for s, text, _h in results:`

**状态**: ✅ 已修复

---

### 问题 20: OpenClaw 环境变量没传递到 RolloutManager

**现象**: PRM 显示 enabled，但 record file 没初始化，`_buffer_record` 不执行，PRM 永远不被触发。

**原因**: `RolloutManager` 是 Ray actor，运行在独立进程中。`placement_group.py` 的 `runtime_env` 没有传递 `OPENCLAW_RECORD_ENABLED`、`OPENCLAW_RECORD_FILE` 等变量。

**修复**: 在 `create_rollout_manager` 的 `runtime_env` 中添加:
```python
"OPENCLAW_RECORD_ENABLED": os.environ.get("OPENCLAW_RECORD_ENABLED", ""),
"OPENCLAW_RECORD_FILE": os.environ.get("OPENCLAW_RECORD_FILE", ""),
"OPENCLAW_PRM_RECORD_FILE": os.environ.get("OPENCLAW_PRM_RECORD_FILE", ""),
"SGLANG_API_KEY": os.environ.get("SGLANG_API_KEY", ""),
"SERVED_MODEL_NAME": os.environ.get("SERVED_MODEL_NAME", ""),
"HOST": os.environ.get("HOST", ""),
"PORT": os.environ.get("PORT", ""),
"PRM_M": os.environ.get("PRM_M", ""),
```

**状态**: ✅ 已修复

---

### 最终验证：PRM 评分 + 训练

**PRM 评分结果** (m=3, 多数投票):
```
PRM session=test-1 votes=[1, 1, 1] → score=1.0
PRM session=test-2 votes=[1, 1, 1] → score=1.0
PRM session=test-3 votes=[1, 1, 1] → score=1.0
```

**训练指标** (Step 0):
```
train/loss:      1.86   ← 真实 loss
train/pg_loss:  -0.44   ← GRPO 策略梯度（非零！）
train/kl_loss:   2.30   ← KL 约束
train/grad_norm: 8.16   ← 真实梯度
rollout/raw_reward: 0.4375  ← 平均 reward
rollout/advantages: 0.4375  ← 非零 advantage
zero_std/count_0.0: 9       ← 9 个 reward=0
zero_std/count_1.0: 7       ← 7 个 reward=+1
```

**结论**: 完整的 RL 训练链路已打通！
- ✅ 对话收集 → PRM 评分 → GRPO advantage → 模型更新
- ✅ 模型在真实学习（非零梯度、非零 policy gradient loss）

---

### 今天总共修改的文件 (4 个，最小改动原则)

| # | 文件 | 改动量 | 内容 |
|---|------|--------|------|
| 1 | `slime/backends/sglang_utils/sglang_engine.py` | +15 行 | `_init_external()` 添加 worker 注册 |
| 2 | `slime/ray/placement_group.py` | +8 行 | `runtime_env` 添加 OpenClaw 环境变量 |
| 3 | `openclaw-rl/openclaw_api_server.py` | ~5 行 | 函数名修复 + 返回值修复 |

**总计约 30 行代码改动**，其中大部分是环境变量传递。核心逻辑修复只有：
- Worker 注册（15 行）
- 函数名 `_build_prm_judge_prompt` → `_build_prm_dual_prompt`（1 行）
- 返回值从 2 个改为 3 个（3 行）
- 解包从 2 个改为 3 个（1 行）

