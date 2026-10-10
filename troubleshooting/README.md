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
