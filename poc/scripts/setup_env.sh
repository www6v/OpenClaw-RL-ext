#!/bin/bash
# =============================================================================
# OpenClaw-RL POC 环境准备脚本
# =============================================================================
# 在单卡 RTX 3090 上准备 Qwen3-1.7B + LoRA 训练环境
# =============================================================================

set -ex

echo "=============================================="
echo "  OpenClaw-RL POC 环境准备"
echo "  Qwen3-1.7B + LoRA + 单卡 RTX 3090"
echo "=============================================="

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
POC_ROOT="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
OPENCLAW_RL_ROOT="$(cd -- "${POC_ROOT}/../.." &>/dev/null && pwd)"
REPO_ROOT="${OPENCLAW_RL_ROOT}/OpenClaw-RL"
SLIME_ROOT="${REPO_ROOT}/slime"

# ---- Step 0: 检查 GPU ----
echo ""
echo "[Step 0] 检查 GPU 和 CUDA..."
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "${GPU_MEM}" -lt 24000 ]; then
    echo "⚠️  警告: GPU 显存 ${GPU_MEM}MB < 24000MB, 可能不够"
fi

# 检查 CUDA 版本 (需要 12.x 兼容 sglang)
CUDA_VER=$(nvidia-smi | grep "CUDA Version" | grep -oP '\d+\.\d+' | head -1)
echo "  CUDA Driver 版本: ${CUDA_VER}"
echo "  注意: OpenClaw-RL 的 sglang/slime 需要 CUDA 12.9 toolkit"
echo "        Driver ${CUDA_VER} 向下兼容, 只需安装 CUDA 12.9 toolkit 即可"

# ---- Step 1: 检查 conda 环境 ----
echo ""
echo "[Step 1] 检查 Python 环境..."
if ! command -v python3 &>/dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi
python3 --version

# ---- Step 2: 检查/安装依赖 ----
echo ""
echo "[Step 2] 检查 Python 依赖..."
REQUIRED_PACKAGES="torch ray sglang transformers"
for pkg in ${REQUIRED_PACKAGES}; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        echo "  ✅ ${pkg} 已安装"
    else
        echo "  ❌ ${pkg} 未安装 — 请先运行 OpenClaw-RL/instructions/README.md 中的环境搭建步骤"
        echo "     或者: cd ${REPO_ROOT} && pip install -r requirements.txt"
        exit 1
    fi
done

# ---- Step 3: 检查 Slime ----
echo ""
echo "[Step 3] 检查 Slime 框架..."
if [ -d "${SLIME_ROOT}" ] && [ -f "${SLIME_ROOT}/train_async.py" ]; then
    echo "  ✅ Slime 框架存在"
else
    echo "  ❌ Slime 框架不存在于 ${SLIME_ROOT}"
    echo "     请确保 OpenClaw-RL/slime/ 目录存在"
    exit 1
fi

# 检查 Qwen3-1.7B 模型配置
if [ -f "${SLIME_ROOT}/scripts/models/qwen3-1.7B.sh" ]; then
    echo "  ✅ Qwen3-1.7B 模型配置存在"
else
    echo "  ❌ Qwen3-1.7B 模型配置不存在"
    exit 1
fi

# ---- Step 4: 下载 Qwen3-1.7B 模型 ----
echo ""
echo "[Step 4] 检查/下载 Qwen3-1.7B 模型..."
# 检查多个可能的位置
MODEL_DIR=""
for candidate in \
    "/data/models/models/Qwen3-1.7B" \
    "${HOME}/models/Qwen3-1.7B" \
    "/data/models/models/Qwen3-1.7B/snapshots/master"; do
    if [ -f "${candidate}/config.json" ]; then
        MODEL_DIR="${candidate}"
        break
    fi
done

if [ -n "${MODEL_DIR}" ]; then
    echo "  ✅ 模型已存在: ${MODEL_DIR}"
else
    echo "  📥 下载 Qwen3-1.7B..."
    mkdir -p /data/models/models 2>/dev/null || mkdir -p "${HOME}/models"
    DOWNLOAD_DIR="/data/models/models/Qwen3-1.7B"
    [ -w "/data/models/models" ] || DOWNLOAD_DIR="${HOME}/models/Qwen3-1.7B"

    # 优先用 modelscope (国内更快)
    python3 -c "
try:
    from modelscope import snapshot_download
    model_dir = snapshot_download('Qwen/Qwen3-1.7B', cache_dir='$(dirname ${DOWNLOAD_DIR})')
    print(f'下载完成: {model_dir}')
except ImportError:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id='Qwen/Qwen3-1.7B', local_dir='${DOWNLOAD_DIR}', local_dir_use_symlinks=False)
    print('下载完成!')
"
    MODEL_DIR="${DOWNLOAD_DIR}"
fi

# 确保 ~/models/Qwen3-1.7B 软链接存在
if [ ! -L "${HOME}/models/Qwen3-1.7B" ] && [ ! -d "${HOME}/models/Qwen3-1.7B" ]; then
    mkdir -p "${HOME}/models"
    ln -sfn "${MODEL_DIR}" "${HOME}/models/Qwen3-1.7B"
    echo "  软链接: ~/models/Qwen3-1.7B -> ${MODEL_DIR}"
fi

# ---- Step 5: 创建输出目录 ----
echo ""
echo "[Step 5] 创建输出目录..."
mkdir -p "${POC_ROOT}/ckpt"
mkdir -p "${POC_ROOT}/openclaw-rl/results"
mkdir -p "${POC_ROOT}/openclaw-combine/results"
echo "  ✅ 输出目录已创建"

# ---- Step 6: 检查原始模块引用 ----
echo ""
echo "[Step 6] 检查原始模块引用..."
for module_dir in openclaw-rl openclaw-combine openclaw-opd; do
    if [ -d "${REPO_ROOT}/${module_dir}" ]; then
        echo "  ✅ ${module_dir}/ 存在"
    else
        echo "  ❌ ${module_dir}/ 不存在于 ${REPO_ROOT}"
        exit 1
    fi
done

# ---- 完成 ----
echo ""
echo "=============================================="
echo "  ✅ 环境检查通过！"
echo "=============================================="
echo ""
echo "  模型路径:   ${MODEL_DIR}"
echo "  Slime:      ${SLIME_ROOT}"
echo "  检查点目录: ${POC_ROOT}/ckpt/"
echo ""
echo "  下一步 — 启动 Binary RL 训练:"
echo "  ─────────────────────────────"
echo "  cd ${SLIME_ROOT}"
echo "  bash ../OpenClaw-RL-ext/poc/openclaw-rl/run_qwen3_1.7b_openclaw_rl_1gpu.sh"
echo ""
echo "  或启动 Hybrid Combine 训练:"
echo "  ─────────────────────────────"
echo "  cd ${SLIME_ROOT}"
echo "  bash ../OpenClaw-RL-ext/poc/openclaw-combine/run_qwen3_1.7b_openclaw_combine_1gpu.sh"
echo ""
echo "  OpenClaw 配置 (124.221.28.203):"
echo "  见 ${POC_ROOT}/docs/openclaw_config.json"
echo "=============================================="
