#!/bin/bash
# Phase 3: On-Policy Distillation (OPD) - 纯配置实现
# 使用 FSDP backend + ref_model 作为 teacher

set -e

echo "=========================================="
echo "Phase 3: On-Policy Distillation (OPD)"
echo "=========================================="

# 配置参数
MODEL_PATH="/home/wangwei/models/Qwen3-1.7B"
SAVE_CKPT="/home/wangwei/wei/workspacePy/agentic-rl/OpenClaw-RL-ext/poc/checkpoints/opd_run"
BATCH_SIZE=16
MAX_RESPONSE_LEN=4096

echo "模型路径: $MODEL_PATH"
echo "保存路径: $SAVE_CKPT"
echo "Batch size: $BATCH_SIZE"
echo "最大响应长度: $MAX_RESPONSE_LEN"
echo ""

# 创建保存目录
mkdir -p "$SAVE_CKPT"

# 启动训练（使用 on_policy_distillation）
echo "启动 OPD 训练..."
echo ""

python train_async.py \
    --train-backend fsdp \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 1 \
    --rollout-num-gpus 1 \
    --rollout-num-gpus-per-engine 1 \
    --num-gpus-per-node 1 \
    --hf-checkpoint "$MODEL_PATH" \
    --ref-load "$MODEL_PATH" \
    --save "$SAVE_CKPT" \
    --save-interval 10 \
    --rollout-batch-size $BATCH_SIZE \
    --n-samples-per-prompt 1 \
    --rollout-max-response-len $MAX_RESPONSE_LEN \
    --rollout-max-context-len 16384 \
    --rollout-temperature 1.0 \
    --advantage-estimator on_policy_distillation \
    --disable-rollout-global-dataset \
    --use-kl-loss \
    --kl-loss-coef 0.01 \
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
    --sglang-mem-fraction-static 0.5 \
    --sglang-context-length 16384 \
    --sglang-reasoning-parser qwen3 \
    --custom-generate-function-path openclaw_api_server.generate \
    --custom-rm-path openclaw_api_server.reward_func \
    --prm-enable \
    --prm-num-gpus 0 \
    --prm-router-ip 127.0.0.1 \
    --prm-router-port 30001 \
    --prm-m 3 \
    --prm-temperature 0.6 \
    --prm-max-new-tokens 256

echo ""
echo "=========================================="
echo "OPD 训练完成！"
echo "=========================================="
