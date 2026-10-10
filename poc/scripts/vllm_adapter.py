#!/usr/bin/env python3
"""
vLLM 兼容适配层 — 让 Slime 的 external rollout 模式对接 vLLM

提供 Slime 需要的两个端点:
  - /health_generate  → 转发到 vLLM 的 /health
  - /get_server_info  → 返回 Slime 需要的配置信息

同时代理 /v1/chat/completions 和 /generate 到 vLLM

启动方式:
  python vllm_adapter.py --vllm-url http://localhost:30000 --port 30001
"""

import argparse
import asyncio
import json
import logging
import sys

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vllm_adapter")

app = FastAPI()

VLLM_URL = ""
VLLM_MODEL_ID = ""  # vLLM 注册的模型名


@app.get("/health")
@app.get("/health_generate")
async def health():
    """Slime 检查 /health_generate, vLLM 只有 /health — 两个都支持"""
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            r = await client.get(f"{VLLM_URL}/health")
            if r.status_code == 200:
                return {"status": "ok"}
    except Exception as e:
        logger.warning(f"vLLM health check failed: {e}")
    return JSONResponse(status_code=503, content={"status": "unhealthy"})


@app.post("/flush_cache")
@app.get("/flush_cache")
async def flush_cache():
    """Slime router 需要的 flush_cache 端点"""
    return {"status": "ok"}


@app.post("/init_weights_update_group")
async def init_weights_update_group(request: Request):
    """Slime 权重同步初始化 — vLLM 不支持在线权重更新, 返回 ok 跳过"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    logger.info(f"init_weights_update_group called (no-op for vLLM)")
    return {"status": "ok"}


@app.post("/update_weights_from_distributed")
async def update_weights_from_distributed(request: Request):
    """Slime 权重同步 — vLLM 不支持在线权重更新, 返回 ok 跳过"""
    logger.info(f"update_weights_from_distributed called (no-op for vLLM)")
    return {"status": "ok", "num_updated": 0}


@app.post("/release_weights_update_group")
async def release_weights_update_group(request: Request):
    """Slime 权重同步释放 — no-op"""
    return {"status": "ok"}


@app.get("/get_server_info")
async def get_server_info():
    """返回 Slime 需要的 server info 格式"""
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            r = await client.get(f"{VLLM_URL}/v1/models", timeout=10)
            if r.status_code == 200:
                models = r.json().get("data", [])
                model_id = models[0]["id"] if models else "unknown"
                return {
                    "model_path": model_id,
                    "tokenizer_path": model_id,
                    "host": "0.0.0.0",
                    "port": int(VLLM_URL.split(":")[-1].rstrip("/")),
                    "tp_size": 1,
                    "dp_size": 1,
                    "chunked_prefill_size": 4096,
                }
    except Exception as e:
        logger.warning(f"Failed to get server info: {e}")
    return JSONResponse(status_code=503, content={})


@app.api_route("/v1/chat/completions", methods=["POST"])
async def chat_completions(request: Request):
    """转发 /v1/chat/completions 到 vLLM，自动修正 model 名称"""
    body = await request.json()
    # 将任意 model 名重写为 vLLM 实际注册的模型 ID
    if VLLM_MODEL_ID and body.get("model"):
        body["model"] = VLLM_MODEL_ID
    try:
        async with httpx.AsyncClient(timeout=300, trust_env=False) as client:
            r = await client.post(f"{VLLM_URL}/v1/chat/completions", json=body)
            try:
                return JSONResponse(status_code=r.status_code, content=r.json())
            except Exception:
                return JSONResponse(status_code=r.status_code, content={"error": r.text[:500]})
    except Exception as e:
        logger.error(f"chat completions error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.api_route("/generate", methods=["POST"])
async def generate(request: Request):
    """转发 /generate 到 vLLM (转换为 vLLM 格式)"""
    body = await request.json()
    # sglang /generate 格式 → vLLM /v1/completions 格式
    prompt = body.get("text", body.get("prompt", ""))
    sampling_params = body.get("sampling_params", {})

    vllm_body = {
        "model": body.get("model", ""),
        "prompt": prompt,
        "max_tokens": sampling_params.get("max_new_tokens", 256),
        "temperature": sampling_params.get("temperature", 1.0),
        "top_p": sampling_params.get("top_p", 1.0),
        "logprobs": sampling_params.get("logprobs", None),
    }

    try:
        async with httpx.AsyncClient(timeout=120, trust_env=False) as client:
            r = await client.post(f"{VLLM_URL}/v1/completions", json=vllm_body)
            result = r.json()
            # 转换回 sglang /generate 格式
            choices = result.get("choices", [{}])
            text = choices[0].get("text", "") if choices else ""
            logprobs_data = None
            if "logprobs" in choices[0] and choices[0]["logprobs"]:
                lp = choices[0]["logprobs"]
                logprobs_data = {
                    "input_token_logprobs": None,
                    "output_token_logprobs": lp.get("token_logprobs", []),
                    "output_tokens": lp.get("tokens", []),
                }
            return JSONResponse(content={
                "text": text,
                "output_ids": [],
                "meta_info": {
                    "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
                    "cached_tokens": 0,
                },
                "output_token_logprobs": logprobs_data["output_token_logprobs"] if logprobs_data else None,
                "output_tokens": logprobs_data["output_tokens"] if logprobs_data else None,
            })
    except Exception as e:
        logger.error(f"generate error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", required=True, help="vLLM server URL (e.g. http://localhost:30000)")
    parser.add_argument("--port", type=int, default=30001, help="Adapter server port")
    parser.add_argument("--host", default="0.0.0.0", help="Adapter bind host")
    args = parser.parse_args()

    VLLM_URL = args.vllm_url.rstrip("/")

    # 自动检测 vLLM 注册的模型名
    try:
        import httpx as _httpx
        _r = _httpx.get(f"{VLLM_URL}/v1/models", timeout=10, proxies={})
        _models = _r.json().get("data", [])
        if _models:
            VLLM_MODEL_ID = _models[0]["id"]
            logger.info(f"vLLM model detected: {VLLM_MODEL_ID}")
    except Exception as e:
        logger.warning(f"Could not detect vLLM model: {e}")

    logger.info(f"vLLM adapter: proxy {args.host}:{args.port} -> {VLLM_URL}")
    if VLLM_MODEL_ID:
        logger.info(f"Model rewrite: any model name -> '{VLLM_MODEL_ID}'")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
