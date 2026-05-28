"""
Uncensored LLM API on Modal — Dolphin-Mixtral 8x7B (AWQ)
=========================================================
Deploy : modal deploy modal_app.py
Download weights first:
         modal run modal_app.py         (calls the local_entrypoint)
Endpoint: POST <url>/chat
          GET  <url>/health
"""

import json
import os
import uuid
import modal
from typing import List, Optional

# ── Configuration ──────────────────────────────────────────────────────────────
#
#  Default: AWQ 4-bit model on a single A10G (~$1.10/hr on Modal).
#  The AWQ model is ~14 GB on disk and loads in < 2 min.
#
MODEL_ID = "TheBloke/dolphin-2.5-mixtral-8x7b-AWQ"
QUANTIZATION = "awq"
DTYPE = "float16"
TENSOR_PARALLEL = 1
MAX_MODEL_LEN = 8192
VOLUME_NAME = "dolphin-mixtral-vol"
MODEL_DIR = "/vol"

GPU_CONFIG = "A10G"   # Modal 1.x uses plain strings for GPU specs

# ── Uncomment for full BF16 quality on 2× A100 80 GB (~$7.34/hr) ──────────────
# MODEL_ID      = "cognitolabs/dolphin-2.5-mixtral-8x7b"
# QUANTIZATION  = None
# DTYPE         = "bfloat16"
# TENSOR_PARALLEL = 2
# GPU_CONFIG    = "A100:80GB:2"   # "<type>:<memory>:<count>" in Modal 1.x
# ──────────────────────────────────────────────────────────────────────────────

# ── Container image ────────────────────────────────────────────────────────────
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm>=0.6.0,<0.7.0",     # 0.6.x ships torch 2.4+ (0.4.3 had torch 2.3)
        "fastapi>=0.110.0",
        "uvicorn[standard]>=0.29.0",
        "huggingface-hub>=0.22.0",
        "hf-xet>=0.1.0",          # replaces deprecated hf-transfer
        "transformers>=4.40.0,<4.44.0",  # 4.44 broke LlamaTokenizer API used by vllm 0.6.x
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # replaces deprecated HF_HUB_ENABLE_HF_TRANSFER
)

app = modal.App("dolphin-mixtral-api", image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Stable path derived from model ID (slashes → double-dashes)
_model_slug = MODEL_ID.replace("/", "--")
_model_path = os.path.join(MODEL_DIR, _model_slug)


# ── Weight download (run once, cached on the Volume) ──────────────────────────
@app.function(volumes={_model_path: vol}, timeout=3600, image=image)
def download_model():
    from huggingface_hub import snapshot_download

    marker = os.path.join(_model_path, "config.json")
    if os.path.exists(marker):
        print(f"Weights already cached at {_model_path}")
        return

    print(f"Downloading {MODEL_ID} → {_model_path} …")
    snapshot_download(
        MODEL_ID,
        local_dir=_model_path,
        # Prefer safetensors; skip redundant pytorch/pickle blobs when present
        ignore_patterns=["*.pt", "*.bin"],
    )
    vol.commit()
    print("Download complete and committed to volume.")


@app.local_entrypoint()
def main():
    """Run `modal run modal_app.py` to pre-download weights before deploying."""
    download_model.remote()
    print("Weights ready. Deploy with: modal deploy modal_app.py")


# ── FastAPI + vLLM inference server ───────────────────────────────────────────
@app.function(
    gpu=GPU_CONFIG,
    volumes={_model_path: vol},
    timeout=600,           # max request lifetime (seconds)
    scaledown_window=300,  # renamed from container_idle_timeout in Modal 1.x
    image=image,
)
@modal.concurrent(max_inputs=64)  # allow_concurrent_inputs renamed in Modal 1.x
@modal.asgi_app()
def serve():  # noqa: C901  (complexity is acceptable for a self-contained server)
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, StreamingResponse
    from pydantic import BaseModel
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from transformers import AutoTokenizer

    # ── Engine init (once per cold start) ─────────────────────────────────────
    kwargs = {}
    if QUANTIZATION:
        kwargs["quantization"] = QUANTIZATION

    engine_args = AsyncEngineArgs(
        model=_model_path,
        tensor_parallel_size=TENSOR_PARALLEL,
        gpu_memory_utilization=0.92,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        trust_remote_code=True,
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(_model_path, trust_remote_code=True)

    web = FastAPI(title="Dolphin Uncensored API", version="1.0")

    # ── Request / response schemas ─────────────────────────────────────────────
    class Message(BaseModel):
        role: str
        content: str

    class ChatRequest(BaseModel):
        messages: List[Message]
        stream: bool = False
        max_tokens: int = 1024
        temperature: float = 0.7
        top_p: float = 0.9
        stop: Optional[List[str]] = None

    # ── Helpers ────────────────────────────────────────────────────────────────
    def build_prompt(messages: List[Message]) -> str:
        """Apply the model's native chat template (ChatML for Dolphin)."""
        return tokenizer.apply_chat_template(
            [m.model_dump() for m in messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    async def sse_stream(prompt: str, params: SamplingParams, req_id: str):
        """Yield Server-Sent Events with OpenAI-style delta chunks."""
        sent = 0
        async for out in engine.generate(prompt, params, req_id):
            if not out.outputs:
                continue
            full = out.outputs[0].text
            delta = full[sent:]
            sent = len(full)
            if delta:
                chunk = {
                    "id": f"chatcmpl-{req_id}",
                    "object": "chat.completion.chunk",
                    "model": MODEL_ID,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk)}\n\n"

        # Final chunk signals end of stream
        yield (
            f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        )
        yield "data: [DONE]\n\n"

    # ── Routes ─────────────────────────────────────────────────────────────────
    @web.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_ID}

    @web.post("/chat")
    async def chat(req: ChatRequest):
        try:
            prompt = build_prompt(req.messages)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Chat template error: {exc}")

        params = SamplingParams(
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
            stop=req.stop or [],
        )
        req_id = str(uuid.uuid4())

        if req.stream:
            return StreamingResponse(
                sse_stream(prompt, params, req_id),
                media_type="text/event-stream",
            )

        # Non-streaming: accumulate and return
        full_text = ""
        async for out in engine.generate(prompt, params, req_id):
            if out.outputs:
                full_text = out.outputs[0].text

        return JSONResponse(
            {
                "id": f"chatcmpl-{req_id}",
                "object": "chat.completion",
                "model": MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": full_text},
                        "finish_reason": "stop",
                    }
                ],
            }
        )

    # OpenAI-compatible alias so drop-in clients work unchanged
    @web.post("/v1/chat/completions")
    async def chat_v1(req: ChatRequest):
        return await chat(req)

    return web
