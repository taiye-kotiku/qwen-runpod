#!/usr/bin/env python3
"""
RunPod Serverless handler for DeepSeek V4 Flash.
First cold start: downloads ~49 GB weights + converts + compiles kernels (~30 min).
Subsequent requests on the same warm container: fast.
"""

import os
import sys
import json
import time
import subprocess
import runpod
from pathlib import Path

APP_DIR = Path("/app")
WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", "/runpod-volume/weights"))
HF_DIR = WEIGHTS_DIR / "hf"
CONVERTED_DIR = WEIGHTS_DIR / "converted"
TL_CACHE = str(WEIGHTS_DIR / "tl_cache")
HF_REPO = "deepseek-ai/DeepSeek-V4-Flash"
MP = 1  # single-GPU model parallelism

os.environ["WORLD_SIZE"] = "1"
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["TL_CACHE_DIR"] = TL_CACHE
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads

sys.path.insert(0, str(APP_DIR))

model = None
tokenizer = None
eos_id = None


def log(msg):
    print(f"[handler] {msg}", flush=True)


# ── Step 1: download HF weights ──────────────────────────────────────────────

def download_hf_weights():
    sentinel = HF_DIR / ".download_complete"
    if sentinel.exists():
        log("HF weights already downloaded, skipping.")
        return
    log(f"Downloading weights from {HF_REPO} (~49 GB)...")
    HF_DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(HF_DIR),
        ignore_patterns=["*.pdf", "assets/*", "encoding/*", "inference/*", "*.md"],
    )
    sentinel.touch()
    log("Weights downloaded.")


# ── Step 2: convert to DeepSeek inference format ─────────────────────────────

def convert_weights():
    out_file = CONVERTED_DIR / f"model0-mp{MP}.safetensors"
    if out_file.exists():
        log("Converted weights already present, skipping convert.")
        return
    log("Converting weights (MP=1)…")
    CONVERTED_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [sys.executable, str(APP_DIR / "convert.py"),
         "--hf-ckpt-path", str(HF_DIR),
         "--save-path", str(CONVERTED_DIR),
         "--n-experts", "256",
         "--model-parallel", str(MP)],
        cwd=str(APP_DIR),
        check=True,
    )
    log("Conversion complete.")


# ── Step 3: load model ────────────────────────────────────────────────────────

def load_model():
    global model, tokenizer, eos_id

    import torch
    from transformers import AutoTokenizer
    from model import Transformer, ModelArgs
    from safetensors.torch import load_model as load_safetensors

    log("Loading tokenizer…")
    tokenizer = AutoTokenizer.from_pretrained(str(HF_DIR))
    eos_id = tokenizer.eos_token_id

    log("Loading model config…")
    with open(APP_DIR / "inference_config.json") as f:
        args = ModelArgs(**json.load(f))
    args.max_batch_size = 1

    log("Instantiating model on GPU…")
    torch.cuda.set_device(0)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    with torch.device("cuda"):
        model = Transformer(args)

    weight_path = CONVERTED_DIR / f"model0-mp{MP}.safetensors"
    log(f"Loading weights from {weight_path}…")
    load_safetensors(model, str(weight_path), strict=False)
    torch.set_default_device("cuda")
    model.eval()
    log("Model ready.")


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(messages, max_new_tokens=512, temperature=0.7):
    import torch
    from generate import generate, sample
    from encoding_dsv4 import encode_messages as ds_encode, parse_message_from_completion_text

    prompt_str = ds_encode(messages, thinking_mode="chat")
    prompt_tokens = tokenizer.encode(prompt_str)

    completion_tokens = generate(
        model, [prompt_tokens], max_new_tokens, eos_id, temperature
    )
    completion_text = tokenizer.decode(completion_tokens[0])
    return completion_text


# ── RunPod handler ────────────────────────────────────────────────────────────

def handler(job):
    job_input = job.get("input", {})
    messages = job_input.get("messages", [])
    max_tokens = int(job_input.get("max_tokens", 512))
    temperature = float(job_input.get("temperature", 0.7))

    if not messages:
        return {"error": "No messages provided"}

    try:
        t0 = time.time()
        text = run_inference(messages, max_tokens, temperature)
        elapsed = round(time.time() - t0, 2)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"completion_time_seconds": elapsed},
        }
    except Exception as e:
        return {"error": str(e)}


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log("Starting DeepSeek V4 Flash serverless worker…")
    download_hf_weights()
    convert_weights()
    load_model()
    log("Entering RunPod handler loop.")
    runpod.serverless.start({"handler": handler})
