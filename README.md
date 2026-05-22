# RunPod vLLM Deployment

Deploy Qwen 2.5 7B Instruct (or any HuggingFace model) to RunPod Serverless using the standard vLLM worker image.

No custom Docker image needed. Just one script.

## Quick Start

```bash
# 1. Set your RunPod API key
export RUNPOD_API_KEY=rpa_...

# 2. Deploy
python3 deploy.py

# 3. Test
python3 test_endpoint.py
```

## What This Does

| Step | What happens |
|------|-------------|
| `deploy.py` | Creates RunPod template + endpoint with vLLM serving Qwen 2.5 7B on A100 80GB |
| `test_endpoint.py` | Sends a chat request and polls for completion |
| `update_endpoint.py` | Updates existing endpoint to a different image/model |

## Configuration

Edit `deploy.py` to change the model:

```python
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"  # Change to any vLLM-compatible model
GPU_IDS = "AMPERE_80"                     # Change GPU type if needed
```

## Available GPU Types

- `AMPERE_80` — A100 80GB (recommended for 7B-70B models)
- `ADA_80_PRO` — L40S
- `HOPPER_141` — H100

## Custom Docker Build (Advanced)

If you need a custom inference engine (e.g., DeepSeek V4 Flash with custom MLA kernels):

```bash
python3 build_on_runpod.py   # Builds Docker on a RunPod GPU pod
python3 update_endpoint.py   # Updates endpoint to use custom image
```

## API

Once deployed, send requests to your RunPod endpoint:

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/run" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [{"role": "user", "content": "Hello!"}],
      "max_tokens": 512
    }
  }'
```
