#!/usr/bin/env python3
"""Deploy Qwen 2.5 7B Instruct to RunPod Serverless using standard vLLM.

No custom Docker image needed — uses RunPod's official vLLM worker.
"""

import os
import sys
import json
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
GRAPHQL_URL = "https://api.runpod.io/graphql"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# GPU pool IDs — A100 80GB
GPU_IDS = "AMPERE_80"

VLLM_IMAGE = "runpod/worker-vllm:stable-cuda12.1.0"


def gql(query: str, variables: dict = None) -> dict:
    resp = requests.post(
        GRAPHQL_URL,
        params={"api_key": RUNPOD_API_KEY},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


def create_template() -> str:
    env_vars = [
        {"key": "MODEL_NAME", "value": MODEL_NAME},
        {"key": "MAX_MODEL_LEN", "value": "32768"},
        {"key": "GPU_MEMORY_UTILIZATION", "value": "0.95"},
        {"key": "TENSOR_PARALLEL_SIZE", "value": "1"},
        {"key": "DTYPE", "value": "bfloat16"},
        {"key": "DISABLE_LOG_STATS", "value": "false"},
    ]

    mutation = """
    mutation SaveTemplate($input: SaveTemplateInput!) {
      saveTemplate(input: $input) {
        id
        name
      }
    }
    """
    variables = {
        "input": {
            "name": f"qwen2.5-7b-instruct-vllm",
            "imageName": VLLM_IMAGE,
            "containerDiskInGb": 40,
            "volumeInGb": 0,
            "volumeMountPath": "/runpod-volume",
            "ports": "8000/http",
            "dockerArgs": "",
            "env": env_vars,
            "readme": f"{MODEL_NAME} served via vLLM on RunPod Serverless",
            "isServerless": True,
        }
    }

    data = gql(mutation, variables)
    template_id = data["saveTemplate"]["id"]
    print(f"[+] Template created: {template_id}")
    return template_id


def create_endpoint(template_id: str) -> dict:
    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) {
        id
        name
        templateId
        gpuIds
        workersMin
        workersMax
        idleTimeout
      }
    }
    """
    variables = {
        "input": {
            "name": "qwen2.5-7b-instruct",
            "templateId": template_id,
            "gpuIds": GPU_IDS,
            "workersMin": 0,
            "workersMax": 3,
            "idleTimeout": 5,
            "gpuCount": 1,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }

    data = gql(mutation, variables)
    endpoint = data["saveEndpoint"]
    print(f"[+] Endpoint created: {endpoint['id']}")
    return endpoint


def main():
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set. Run: export RUNPOD_API_KEY=your_key_here")
        sys.exit(1)

    print(f"Deploying {MODEL_NAME} to RunPod Serverless...")
    print(f"  GPU:   {GPU_IDS} (A100 80GB)")
    print(f"  Image: {VLLM_IMAGE}")
    print()

    template_id = create_template()
    endpoint = create_endpoint(template_id)

    endpoint_id = endpoint["id"]
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    print()
    print("=" * 60)
    print("DEPLOYMENT COMPLETE")
    print("=" * 60)
    print(f"Endpoint ID : {endpoint_id}")
    print(f"Run URL     : {base_url}/run")
    print(f"Status URL  : {base_url}/status/<job_id>")
    print(f"Health URL  : {base_url}/health")
    print()
    print("vLLM exposes an OpenAI-compatible API internally.")
    print("RunPod wraps it: POST /run with input = openai chat payload.")
    print()
    print("Test with:")
    print(f"  python3 test_endpoint.py {endpoint_id}")
    print()
    print("NOTE: First request may take 2-5 min while the worker cold-starts")
    print("      and downloads the model weights (~15 GB).")

    with open(".endpoint_id", "w") as f:
        f.write(endpoint_id)


if __name__ == "__main__":
    main()
