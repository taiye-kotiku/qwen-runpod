#!/usr/bin/env python3
"""Test a RunPod vLLM endpoint with OpenAI-compatible chat format."""

import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")


def get_endpoint_id() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    if os.path.exists(".endpoint_id"):
        return open(".endpoint_id").read().strip()
    print("Usage: python3 test_endpoint.py <endpoint_id>")
    sys.exit(1)


def run_job(endpoint_id: str, prompt: str) -> str:
    """RunPod vLLM expects OpenAI-compatible chat payload under 'input'."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/run"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    payload = {
        "input": {
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
        }
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def poll_job(endpoint_id: str, job_id: str, timeout: int = 300) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    start = time.time()
    print(f"  Waiting for job {job_id}...", end="", flush=True)
    while time.time() - start < timeout:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status == "COMPLETED":
            print(" done")
            return data
        elif status in ("FAILED", "CANCELLED"):
            print(f" {status}")
            raise RuntimeError(f"Job {status}: {json.dumps(data, indent=2)}")
        print(".", end="", flush=True)
        time.sleep(5)
    raise TimeoutError(f"Job did not complete within {timeout}s")


def check_health(endpoint_id: str) -> dict:
    url = f"https://api.runpod.ai/v2/{endpoint_id}/health"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json()


def main():
    if not RUNPOD_API_KEY:
        print("ERROR: RUNPOD_API_KEY not set.")
        sys.exit(1)

    endpoint_id = get_endpoint_id()
    print(f"Testing endpoint: {endpoint_id}")

    print("\n[1] Health check...")
    try:
        health = check_health(endpoint_id)
        print(f"    Status: {json.dumps(health, indent=4)}")
    except Exception as e:
        print(f"    Health check failed: {e}")

    prompt = "Explain what Qwen 2.5 is in 2 sentences."
    print(f"\n[2] Inference test...")
    print(f"    Prompt: {prompt}")

    job_id = run_job(endpoint_id, prompt)
    result = poll_job(endpoint_id, job_id)

    output = result.get("output", {})
    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content", "")
            print(f"\n    Response:\n    {text}")
        else:
            print(f"\n    Raw output: {json.dumps(output, indent=4)}")
    else:
        print(f"\n    Output: {output}")

    print("\n[+] Test complete.")


if __name__ == "__main__":
    main()
