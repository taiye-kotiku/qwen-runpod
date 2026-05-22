#!/usr/bin/env python3
"""Update a RunPod endpoint to use a different template/image."""

import os, json, requests, sys

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
GRAPHQL_URL = "https://api.runpod.io/graphql"


def gql(query, variables=None):
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


def create_template(image_name: str, env_vars: list) -> str:
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
            "name": f"custom-{image_name.split('/')[-1].split(':')[0]}",
            "imageName": image_name,
            "containerDiskInGb": 40,
            "volumeInGb": 0,
            "volumeMountPath": "/runpod-volume",
            "env": env_vars,
            "isServerless": True,
        }
    }
    result = gql(mutation, variables)
    tid = result["saveTemplate"]["id"]
    print(f"[+] Template created: {tid}")
    return tid


def update_endpoint(endpoint_id: str, template_id: str, gpu_ids: str = "AMPERE_80") -> None:
    mutation = """
    mutation SaveEndpoint($input: EndpointInput!) {
      saveEndpoint(input: $input) {
        id
        name
        templateId
      }
    }
    """
    variables = {
        "input": {
            "id": endpoint_id,
            "name": "qwen3.6-35b-a3b",
            "templateId": template_id,
            "gpuIds": gpu_ids,
            "workersMin": 0,
            "workersMax": 5,
            "idleTimeout": 5,
            "flashboot": True,
            "gpuCount": 1,
            "scalerType": "QUEUE_DELAY",
            "scalerValue": 4,
        }
    }
    result = gql(mutation, variables)
    print(f"[+] Endpoint updated: {result['saveEndpoint']}")


if __name__ == "__main__":
    endpoint_id_path = "/opt/deepseek/.endpoint_id"
    if os.path.exists(endpoint_id_path):
        endpoint_id = open(endpoint_id_path).read().strip()
    else:
        print("ERROR: No .endpoint_id found. Run deploy.py first.")
        sys.exit(1)

    image_name = os.environ.get("IMAGE_NAME", "runpod/worker-vllm:stable-cuda12.1.0")
    env_vars = [
        {"key": "MODEL_NAME", "value": os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")},
        {"key": "MAX_MODEL_LEN", "value": "32768"},
        {"key": "GPU_MEMORY_UTILIZATION", "value": "0.95"},
        {"key": "DTYPE", "value": "bfloat16"},
    ]

    print(f"Updating endpoint {endpoint_id}...")
    print(f"  Image: {image_name}")
    print(f"  Model: {[v['value'] for v in env_vars if v['key'] == 'MODEL_NAME'][0]}")

    new_template_id = create_template(image_name, env_vars)
    update_endpoint(endpoint_id, new_template_id)
