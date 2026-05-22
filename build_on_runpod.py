#!/usr/bin/env python3
"""Build a custom Docker image on a RunPod GPU pod and push to Docker Hub.

Use this when you need a custom inference engine (e.g., custom kernels, model changes).
For standard vLLM deployments, just use deploy.py instead.
"""

import json, subprocess, sys, time
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()

RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY", "")
DOCKER_USER = os.environ["DOCKER_USER"]
DOCKER_PASS = os.environ["DOCKER_PASS"]
IMAGE = f"{DOCKER_USER}/runpod-custom:latest"
DOCKER_DIR = Path("/opt/deepseek/docker")
SSH_KEY_PATH = Path.home() / ".ssh" / "runpod_builder"
GRAPHQL_URL = "https://api.runpod.io/graphql"


def gql(query, variables=None):
    r = requests.post(
        GRAPHQL_URL,
        params={"api_key": RUNPOD_API_KEY},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data["data"]


def log(msg):
    print(f"[build_on_runpod] {msg}", flush=True)


def ssh(host, port, cmd, *, check=True, capture=False):
    args = [
        "ssh", "-i", str(SSH_KEY_PATH),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=20",
        "-p", str(port),
        f"root@{host}", cmd,
    ]
    if capture:
        return subprocess.run(args, capture_output=True, text=True)
    return subprocess.run(args, check=check)


def ensure_ssh_key():
    SSH_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SSH_KEY_PATH.exists():
        log("Generating SSH key...")
        subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", str(SSH_KEY_PATH), "-N", ""],
            check=True, capture_output=True,
        )
    pub = SSH_KEY_PATH.with_suffix(".pub").read_text().strip()
    log(f"Public key: {pub[:60]}...")
    return pub


def pick_gpu():
    query = """
    query {
      gpuTypes {
        id displayName memoryInGb secureCloud
        lowestPrice(input:{gpuCount:1}) { minimumBidPrice stockStatus }
      }
    }
    """
    types = gql(query)["gpuTypes"]
    candidates = [
        g for g in types
        if g["secureCloud"]
        and g["memoryInGb"] >= 16
        and g["lowestPrice"]
        and g["lowestPrice"]["stockStatus"] in ("High", "Medium")
    ]
    candidates.sort(key=lambda g: g["lowestPrice"]["minimumBidPrice"] or 99)
    if not candidates:
        candidates = sorted(types, key=lambda g: g.get("lowestPrice", {}).get("minimumBidPrice") or 99)
    if not candidates:
        raise RuntimeError("No GPUs available right now!")
    g = candidates[0]
    log(f"Selected GPU: {g['displayName']} ({g['memoryInGb']} GB) "
        f"@ ${g['lowestPrice']['minimumBidPrice']:.3f}/hr")
    return g["id"]


def create_pod(gpu_type_id, pub_key):
    mutation = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id name desiredStatus
      }
    }
    """
    variables = {
        "input": {
            "cloudType":         "SECURE",
            "gpuCount":          1,
            "gpuTypeId":         gpu_type_id,
            "containerDiskInGb": 200,
            "volumeInGb":        0,
            "minMemoryInGb":     15,
            "minVcpuCount":      4,
            "name":              "custom-docker-builder",
            "imageName":         "runpod/base:0.6.2",
            "startSsh":          True,
            "supportPublicIp":   True,
            "ports":             "22/tcp",
            "dockerArgs":        "--privileged",
            "env": [{"key": "PUBLIC_KEY", "value": pub_key}],
        }
    }
    pod = gql(mutation, variables)["podFindAndDeployOnDemand"]
    log(f"Pod created: {pod['id']} (status={pod['desiredStatus']})")
    return pod["id"]


def wait_for_ssh(pod_id, timeout=600):
    query = """
    query Pod($id: String!) {
      pod(input:{podId:$id}) {
        desiredStatus
        runtime {
          ports { ip isIpPublic privatePort publicPort }
        }
      }
    }
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        pod    = gql(query, {"id": pod_id})["pod"]
        status = pod["desiredStatus"]
        log(f"  Pod status: {status}")
        if status == "RUNNING" and pod.get("runtime"):
            for p in pod["runtime"]["ports"]:
                if p["privatePort"] == 22 and p["isIpPublic"]:
                    host, port = p["ip"], p["publicPort"]
                    log(f"  Trying SSH at {host}:{port}...")
                    for _ in range(30):
                        r = ssh(host, port, "echo ok", check=False, capture=True)
                        if r.returncode == 0:
                            log("  SSH connected.")
                            return host, port
                        time.sleep(10)
        time.sleep(15)
    raise TimeoutError("Pod SSH never became available")


def run_build(host, port):
    log("Installing Docker...")
    ssh(host, port,
        "apt-get update -qq && "
        "apt-get install -y -qq docker.io ca-certificates 2>&1 | tail -3")

    log("Starting Docker daemon (privileged mode)...")
    ssh(host, port,
        "nohup dockerd > /dockerd.log 2>&1 & sleep 10 && "
        "timeout 60 bash -c 'until docker info &>/dev/null; do sleep 2; done' && "
        "echo Docker_OK")

    log("Copying build context...")
    ssh(host, port, "mkdir -p /build", capture=True)
    for f in DOCKER_DIR.iterdir():
        subprocess.run(
            ["scp", "-i", str(SSH_KEY_PATH),
             "-o", "StrictHostKeyChecking=no",
             "-P", str(port),
             str(f), f"root@{host}:/build/"],
            check=True,
        )
    log("Files copied.")

    log("Running docker build + push (~10-15 min)...")
    build_cmd = (
        f"cd /build && "
        f"echo '{DOCKER_PASS}' | docker login -u '{DOCKER_USER}' --password-stdin && "
        f"docker build -t {IMAGE} . 2>&1 && "
        f"docker push {IMAGE} && "
        f"echo BUILD_AND_PUSH_COMPLETE"
    )
    result = ssh(host, port, build_cmd, check=False)
    return result.returncode == 0


def terminate_pod(pod_id):
    mutation = "mutation($i:PodTerminateInput!){podTerminate(input:$i)}"
    gql(mutation, {"i": {"podId": pod_id}})
    log(f"Pod {pod_id} terminated.")


def main():
    pod_id = None
    try:
        pub_key = ensure_ssh_key()
        gpu_id = pick_gpu()
        pod_id = create_pod(gpu_id, pub_key)

        log("Waiting for pod SSH (up to 10 min)...")
        host, port = wait_for_ssh(pod_id)

        success = run_build(host, port)
        if success:
            log(f"SUCCESS - image pushed: {IMAGE}")
            log("Next step: update endpoint to use this image")
        else:
            log("FAILED - check output above.")
            sys.exit(1)

    finally:
        if pod_id:
            terminate_pod(pod_id)


if __name__ == "__main__":
    main()
