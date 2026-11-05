import subprocess
import json
import os
import logging

os.environ['HF_HUB_OFFLINE'] = "0"
# os.environ['HF_HOME'] = "/mnt/bn/lsh-lq/zhr/hf/huggingface"
# os.environ['FLASHINFER_NVCC'] = "/mnt/bn/lsh-lq/anaconda3/envs/zhr/bin/nvcc"
# conda_prefix = os.environ["CONDA_PREFIX"]
# old_ld = os.environ.get("LD_LIBRARY_PATH", "")
# os.environ["LD_LIBRARY_PATH"] = f"{conda_prefix}/lib:{old_ld}"


with open("config.json", "r") as f:
    config = json.load(f)

all_proc = []

for conf in [config[-1]]:
    model_path = conf["agent"]["model_name"]
    port = conf["agent"]["port"]
    device = conf["agent"]["device"]
    gpu_util = conf["agent"].get("gpu_util", 0.5)

    my_env = os.environ.copy()

    cuda_visible_devices = ",".join(map(str, device))
    my_env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--host", "0.0.0.0",
        "--max-model-len", "8192",
        "--max-num-batched-tokens", "4096",
        "--tensor-parallel-size", str(len(device)),
        "--gpu-memory-utilization", str(gpu_util),
        "--no-enable-prefix-caching"
    ]

    proc = subprocess.Popen(cmd, env=my_env)
    all_proc.append(proc)

for proc in all_proc:
    proc.wait()
