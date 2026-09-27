"""Explicit device selection for Stage 06 tests; no cross-device fallback."""

import json
import subprocess

import pytest
import torch
import transformers


def pytest_addoption(parser):
    parser.addoption("--device-role", choices=("rtx3090", "rtx4080super"), required=True)
    parser.addoption("--evidence-out", default=None)


@pytest.fixture(scope="session")
def gpu_evidence(request):
    role = request.config.getoption("--device-role")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name, "total_bytes": props.total_memory})
    found = next((d for d in devices if ("3090" if role == "rtx3090" else "4080 SUPER") in d["name"].upper()), None)
    query = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,memory.free,driver_version",
                            "--format=csv,noheader"], capture_output=True, text=True, check=False)
    evidence = {"requested_role": role, "visible_devices": devices,
                "nvidia_smi": query.stdout.strip().splitlines() if query.returncode == 0 else None,
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "status": "available" if found else "blocked_missing_gpu", "measurements": {}}
    yield evidence, found
    path = request.config.getoption("--evidence-out")
    if path:
        from pathlib import Path
        Path(path).write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def selected_cuda(gpu_evidence):
    evidence, found = gpu_evidence
    if found is None:
        pytest.skip(f"required {evidence['requested_role']} is not visible")
    torch.cuda.set_device(found["index"])
    return torch.device(f"cuda:{found['index']}")
