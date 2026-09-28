"""Approved-device and cross-role exact-environment capture gates."""

import json
from pathlib import Path

import pytest

from scripts.capture_stage06_environment import (
    verify_gpu_identity, verify_locked_distribution_equality,
)
from sinklab.provenance import verify_envelope


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "protocols/s1_environment_4080_exact_v1.json")
                       .read_text(encoding="utf-8"))


def test_historical_4080_capture_identity_is_unchanged():
    fields = ["NVIDIA GeForce RTX 4080 SUPER",
              "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee", "16376 MiB", "591.86"]
    verify_gpu_identity(fields, "rtx4080super", None, None)
    with pytest.raises(RuntimeError, match="unexpected GPU identity"):
        verify_gpu_identity([*fields[:1], "GPU-wrong", *fields[2:]],
                            "rtx4080super", None, None)


def test_3090_capture_requires_explicit_approved_model_and_uuid():
    fields = ["NVIDIA GeForce RTX 3090",
              "GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e", "24576 MiB", "591.86"]
    with pytest.raises(RuntimeError, match="requires explicit"):
        verify_gpu_identity(fields, "rtx3090", None, None)
    verify_gpu_identity(fields, "rtx3090", fields[0], fields[1])
    with pytest.raises(RuntimeError, match="approved device role"):
        verify_gpu_identity(fields, "rtx3090", "NVIDIA GeForce RTX 4080 SUPER", fields[1])
    with pytest.raises(RuntimeError, match="unexpected GPU identity"):
        verify_gpu_identity(["NVIDIA GeForce RTX 4080 SUPER", *fields[1:]],
                            "rtx3090", fields[0], fields[1])
    with pytest.raises(RuntimeError, match="unexpected GPU identity"):
        verify_gpu_identity([fields[0], "GPU-wrong", *fields[2:]],
                            "rtx3090", fields[0], fields[1])


def test_lock_managed_distribution_map_requires_exact_4080_equality():
    payload, _ = verify_envelope(REFERENCE)
    installed = payload["installed_locked_distributions"]
    verify_locked_distribution_equality(installed, REFERENCE)
    with pytest.raises(RuntimeError, match="missing"):
        verify_locked_distribution_equality({key: value for key, value in installed.items()
                                             if key != "torch"}, REFERENCE)
    with pytest.raises(RuntimeError, match="changed"):
        verify_locked_distribution_equality({**installed, "torch": "0"}, REFERENCE)
