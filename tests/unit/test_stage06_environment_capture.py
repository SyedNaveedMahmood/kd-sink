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


def test_second_4080_capture_requires_distinct_explicit_uuid_without_changing_first_gate():
    first = ["NVIDIA GeForce RTX 4080 SUPER",
             "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee", "16376 MiB", "591.86"]
    second = [first[0], "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0", *first[2:]]
    verify_gpu_identity(second, "rtx4080super", second[0], second[1],
                        second_4080_qualification=True)
    with pytest.raises(RuntimeError, match="unexpected GPU identity"):
        verify_gpu_identity(second, "rtx4080super", None, None)
    with pytest.raises(RuntimeError, match="distinct explicit UUID"):
        verify_gpu_identity(first, "rtx4080super", first[0], first[1],
                            second_4080_qualification=True)
    with pytest.raises(RuntimeError, match="distinct explicit UUID"):
        verify_gpu_identity(second, "rtx4080super", None, second[1],
                            second_4080_qualification=True)
    with pytest.raises(RuntimeError, match="unexpected second 4080 GPU identity"):
        verify_gpu_identity(first, "rtx4080super", second[0], second[1],
                            second_4080_qualification=True)
    with pytest.raises(RuntimeError, match="distinct explicit UUID"):
        verify_gpu_identity(second, "rtx3090", second[0], second[1],
                            second_4080_qualification=True)


def test_class_environment_accepts_unseen_exact_model_and_records_uuid():
    peer = ["NVIDIA GeForce RTX 4080 SUPER",
            "GPU-11111111-2222-3333-4444-555555555555", "16376 MiB", "591.86"]
    verify_gpu_identity(peer, "rtx4080super", None, None, class_environment=True)
    with pytest.raises(RuntimeError, match="exact model"):
        verify_gpu_identity(["NVIDIA GeForce RTX 4080", *peer[1:]],
                            "rtx4080super", None, None, class_environment=True)
    with pytest.raises(RuntimeError, match="actual UUID"):
        verify_gpu_identity([peer[0], "", *peer[2:]],
                            "rtx4080super", None, None, class_environment=True)
    with pytest.raises(RuntimeError, match="exact model"):
        verify_gpu_identity(peer, "rtx3090", None, None, class_environment=True)


def test_lock_managed_distribution_map_requires_exact_4080_equality():
    payload, _ = verify_envelope(REFERENCE)
    installed = payload["installed_locked_distributions"]
    verify_locked_distribution_equality(installed, REFERENCE)
    with pytest.raises(RuntimeError, match="missing"):
        verify_locked_distribution_equality({key: value for key, value in installed.items()
                                             if key != "torch"}, REFERENCE)
    with pytest.raises(RuntimeError, match="changed"):
        verify_locked_distribution_equality({**installed, "torch": "0"}, REFERENCE)
