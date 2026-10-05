import pytest
import torch

from sinklab.metrics import attention_jsd_decomposition


def test_s7_decomposition_cuda_20_to_16_heads_length128(selected_cuda, gpu_evidence):
    device = selected_cuda
    length = 128
    support = torch.ones((length, length), dtype=torch.bool, device=device).tril()
    generator = torch.Generator(device=device).manual_seed(20261005)
    teacher_scores = torch.randn((1, 20, length, length), generator=generator, device=device)
    student_scores = torch.randn((1, 16, length, length), generator=generator, device=device)
    teacher = torch.softmax(teacher_scores.masked_fill(~support, -1e9), dim=-1)
    student = torch.softmax(student_scores.masked_fill(~support, -1e9), dim=-1)
    result = attention_jsd_decomposition(teacher, student,
                                          torch.ones((1, length), dtype=torch.bool, device=device))
    assert result["valid_query_count"] == 127
    assert result["conditional_valid_query_count"] == 127
    assert result["full_jsd_nats"] == pytest.approx(
        result["mass_jsd_nats"] + result["shape_jsd_nats"], abs=1e-8)
    assert result["full_jsd_nats"] == pytest.approx(
        result["key0_jsd_nats"] + result["other_columns_jsd_nats"], abs=1e-8)
    gpu_evidence[0]["measurements"]["s7_synthetic_decomposition"] = {
        "teacher_heads": 20, "student_heads": 16, "sequence_length": length,
        "valid_queries": result["valid_query_count"],
        "max_closure_error_nats": result["max_closure_error_nats"],
        "scope": "synthetic numerical smoke only; no model or production qualification"}
