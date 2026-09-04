# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
from pathlib import Path

from collector.customized_qwen_micro import WORKLOADS, _write_rows, moe_tokens_per_expert


def test_workloads_match_requested_micro_list():
    assert [workload.name for workload in WORKLOADS] == [
        "gated_gqa_decode",
        "gated_gqa_prefill",
        "gdn_decode",
        "gdn_prefill",
        "moe_down_prefill",
        "moe_gate_up_decode",
        "moe_gate_up_prefill",
    ]
    attention = WORKLOADS[0]
    assert (attention.batch_size, attention.sequence_length, attention.cache_tokens) == (18, 1, 131072)
    assert (attention.query_heads, attention.kv_heads, attention.head_dim) == (32, 2, 256)


def test_moe_distribution_retains_requested_routed_token_count():
    workload = next(workload for workload in WORKLOADS if workload.name == "moe_down_prefill")

    tokens = moe_tokens_per_expert(workload)

    assert len(tokens) == 64
    assert sum(tokens) == 16 * 512 * 10
    assert tokens[0] == sum(tokens) // 2
    assert max(tokens[1:]) - min(tokens[1:]) <= 1


def test_moe_workloads_use_deepgemm_grouped_projection():
    source = (Path(__file__).parents[3] / "collector" / "customized_qwen_micro.py").read_text(encoding="utf-8")

    assert "deep_gemm.m_grouped_fp8_gemm_nt_contiguous" in source
    assert "torch.matmul(input_tensor, weight)" not in source


def test_attention_and_gdn_workloads_use_aic_trtllm_collectors():
    source = (Path(__file__).parents[3] / "collector" / "customized_qwen_micro.py").read_text(encoding="utf-8")

    assert "collect_attn.run_attention_torch" in source
    assert "from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule" in source
    assert "from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule" in source
    assert "causal_conv1d" not in source
    assert "cached_tokens_per_seq=workload.cache_tokens if is_context else 0" in source
    assert "from collector.helper import benchmark_with_power" in source
    assert '"--workloads"' in source


def test_cached_prefill_uses_proven_aic_metadata_contract():
    source = (Path(__file__).parents[3] / "collector" / "trtllm" / "collect_attn.py").read_text(encoding="utf-8")

    assert 'raise ValueError("cached_tokens_per_seq is supported only for context attention")' in source
    assert "num_cached_tokens_per_seq=[cached_tokens_per_seq for _ in range(batch_size)]" in source
    assert "chunked_prefill=False" in source
    assert "cache_reuse=cached_tokens_per_seq > 0" in source
    assert "step = cached_tokens_per_seq" in source
    assert "time.sleep" not in source


def test_write_rows_serializes_all_result_fields(tmp_path):
    output = tmp_path / "results.csv"

    _write_rows(output, [{"name": "one", "latency_median_ms": 1.2}, {"name": "two", "status": "ok"}])

    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == [
        {"latency_median_ms": "1.2", "name": "one", "status": ""},
        {"latency_median_ms": "", "name": "two", "status": "ok"},
    ]
