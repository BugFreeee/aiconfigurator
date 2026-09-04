# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect the seven requested Qwen micro workloads with Torch CUDA operations.

This is deliberately a standalone report generator, not a production AIC
performance-table collector.  It records the reference Torch operation actually
timed, so its rows cannot be mistaken for framework-selected serving kernels.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Workload:
    name: str
    kind: str
    phase: str
    batch_size: int
    sequence_length: int
    requested_dtype: str
    cache_tokens: int = 0
    query_heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    k_heads: int = 0
    v_heads: int = 0
    hidden_size: int = 0
    intermediate_size: int = 0
    top_k: int = 0
    local_experts: int = 0
    global_experts: int = 0
    data_parallel_size: int = 1
    expert_parallel_size: int = 1


WORKLOADS: tuple[Workload, ...] = (
    Workload("gated_gqa_decode", "attention", "decode", 18, 1, "mxfp8", 131072, 32, 2, 256, data_parallel_size=8),
    Workload("gated_gqa_prefill", "attention", "prefill", 18, 512, "mxfp8", 131072, 32, 2, 256, data_parallel_size=8),
    Workload("gdn_decode", "gdn", "decode", 16, 1, "bf16", k_heads=16, v_heads=64, head_dim=128, data_parallel_size=8),
    Workload(
        "gdn_prefill",
        "gdn",
        "prefill",
        16,
        512,
        "bf16",
        k_heads=16,
        v_heads=64,
        head_dim=128,
        data_parallel_size=8,
    ),
    Workload(
        "moe_down_prefill",
        "moe_down",
        "prefill",
        16,
        512,
        "fp8",
        hidden_size=1024,
        intermediate_size=4096,
        top_k=10,
        local_experts=64,
        global_experts=512,
        data_parallel_size=8,
        expert_parallel_size=8,
    ),
    Workload(
        "moe_gate_up_decode",
        "moe_gate_up",
        "decode",
        16,
        1,
        "fp8",
        hidden_size=1024,
        intermediate_size=4096,
        top_k=10,
        local_experts=64,
        global_experts=512,
        data_parallel_size=8,
        expert_parallel_size=8,
    ),
    Workload(
        "moe_gate_up_prefill",
        "moe_gate_up",
        "prefill",
        16,
        512,
        "fp8",
        hidden_size=1024,
        intermediate_size=4096,
        top_k=10,
        local_experts=64,
        global_experts=512,
        data_parallel_size=8,
        expert_parallel_size=8,
    ),
)


def moe_tokens_per_expert(workload: Workload) -> tuple[int, ...]:
    """Return a deterministic, unbalanced local-expert token distribution."""
    total = workload.batch_size * workload.sequence_length * workload.top_k
    total = total * workload.data_parallel_size // workload.expert_parallel_size
    hottest_expert = total // 2
    base, remainder = divmod(total - hottest_expert, workload.local_experts - 1)
    return (hottest_expert,) + tuple(
        base + (1 if expert < remainder else 0) for expert in range(workload.local_experts - 1)
    )


def _timed(operation: Callable[[], object], device: str, warmups: int, runs: int) -> tuple[float, int]:
    """Time one operation with AIC's shared CUDA-graph/power-aware harness."""
    import torch

    from collector.helper import benchmark_with_power

    with benchmark_with_power(
        device=torch.device(device),
        kernel_func=operation,
        num_warmups=warmups,
        num_runs=runs,
        repeat_n=1,
    ) as results:
        return float(results["latency_ms"]), int(results["num_runs_executed"])


def _captured_aic_rows(module, runner: Callable[[], object]) -> list[dict[str, object]]:
    """Run an AIC collector entry point while retaining its measured payload rows."""
    rows: list[dict[str, object]] = []
    original_log_perf = module.log_perf
    original_benchmark = module.benchmark_with_power
    measurements: list[dict[str, object]] = []

    @contextmanager
    def capture_benchmark(*args, **kwargs):
        with original_benchmark(*args, **kwargs) as results:
            yield results
        measurements.append(dict(results))

    def capture_log_perf(*, item_list, **kwargs):
        for item in item_list:
            if len(measurements) <= len(rows):
                raise RuntimeError("AIC collector logged a row before reporting its timing result")
            rows.append(
                {
                    **item,
                    "kernel_source": kwargs["kernel_source"],
                    "num_runs_executed": measurements[len(rows)]["num_runs_executed"],
                }
            )
        return True

    module.log_perf = capture_log_perf
    module.benchmark_with_power = capture_benchmark
    try:
        runner()
    finally:
        module.log_perf = original_log_perf
        module.benchmark_with_power = original_benchmark
    return rows


def _aic_attention_row(workload: Workload, device: str) -> tuple[str, str, float, int]:
    from collector.trtllm import collect_attn

    is_context = workload.phase == "prefill"
    rows = _captured_aic_rows(
        collect_attn,
        lambda: collect_attn.run_attention_torch(
            workload.batch_size,
            workload.sequence_length if is_context else workload.cache_tokens,
            workload.query_heads,
            workload.kv_heads,
            workload.head_dim,
            0,
            True,
            is_context,
            is_context,
            perf_filename="customized_qwen_micro_unused.txt",
            device=device,
            cached_tokens_per_seq=workload.cache_tokens if is_context else 0,
        ),
    )
    if len(rows) != 1:
        raise RuntimeError(f"AIC attention collector emitted {len(rows)} rows, expected one")
    return (
        "trtllm_torch_flow",
        str(rows[0]["attn_dtype"]),
        float(rows[0]["latency"]),
        int(rows[0]["num_runs_executed"]),
    )


def _aic_gdn_row(workload: Workload, device: str) -> tuple[str, str, float, int]:
    import torch
    if workload.phase == "prefill":
        from tensorrt_llm._torch.modules.fla.chunk import chunk_gated_delta_rule

        inputs = (
            torch.randn(
                workload.batch_size,
                workload.sequence_length,
                workload.k_heads,
                workload.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.randn(
                workload.batch_size,
                workload.sequence_length,
                workload.k_heads,
                workload.head_dim,
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.randn(
                workload.batch_size,
                workload.sequence_length,
                workload.v_heads,
                128,
                dtype=torch.bfloat16,
                device=device,
            ),
            torch.nn.functional.logsigmoid(
                torch.randn(
                    workload.batch_size,
                    workload.sequence_length,
                    workload.v_heads,
                    dtype=torch.bfloat16,
                    device=device,
                )
            ),
            torch.sigmoid(
                torch.randn(
                    workload.batch_size,
                    workload.sequence_length,
                    workload.v_heads,
                    dtype=torch.bfloat16,
                    device=device,
                )
            ),
        )
        operation = lambda: chunk_gated_delta_rule(*inputs)
        kernel = "trtllm_chunk_gated_delta_rule"
    else:
        from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule

        inputs = (
            torch.randn(
                workload.batch_size, 1, workload.k_heads, workload.head_dim, dtype=torch.bfloat16, device=device
            ),
            torch.randn(
                workload.batch_size, 1, workload.k_heads, workload.head_dim, dtype=torch.bfloat16, device=device
            ),
            torch.randn(workload.batch_size, 1, workload.v_heads, 128, dtype=torch.bfloat16, device=device),
            torch.nn.functional.logsigmoid(
                torch.randn(workload.batch_size, 1, workload.v_heads, dtype=torch.bfloat16, device=device)
            ),
            torch.sigmoid(torch.randn(workload.batch_size, 1, workload.v_heads, dtype=torch.bfloat16, device=device)),
        )
        state = torch.zeros(
            workload.batch_size,
            workload.v_heads,
            workload.head_dim,
            128,
            dtype=torch.bfloat16,
            device=device,
        )
        operation = lambda: fused_recurrent_gated_delta_rule(*inputs, initial_state=state, output_final_state=True)
        kernel = "trtllm_fused_recurrent_gated_delta_rule"

    latency_ms, actual_runs = _timed(operation, device, warmups=3, runs=10)
    return kernel, "bfloat16", latency_ms, actual_runs


def _moe_operation(workload: Workload, device: str) -> Callable[[], object]:
    import deep_gemm
    import torch
    from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

    tokens = moe_tokens_per_expert(workload)
    if workload.kind == "moe_down":
        input_width, output_width = workload.intermediate_size, workload.hidden_size
    else:
        input_width, output_width = workload.hidden_size, 2 * workload.intermediate_size
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    padded_tokens = tuple(((count + alignment - 1) // alignment) * alignment for count in tokens)
    total_padded_tokens = sum(padded_tokens)
    inputs = torch.zeros(total_padded_tokens, input_width, dtype=torch.bfloat16, device=device)
    weights = torch.randn(
        workload.local_experts, output_width, input_width, dtype=torch.bfloat16, device=device
    )
    expert_layout = torch.empty(total_padded_tokens, dtype=torch.int32, device=device)
    offset = 0
    for expert, (count, padded_count) in enumerate(zip(tokens, padded_tokens, strict=True)):
        inputs[offset : offset + count].normal_()
        expert_layout[offset : offset + count] = expert
        expert_layout[offset + count : offset + padded_count] = -1
        offset += padded_count

    inputs_fp8, input_scales = per_token_cast_to_fp8(inputs, use_ue8m0=True)
    weight_parts = [per_block_cast_to_fp8(weight, use_ue8m0=True) for weight in weights]
    weights_fp8 = torch.stack([part[0] for part in weight_parts])
    weight_scales = torch.stack([part[1] for part in weight_parts])
    output = torch.empty((total_padded_tokens, output_width), dtype=torch.bfloat16, device=device)
    expert_layout = expert_layout.contiguous()
    return_value = deep_gemm.m_grouped_fp8_gemm_nt_contiguous

    def operation():
        return return_value(
            (inputs_fp8, input_scales),
            (weights_fp8, weight_scales),
            output,
            expert_layout,
        )

    return operation


def _aic_row(workload: Workload, device: str) -> tuple[str, str, float, int] | None:
    if workload.kind == "attention":
        return _aic_attention_row(workload, device)
    if workload.kind == "gdn":
        return _aic_gdn_row(workload, device)
    return None


def _operation_for(workload: Workload, device: str) -> tuple[str, str, Callable[[], object]]:
    return (
        "deepgemm_m_grouped_fp8_gemm_nt_contiguous",
        "fp8",
        _moe_operation(workload, device),
    )


def _reset_clocks(nvidia_smi_index: int) -> dict[str, object]:
    command = ["nvidia-smi", "-i", str(nvidia_smi_index), "-rac"]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _run_one(
    workload: Workload,
    device: str,
    warmups: int,
    runs: int,
    comment: str,
    output_dir: Path,
    smi_index: int,
) -> dict[str, object]:
    import torch

    started_at = datetime.now(UTC).isoformat()
    dmon_path = output_dir / f"nvidia_dmon_{workload.name}.csv"
    with dmon_path.open("w", encoding="utf-8") as dmon_output:
        dmon = subprocess.Popen(
            ["nvidia-smi", "dmon", "-i", str(smi_index), "-s", "p", "-d", "1", "-o", "DT"],
            stdout=dmon_output,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            aic_row = _aic_row(workload, device)
            if aic_row is None:
                kernel, executed_dtype, operation = _operation_for(workload, device)
                mean_ms, actual_runs = _timed(operation, device, warmups, runs)
                median_ms = mean_ms
            else:
                kernel, executed_dtype, mean_ms, actual_runs = aic_row
                median_ms = mean_ms
            status, error = "ok", ""
        except Exception as exc:
            kernel, executed_dtype, mean_ms, median_ms, actual_runs = "", "", None, None, 0
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        finally:
            dmon.terminate()
            dmon.wait(timeout=5)

    row: dict[str, object] = {
        **asdict(workload),
        "device": device,
        "kernel": kernel,
        "executed_dtype": executed_dtype,
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "configured_warmups": warmups,
        "configured_runs": runs,
        "actual_timing_runs": actual_runs,
        "status": status,
        "error": error,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "time_tag": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "comment": comment,
        "dmon_path": str(dmon_path),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(device))),
    }
    return row


def _write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    values = list(rows)
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--nvidia-smi-index",
        type=int,
        default=0,
        help="Visible GPU index (host GPU 1 becomes 0 when passed through --gpus device=1).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--comment", default="")
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=[workload.name for workload in WORKLOADS],
        help="Optional workload names. Omit to run the full seven-workload suite.",
    )
    args = parser.parse_args()
    if args.warmups < 0 or args.runs < 1:
        parser.error("--warmups must be non-negative and --runs must be positive")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this collector does not fall back to CPU")
    torch.cuda.set_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reset = _reset_clocks(args.nvidia_smi_index)
    selected_workloads = (
        tuple(workload for workload in WORKLOADS if workload.name in set(args.workloads)) if args.workloads else WORKLOADS
    )
    rows = [
        _run_one(
            workload,
            args.device,
            args.warmups,
            args.runs,
            args.comment,
            args.output_dir,
            args.nvidia_smi_index,
        )
        for workload in selected_workloads
    ]
    _write_rows(args.output_dir / "qwen_micro_results.csv", rows)
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps({"clock_reset": reset, "workloads": [asdict(workload) for workload in selected_workloads]}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 1 if any(row["status"] != "ok" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
