# Customized Qwen Micro Collection

[`customized_qwen_micro.py`](customized_qwen_micro.py) runs the seven
workloads declared in `QWEN_micro_listnew.xlsx`. This is a standalone report
collector: it does not add or publish AIC production performance tables.

## What runs

- Gated GQA decode and prefill through AIC's TRT-LLM attention collector,
  including the requested 128k cached prefix. The prefill case allocates the
  prefix, declares it through `KVCacheParams.num_cached_tokens_per_seq`, enables
  cache reuse, and times only the new 512-token chunk.
- GDN decode and prefill through AIC's TRT-LLM FLA delta-rule collectors.
- MoE down and gate/up projections through
  `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`.

DeepGEMM uses the requested `--warmups` and `--runs` values. Attention and GDN
retain their established AIC collector loops (attention: 10 warmups/6 runs;
GDN: 3 warmups/10 runs). Set
`COLLECTOR_MEASURE_POWER=1` to enable AIC's adaptive timing-loop extension for
power sampling. The full-run command below enables it: this keeps each timed
kernel loop active for at least the AIC power-sampling duration, allowing dmon
to sample active work rather than an idle cooldown period.

## Full run on host GPU 1

From the repository root:

Run the attention and GDN workloads in TensorRT-LLM:

```bash
docker run --rm \
  --gpus '"device=1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e COLLECTOR_MEASURE_POWER=1 \
  -v "$PWD:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc23 \
  python3 -m collector.customized_qwen_micro \
    --device cuda:0 \
    --output-dir collector_runs/qwen_micro_gpu1/trtllm \
    --warmups 10 \
    --runs 50 \
    --comment "QWEN_micro_listnew TRT-LLM run; host GPU 1 only" \
    --workloads gated_gqa_decode gated_gqa_prefill gdn_decode gdn_prefill
```

Docker exposes only host GPU 1, so it is `cuda:0` and `nvidia-smi -i 0` in the
container.

Run the three MoE projections in an upstream DeepGEMM image that provides
`deep_gemm` 2.6.1 or newer:

```bash
docker run --rm --entrypoint /bin/bash \
  --gpus '"device=1"' \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e COLLECTOR_MEASURE_POWER=1 \
  -e PYTHONPATH=/workspace \
  -v "$PWD:/workspace" \
  -w /workspace \
  deepgemm:fusion \
  -lc '/opt/venv/bin/python -m collector.customized_qwen_micro \
    --device cuda:0 \
    --output-dir collector_runs/qwen_micro_gpu1/deepgemm \
    --warmups 10 \
    --runs 50 \
    --comment "QWEN_micro_listnew DeepGEMM run; host GPU 1 only" \
    --workloads moe_down_prefill moe_gate_up_decode moe_gate_up_prefill'
```

The collector requests the GPU's default application clocks before each
runtime-specific run. The manifest records success or a permission failure.
Dmon starts immediately before each active benchmark loop and stops immediately
after it; the collector deliberately adds no pre-run or post-run sleep.

## Output

`--output-dir` contains:

- `qwen_micro_results.csv`: latency, configured and effective iteration count,
  kernel identity, requested/executed dtype, software/GPU identity, timestamps,
  and comments.
- `run_manifest.json`: workload definitions and the default-clock reset result.
- `nvidia_dmon_<workload>.csv`: one power-only dmon trace per workload.

The collector does not calculate or report average power. Calculate it from the
`pwr` column in the raw dmon trace using your chosen sampling policy.
