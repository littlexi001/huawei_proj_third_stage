# Adaptive-Metis Local Toolchain

## 1. Download calibration data

```bash
python3 scripts/download_eval_data.py \
  --dataset all \
  --limit 256 \
  --cache-dir data/hf_cache \
  --out-dir data/eval
```

This writes:

- `data/eval/mmlu_validation_*.jsonl`
- `data/eval/gsm8k_test_*.jsonl`

The script uses:

- MMLU: `load_dataset("cais/mmlu", "all", split="validation")`
- GSM8K: `load_dataset("openai/gsm8k", "main", split="test")`

## 2. Run a small local rank-search proxy

```bash
python3 scripts/adaptive_metis_rank_search.py \
  --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
  --data data/eval/mmlu_validation_256.jsonl \
  --task mmlu \
  --format hif8 \
  --limit 64 \
  --max-modules 0 \
  --max-act-rows 512 \
  --rank-candidates 0,5,10,15,20,25,30,40,50,60,80,100 \
  --output data/results/qwen3_0.6b_hif8_rank_search.json
```

For a fast smoke test:

```bash
python3 scripts/adaptive_metis_rank_search.py \
  --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
  --data data/eval/mmlu_validation_16.jsonl \
  --task mmlu \
  --format hif8 \
  --limit 2 \
  --max-modules 2 \
  --max-act-rows 16 \
  --rank-candidates 0,5 \
  --output data/results/smoke_rank_search.json
```

## 3. Current scope

`adaptive_metis_rank_search.py` is a calibration-proxy search tool. It does not run full MMLU accuracy evaluation. Its HIF4/HIF8 quantize-dequantize functions match the PyTorch/NumPy reference behavior under `Metis/`. It also supports `fp8` and `nvfp8` for the current third-stage experiment.

It currently computes a calibration proxy:

```math
N_i(k_a,k_p) = N_i^X(k_a) + N_i^W(k_p)
```

where `N_i^X` and `N_i^W` are residual quantization errors after low-rank spectral removal.

HIF4 follows the `hifx4` 64 -> 8 -> 4 hierarchical shared-scale implementation. HIF8 follows the per-value tapered mantissa implementation. FP8 uses tensor-wise scaled E4M3. NVFP8 uses block-wise scaled E4M3 with FP8-quantized scales.

## 4. Run the held-out global oracle sweep

The oracle sweep evaluates all 12 x 12 global `(ka, kp)` combinations with
real model forward passes. The first 256 deterministically shuffled MMLU test
examples remain the calibration set; the following 512 examples are held out.

HIF4:

```bash
python3 scripts/adaptive_metis_oracle_sweep.py \
  --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
  --data data/eval/mmlu_test_14042.jsonl \
  --format hif4 \
  --rank-candidates 0,5,10,15,20,25,30,40,50,60,80,100 \
  --calibration-size 256 \
  --eval-limit 512 \
  --batch-size 8 \
  --device cuda \
  --output data/results/qwen3_0.6b_hif4_oracle.jsonl
```

Run the same command with `--format hif8`, `--format fp8`, or `--format nvfp8`
and a different output file for each format. The JSONL output is append-only and resumable. Formal runs should leave
`--max-modules 0`, which quantizes every Linear module except `lm_head`.

To evaluate sparse MMLU anchor points instead of the full Cartesian grid, pass
explicit pairs:

```bash
python3 scripts/adaptive_metis_oracle_sweep.py \
  --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
  --data data/eval/mmlu_test_14042.jsonl \
  --format nvfp8 \
  --rank-candidates 0,5,10,15,20,25,30,40,50,60,80,100 \
  --pairs 0:0,0:100,100:0,20:20,40:40,60:60,80:80,100:100,20:80,80:20 \
  --calibration-size 256 \
  --eval-limit 512 \
  --batch-size 8 \
  --device cuda \
  --output data/results/qwen3_0.6b_nvfp8_anchors.jsonl
```

For local Apple Silicon execution, use MPS and a conservative batch size:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 scripts/adaptive_metis_oracle_sweep.py \
  --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
  --data data/eval/mmlu_test_14042.jsonl \
  --format hif4 \
  --rank-candidates 0,5,10,15,20,25,30,40,50,60,80,100 \
  --calibration-size 256 \
  --eval-limit 512 \
  --batch-size 2 \
  --device mps \
  --dtype float16 \
  --svd-device auto \
  --output data/results/qwen3_0.6b_hif4_oracle.jsonl
```

`--svd-device auto` first runs SVD on MPS and retries unsupported operations
on CPU. If the MPS SVD path is unstable on a particular macOS/PyTorch build,
use `--svd-device cpu`. The latter is slower but leaves model forward and
quantization on MPS. Start with `--eval-limit 32` to estimate runtime before
launching the 512-example resumable sweep.

Until hardware profiling provides a latency model, `rank_cost` defaults to
`ka + kp`. Measured per-rank costs can be supplied with `--ka-cost` and
`--kp-cost`.

## 5. Analyze calibration-proxy correlation

HIF4 uses a maximum allowed absolute accuracy drop of 0.01; HIF8 uses 0.005.

```bash
python3 scripts/analyze_proxy_oracle.py \
  --proxy data/results/qwen3_0.6b_hif4_rank_search.json \
  --oracle data/results/qwen3_0.6b_hif4_oracle.jsonl \
  --max-accuracy-drop 0.01 \
  --output data/results/qwen3_0.6b_hif4_proxy_oracle_analysis.json
```

The report includes Pearson and Spearman correlations against final-hidden
MSE, final-logit MSE, and MMLU accuracy drop, plus the minimum-cost feasible
oracle configuration.

## 6. Predict the dense grid from sparse anchors

After running a dense proxy grid and sparse anchor MMLU evaluation, fit a small
regularized response surface and select all configurations whose conservative
predicted accuracy drop is within the target. For a 1% drop budget:

```bash
python3 scripts/select_metis_configs.py \
  --proxy data/results/qwen3_0.6b_nvfp8_rank_search.json \
  --anchors data/results/qwen3_0.6b_nvfp8_anchors.jsonl \
  --max-accuracy-drop 0.01 \
  --uncertainty-lambda 1.0 \
  --output data/results/qwen3_0.6b_nvfp8_selected.json
```

The output contains `top_feasible`, `all_candidates`, and
`best_min_cost_feasible`. Run the best point once more on the full downstream
task to verify whether the prediction is accurate.

For the HIF8/FP8/NVFP8 third-stage experiment, the same flow can be launched by
one script:

```bash
MODEL_PATH=/path/to/Qwen3-0.6B bash run_adaptive_metis_fp8_experiment.sh
```

Useful overrides:

```bash
FORMATS="hif8 fp8 nvfp8" \
DEVICE=cuda \
RUN_VERIFY=1 \
MODEL_PATH=/path/to/Qwen3-0.6B \
bash run_adaptive_metis_fp8_experiment.sh
```

## 7. Scaling to larger models

The scripts are model-path agnostic as long as the model can be loaded by:

```python
AutoModelForCausalLM.from_pretrained(model_path)
AutoTokenizer.from_pretrained(model_path)
```

For larger models, start with:

- smaller `--limit`
- smaller `--max-act-rows`
- module filters such as `--module-name-contains q_proj,k_proj,v_proj,o_proj`
- a nonzero `--max-modules` during smoke tests
