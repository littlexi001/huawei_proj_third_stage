import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers


def parse_thresholds(spec: str):
    """Parse either '0,0.1,0.2' or 'linspace:0:1:21'."""
    spec = spec.strip()
    if spec.startswith("linspace:"):
        _, start, end, count = spec.split(":")
        return np.linspace(float(start), float(end), int(count)).tolist()
    return [float(x) for x in spec.split(",") if x.strip()]


def _reshape_blocks(x: torch.Tensor, block_size: int):
    tokens, dim = x.shape
    pad = (block_size - dim % block_size) % block_size
    if pad:
        x_pad = torch.nn.functional.pad(x, (0, pad))
    else:
        x_pad = x
    padded_dim = x_pad.shape[-1]
    num_blocks = padded_dim // block_size
    return x_pad.reshape(tokens, num_blocks, block_size), pad, padded_dim


def quantize_int4_blockwise(x: torch.Tensor, block_size: int, zero_abs_threshold: float):
    """
    Symmetric signed INT4 simulation with one scale per block.

    x: [tokens, dim]
    returns:
      xq: dequantized tensor, [tokens, dim]
      zero_ratio: [tokens, num_blocks], fraction of meaningful values quantized to 0

    Values whose original magnitude is below zero_abs_threshold are not counted
    as harmful zeroing even if their quantized value is 0.
    """
    tokens, dim = x.shape
    xb, pad, padded_dim = _reshape_blocks(x, block_size)

    qmax = 7.0
    max_abs = xb.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / qmax).clamp_min(1e-8)
    q = torch.round(xb / scale).clamp(-7, 7)
    xq = q * scale

    meaningful_zero = (q == 0) & (xb.abs() >= zero_abs_threshold)
    zero_ratio = meaningful_zero.to(torch.float32).mean(dim=-1)
    xq = xq.reshape(tokens, padded_dim)
    if pad:
        xq = xq[:, :dim]
    return xq, zero_ratio


def quantize_nvfp4_blockwise(x: torch.Tensor, block_size: int, zero_abs_threshold: float):
    """
    Approximate NVFP4-style microscaled FP4 quantization.

    This simulates an E2M1 FP4 codebook with one scale per micro-block. NVIDIA
    NVFP4 also uses implementation-specific scale formats and an additional
    tensor-level scaling strategy; for this patent-effect simulation, the key
    behavior is block scaling plus FP4 E2M1 rounding.

    The normalized E2M1 codebook used here is:
      0, +/-0.5, +/-1, +/-1.5, +/-2, +/-3, +/-4, +/-6

    Values whose original magnitude is below zero_abs_threshold are not counted
    as harmful zeroing even if their quantized value is 0.
    """
    tokens, dim = x.shape
    xb, pad, padded_dim = _reshape_blocks(x, block_size)

    codebook = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=x.device,
        dtype=x.dtype,
    )
    max_finite = codebook[-1]
    max_abs = xb.abs().amax(dim=-1, keepdim=True)
    scale = (max_abs / max_finite).clamp_min(1e-8)
    normalized_abs = (xb.abs() / scale).clamp(max=max_finite)

    # Nearest-neighbor quantization to the positive E2M1 codebook, then restore sign.
    distances = (normalized_abs.unsqueeze(-1) - codebook).abs()
    indices = distances.argmin(dim=-1)
    q_abs = codebook[indices]
    q = torch.sign(xb) * q_abs
    xq = q * scale

    meaningful_zero = (q_abs == 0) & (xb.abs() >= zero_abs_threshold)
    zero_ratio = meaningful_zero.to(torch.float32).mean(dim=-1)
    xq = xq.reshape(tokens, padded_dim)
    if pad:
        xq = xq[:, :dim]
    return xq, zero_ratio


def quantize_blockwise(x: torch.Tensor, block_size: int, zero_abs_threshold: float, quant_format: str):
    if quant_format == "nvfp4":
        return quantize_nvfp4_blockwise(x, block_size, zero_abs_threshold)
    if quant_format == "int4":
        return quantize_int4_blockwise(x, block_size, zero_abs_threshold)
    raise ValueError(f"Unsupported quantization format: {quant_format}")


@torch.no_grad()
def evaluate_layer(
    x: torch.Tensor,
    weight: torch.Tensor,
    thresholds,
    block_size: int,
    alpha: float,
    use_fp32_matmul: bool,
    zero_abs_threshold: float,
    quant_format: str,
    quantize_weight: bool,
):
    """
    Simulate per-token/per-block fallback on one layer.

    For each threshold tau:
      p_zero >  tau: fallback to original high-precision block
      p_zero <= tau: use quantized/dequantized low-precision block

    If quantize_weight is enabled, both X and W are quantized. A block
    contribution falls back when either the X-side block or the W-side block
    exceeds tau. This exact pairwise rule is more expensive than X-only
    simulation because fallback can depend on both token block and output-row
    block risk.

    Output error is ||Y_mix - Y_ref||_F / ||Y_ref||_F.
    """
    if x.dim() == 3:
        # [1, seq, dim] -> [seq, dim]
        x = x.squeeze(0)
    x = x.contiguous()
    weight = weight.contiguous()

    if use_fp32_matmul:
        x_mm = x.float()
        w_mm = weight.float()
    else:
        x_mm = x
        w_mm = weight

    y_ref = x_mm @ w_mm.t()
    ref_norm_sq = torch.sum(y_ref.float() ** 2).item()

    xq, x_zero_ratio = quantize_blockwise(
        x_mm,
        block_size=block_size,
        zero_abs_threshold=zero_abs_threshold,
        quant_format=quant_format,
    )

    if quantize_weight:
        wq, w_zero_ratio = quantize_blockwise(
            w_mm,
            block_size=block_size,
            zero_abs_threshold=zero_abs_threshold,
            quant_format=quant_format,
        )
    else:
        wq = w_mm
        w_zero_ratio = None

    tokens, dim = x_mm.shape
    out_dim = w_mm.shape[0]
    x_blocks, x_pad, x_padded_dim = _reshape_blocks(x_mm, block_size)
    xq_blocks, _, _ = _reshape_blocks(xq, block_size)
    w_blocks, w_pad, w_padded_dim = _reshape_blocks(w_mm, block_size)
    wq_blocks, _, _ = _reshape_blocks(wq, block_size)
    assert x_padded_dim == w_padded_dim
    num_blocks = x_padded_dim // block_size

    results = []
    if quantize_weight:
        # Exact pairwise fallback opportunities: token x output-row x K-block.
        total_blocks = x_zero_ratio.shape[0] * out_dim * num_blocks
    else:
        total_blocks = x_zero_ratio.numel()

    for tau in thresholds:
        # The patent threshold semantics: fallback only when the effective
        # zeroed-value ratio of a block exceeds tau.
        if quantize_weight:
            y_err = torch.zeros(tokens, out_dim, device=x_mm.device, dtype=x_mm.dtype)
            fallback_blocks = 0
            for block_idx in range(num_blocks):
                x_b = x_blocks[:, block_idx, :]
                xq_b = xq_blocks[:, block_idx, :]
                w_b = w_blocks[:, block_idx, :]
                wq_b = wq_blocks[:, block_idx, :]

                # Error if this K-block contribution uses low precision on
                # both operands: Xq_b @ Wq_b.T - X_b @ W_b.T.
                block_err = xq_b @ wq_b.t() - x_b @ w_b.t()
                fallback_mask = (
                    (x_zero_ratio[:, block_idx].unsqueeze(1) > tau)
                    | (w_zero_ratio[:, block_idx].unsqueeze(0) > tau)
                )
                fallback_blocks += fallback_mask.sum().item()
                low_mask = (~fallback_mask).to(block_err.dtype)
                y_err = y_err + block_err * low_mask
        else:
            delta_blocks = xq_blocks - x_blocks
            fallback_mask = x_zero_ratio > tau
            low_mask = (~fallback_mask).to(delta_blocks.dtype)
            fallback_blocks = fallback_mask.sum().item()
            # Non-fallback X blocks use low precision and therefore contribute
            # quantization residual. Fallback X blocks contribute zero residual.
            masked_delta = (delta_blocks * low_mask.unsqueeze(-1)).reshape(tokens, -1)
            if x_pad:
                masked_delta = masked_delta[:, :dim]
            y_err = masked_delta @ w_mm.t()

        fallback_rate = min(max(fallback_blocks / max(total_blocks, 1), 0.0), 1.0)
        err_norm_sq = torch.sum(y_err.float() ** 2).item()
        rel_error = math.sqrt(err_norm_sq / max(ref_norm_sq, 1e-30))
        latency_overhead = fallback_rate * (alpha - 1.0)

        results.append(
            {
                "tau": float(tau),
                "fallback_blocks": fallback_blocks,
                "total_blocks": total_blocks,
                "fallback_rate": fallback_rate,
                "latency_overhead": latency_overhead,
                "err_num_sq": err_norm_sq,
                "err_den_sq": ref_norm_sq,
                "relative_output_error": rel_error,
            }
        )

    return results


def merge_results(acc, layer_results):
    for r in layer_results:
        tau = r["tau"]
        if tau not in acc:
            acc[tau] = {
                "tau": tau,
                "fallback_blocks": 0.0,
                "total_blocks": 0.0,
                "err_num_sq": 0.0,
                "err_den_sq": 0.0,
                "relative_output_errors": [],
                "fallback_rates": [],
                "latency_overheads": [],
            }
        acc[tau]["fallback_blocks"] += r["fallback_blocks"]
        acc[tau]["total_blocks"] += r["total_blocks"]
        acc[tau]["err_num_sq"] += r["err_num_sq"]
        acc[tau]["err_den_sq"] += r["err_den_sq"]
        acc[tau]["relative_output_errors"].append(r["relative_output_error"])
        acc[tau]["fallback_rates"].append(r["fallback_rate"])
        acc[tau]["latency_overheads"].append(r["latency_overhead"])


def finalize_results(acc, alpha: float):
    rows = []
    for tau in sorted(acc):
        r = acc[tau]
        global_fallback_rate = r["fallback_blocks"] / max(r["total_blocks"], 1.0)
        global_latency_overhead = global_fallback_rate * (alpha - 1.0)
        global_rel_error = math.sqrt(r["err_num_sq"] / max(r["err_den_sq"], 1e-30))

        errors = np.asarray(r["relative_output_errors"], dtype=np.float64)
        fallback_rates = np.asarray(r["fallback_rates"], dtype=np.float64)
        latency_overheads = np.asarray(r["latency_overheads"], dtype=np.float64)

        rows.append(
            {
                "tau": tau,
                "mean_fallback_rate": float(fallback_rates.mean()),
                "median_fallback_rate": float(np.median(fallback_rates)),
                "p95_fallback_rate": float(np.percentile(fallback_rates, 95)),
                "max_fallback_rate": float(fallback_rates.max()),
                "mean_latency_overhead": float(latency_overheads.mean()),
                "median_latency_overhead": float(np.median(latency_overheads)),
                "p95_latency_overhead": float(np.percentile(latency_overheads, 95)),
                "max_latency_overhead": float(latency_overheads.max()),
                "mean_relative_output_error": float(errors.mean()),
                "median_relative_output_error": float(np.median(errors)),
                "p95_relative_output_error": float(np.percentile(errors, 95)),
                "max_relative_output_error": float(errors.max()),
                "global_fallback_rate": global_fallback_rate,
                "global_latency_overhead": global_latency_overhead,
                "global_energy_weighted_relative_output_error": global_rel_error,
                "fallback_blocks": r["fallback_blocks"],
                "total_blocks": r["total_blocks"],
                "num_layer_samples": int(errors.size),
            }
        )

    if rows:
        # The largest tau is the no-fallback / full low-precision baseline
        # under the current threshold semantics. Error reduction is reported
        # relative to that baseline for presentation-friendly patent figures.
        baseline = rows[-1]
        baseline_mean_error = max(baseline["mean_relative_output_error"], 1e-30)
        baseline_median_error = max(baseline["median_relative_output_error"], 1e-30)
        baseline_p95_error = max(baseline["p95_relative_output_error"], 1e-30)
        baseline_max_error = max(baseline["max_relative_output_error"], 1e-30)
        baseline_global_error = max(
            baseline["global_energy_weighted_relative_output_error"],
            1e-30,
        )
        for row in rows:
            row["mean_error_reduction"] = (
                baseline_mean_error - row["mean_relative_output_error"]
            ) / baseline_mean_error
            row["median_error_reduction"] = (
                baseline_median_error - row["median_relative_output_error"]
            ) / baseline_median_error
            row["p95_error_reduction"] = (
                baseline_p95_error - row["p95_relative_output_error"]
            ) / baseline_p95_error
            row["max_error_reduction"] = (
                baseline_max_error - row["max_relative_output_error"]
            ) / baseline_max_error
            row["global_energy_weighted_error_reduction"] = (
                baseline_global_error
                - row["global_energy_weighted_relative_output_error"]
            ) / baseline_global_error
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tau",
                "mean_fallback_rate",
                "median_fallback_rate",
                "p95_fallback_rate",
                "max_fallback_rate",
                "mean_latency_overhead",
                "median_latency_overhead",
                "p95_latency_overhead",
                "max_latency_overhead",
                "mean_relative_output_error",
                "median_relative_output_error",
                "p95_relative_output_error",
                "max_relative_output_error",
                "mean_error_reduction",
                "median_error_reduction",
                "p95_error_reduction",
                "max_error_reduction",
                "global_fallback_rate",
                "global_latency_overhead",
                "global_energy_weighted_relative_output_error",
                "global_energy_weighted_error_reduction",
                "fallback_blocks",
                "total_blocks",
                "num_layer_samples",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    tau = np.array([r["tau"] for r in rows])
    fallback = np.array([r["mean_fallback_rate"] for r in rows])
    latency = np.array([r["mean_latency_overhead"] for r in rows])
    error = np.array([r["mean_relative_output_error"] for r in rows])
    p95_error = np.array([r["p95_relative_output_error"] for r in rows])
    reduction = np.array([r["mean_error_reduction"] for r in rows])
    p95_reduction = np.array([r["p95_error_reduction"] for r in rows])

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(latency * 100.0, error * 100.0, marker="o")
    for x, y, t in zip(latency * 100.0, error * 100.0, tau):
        plt.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Estimated latency overhead (%)")
    plt.ylabel("Mean relative output error (%)")
    plt.title("Block Fallback Tradeoff: Mean Latency vs Mean Output Error")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_error.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(latency * 100.0, error * 100.0, marker="o", label="Mean")
    plt.plot(latency * 100.0, p95_error * 100.0, marker="s", label="P95")
    for x, y, t in zip(latency * 100.0, error * 100.0, tau):
        plt.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Mean estimated latency overhead (%)")
    plt.ylabel("Relative output error (%)")
    plt.title("Mean/P95 Error vs Mean Latency Overhead")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_error_mean_p95.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(latency * 100.0, reduction * 100.0, marker="o")
    for x, y, t in zip(latency * 100.0, reduction * 100.0, tau):
        plt.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Mean estimated latency overhead (%)")
    plt.ylabel("Mean output error reduction vs NVFP4 baseline (%)")
    plt.title("Error Reduction vs Latency Overhead")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_error_reduction.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 5.0))
    plt.plot(latency * 100.0, reduction * 100.0, marker="o", label="Mean")
    plt.plot(latency * 100.0, p95_reduction * 100.0, marker="s", label="P95")
    for x, y, t in zip(latency * 100.0, reduction * 100.0, tau):
        plt.annotate(f"{t:.2f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8)
    plt.xlabel("Mean estimated latency overhead (%)")
    plt.ylabel("Output error reduction vs NVFP4 baseline (%)")
    plt.title("Mean/P95 Error Reduction vs Latency Overhead")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "latency_vs_error_reduction_mean_p95.png", dpi=200)
    plt.close()

    fig, ax1 = plt.subplots(figsize=(7.2, 5.0))
    ax1.plot(tau, fallback * 100.0, marker="o", color="#1f77b4", label="Fallback rate")
    ax1.set_xlabel("Zero-ratio threshold tau")
    ax1.set_ylabel("Fallback rate (%)", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(tau, error * 100.0, marker="s", color="#d62728", label="Relative output error")
    ax2.set_ylabel("Mean relative output error (%)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    plt.title("Threshold Sweep")
    fig.tight_layout()
    plt.savefig(out_dir / "threshold_sweep.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/mnt/workspace/Qwen3-0.6B/")
    parser.add_argument("--tensor-dir", default="../tensors/qwen3-0.6B/attn_input")
    parser.add_argument("--out-dir", default="../figures/qwen3-0.6B/fallback_patent")
    parser.add_argument("--num-files", type=int, default=50)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--quant-format", default="nvfp4", choices=["nvfp4", "int4"])
    parser.add_argument(
        "--quantize-weight",
        action="store_true",
        help="Also quantize q_proj weights; fallback if either X or W block exceeds tau.",
    )
    parser.add_argument(
        "--zero-abs-threshold",
        type=float,
        default=1e-5,
        help="Original values below this magnitude are not counted as harmful zeroing.",
    )
    parser.add_argument("--alpha", type=float, default=4.0, help="BF16 cost / FP4 cost")
    parser.add_argument("--thresholds", default="linspace:0:1:21")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--fp32-matmul", action="store_true", help="Use fp32 matmul for simulation stability")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Optional prefix length for quick tests")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    thresholds = parse_thresholds(args.thresholds)

    if args.dtype == "bf16":
        model_dtype = torch.bfloat16
    elif args.dtype == "fp16":
        model_dtype = torch.float16
    else:
        model_dtype = torch.float32

    print(f"Loading model from {args.model_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        device_map=None,
    ).to(args.device)
    model.eval()

    acc = {}
    tensor_dir = Path(args.tensor_dir)

    for file_idx in range(1, args.num_files + 1):
        tensor_path = tensor_dir / f"data_{file_idx}.tensor"
        print(f"[file {file_idx}/{args.num_files}] loading {tensor_path}")
        data = torch.load(tensor_path, weights_only=True, map_location="cpu")
        # data: [layer_num, 1, seq_len, dim]

        for layer in range(args.num_layers):
            x = data[layer].to(device=args.device, dtype=model_dtype)
            if args.max_seq_len is not None:
                x = x[:, : args.max_seq_len, :]

            linear_layer = model.model.layers[layer].self_attn.q_proj
            weight = linear_layer.weight.detach()

            layer_results = evaluate_layer(
                x=x,
                weight=weight,
                thresholds=thresholds,
                block_size=args.block_size,
                alpha=args.alpha,
                use_fp32_matmul=args.fp32_matmul,
                zero_abs_threshold=args.zero_abs_threshold,
                quant_format=args.quant_format,
                quantize_weight=args.quantize_weight,
            )
            merge_results(acc, layer_results)

        rows = finalize_results(acc, args.alpha)
        write_csv(rows, out_dir / "partial_results.csv")
        print(f"  wrote partial results to {out_dir / 'partial_results.csv'}")

    rows = finalize_results(acc, args.alpha)
    write_csv(rows, out_dir / "results.csv")
    plot_results(rows, out_dir)

    print("\nDone.")
    print(f"CSV: {out_dir / 'results.csv'}")
    print(f"Main plot: {out_dir / 'latency_vs_error.png'}")
    print(f"Threshold plot: {out_dir / 'threshold_sweep.png'}")


if __name__ == "__main__":
    main()
