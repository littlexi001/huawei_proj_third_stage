#!/usr/bin/env python3
"""Predict MMLU accuracy from fast Metis residual-MSE proxies.

This script does not run downstream MMLU evaluation. It uses existing
`acc_matrix.csv` files as supervision, samples a small set of anchor
configurations from those matrices, fits a ridge response surface from MSE
features to MMLU accuracy drop, then predicts every ka/kb grid point.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


CHOICES = ["A", "B", "C", "D"]
DEFAULT_ROOT = Path("/mnt/workspace/lym_code/scripts/huawei_proj_third_stage")
DEFAULT_MODEL = Path("/mnt/workspace/Qwen3-8B")
DEFAULT_OUTPUT = DEFAULT_ROOT / "outputs" / "mse_acc_prediction_qwen3_8b"
DEFAULT_DATA_CANDIDATES = [
    DEFAULT_ROOT / "quant_error-main" / "quant_error-main" / "data" / "eval" / "mmlu_validation_1531.jsonl",
    DEFAULT_ROOT / "data" / "eval" / "mmlu_validation_1531.jsonl",
]
DEFAULT_ACC = {
    "hif4": DEFAULT_ROOT / "outputs" / "qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif4" / "plots" / "acc_matrix.csv",
    "hif8": DEFAULT_ROOT / "outputs" / "qwen3_svd_resid_nvfp4_mmlu_grid_8B_hif8_cuda" / "plots" / "acc_matrix.csv",
}


@dataclass
class AccMatrix:
    ka_values: list[int]
    kb_values: list[int]
    values: dict[tuple[int, int], float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--formats", default="hif4,hif8")
    parser.add_argument("--hif4-acc-csv", default=str(DEFAULT_ACC["hif4"]))
    parser.add_argument("--hif8-acc-csv", default=str(DEFAULT_ACC["hif8"]))
    parser.add_argument("--data", default="", help="Calibration JSONL for activation MSE; default tries local MMLU files")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--rank-candidates", default="0,5,10,15,20,25,30,40,50,60,80,100")
    parser.add_argument("--anchor-pairs", default="0:0,0:100,100:0,20:20,40:40,60:60,80:80,100:100,20:80,80:20,40:80,80:40")
    parser.add_argument("--random-anchors", type=int, default=0, help="Add N deterministic random anchors")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--baseline-acc", type=float, default=None, help="Default: max acc in each format matrix")
    parser.add_argument("--max-acc-drop", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=128)
    parser.add_argument("--max-act-rows", type=int, default=512)
    parser.add_argument("--max-modules", type=int, default=0, help="0 means all Linear modules except lm_head")
    parser.add_argument("--module-name-contains", default="")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--weight-only", action="store_true", help="Skip activation collection and fit from weight MSE only")
    parser.add_argument("--proxy-cache", default="", help="Optional JSON cache path for computed MSE features")
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--ka-cost", type=float, default=1.0)
    parser.add_argument("--kb-cost", type=float, default=1.0)
    return parser.parse_args()


def dtype_from_arg(arg: str) -> torch.dtype | str:
    if arg == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[arg]


def resolve_data_path(arg: str) -> Path | None:
    if arg:
        return Path(arg)
    for path in DEFAULT_DATA_CANDIDATES:
        if path.exists():
            return path
    return None


def load_acc_matrix(path: str) -> AccMatrix:
    values: dict[tuple[int, int], float] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ka_values = [int(x) for x in header[1:]]
        kb_values = []
        for row in reader:
            if not row:
                continue
            kb = int(row[0])
            kb_values.append(kb)
            for ka, cell in zip(ka_values, row[1:]):
                values[(ka, kb)] = float(cell)
    return AccMatrix(ka_values=ka_values, kb_values=kb_values, values=values)


def parse_pairs(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        left, right = item.replace("/", ":").split(":", 1)
        out.append((int(left), int(right)))
    return out


def mmlu_prompt(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if not isinstance(choices, list):
        choices = [row.get(c, "") for c in CHOICES]
    lines = [f"Question: {row['question']}"]
    for label, choice in zip(CHOICES, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def load_prompts(path: Path, limit: int, seed: int) -> list[str]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return [mmlu_prompt(row) for row in rows[: min(limit, len(rows))]]


def selected_linear_modules(model: nn.Module, contains: str, max_modules: int) -> list[tuple[str, nn.Linear]]:
    filters = [part.strip() for part in contains.split(",") if part.strip()]
    modules = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear) or name.endswith("lm_head"):
            continue
        if filters and not any(token in name for token in filters):
            continue
        modules.append((name, module))
        if max_modules and len(modules) >= max_modules:
            break
    return modules


def quantize_hif8(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    x_unsigned = x.abs()
    sign = x.sign()
    exponent = torch.floor(torch.log2(x_unsigned + 2.0**-45))
    abs_exponent = exponent.abs()
    mantissa_bits = torch.zeros_like(abs_exponent)
    mantissa_bits[abs_exponent <= 15] = 1
    mantissa_bits[abs_exponent <= 7] = 2
    mantissa_bits[abs_exponent <= 3] = 3
    out = (
        torch.floor(x_unsigned * torch.exp2(-exponent + mantissa_bits) + 0.5)
        * torch.exp2(exponent - mantissa_bits)
        * sign
    )
    out[x_unsigned < 2.0**-23] = 0.0
    out[x_unsigned >= 2.0**15 * 1.25] = torch.inf * sign[x_unsigned >= 2.0**15 * 1.25]
    return out


def quantize_hif4(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    original_shape = x.shape
    last_dim = original_shape[-1]
    pad = (-last_dim) % 64
    if pad:
        x = torch.nn.functional.pad(x, (0, pad), value=0.0)

    grouped = x.unflatten(-1, (-1, 8, 2, 4))
    x_unsigned = grouped.abs()
    sign = grouped.sign()
    max_level3 = x_unsigned.amax(dim=-1, keepdim=True)
    max_level2 = max_level3.amax(dim=-2, keepdim=True)
    max_level1 = max_level2.amax(dim=-3, keepdim=True)
    div7 = torch.ones_like(max_level1).div_(7.0).to(torch.bfloat16).float()
    scale_factor = (max_level1 * div7).to(torch.bfloat16).float().clamp(min=2.0**-48, max=49152.0)
    scale_exp = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(7 - scale_exp)) * torch.exp2(scale_exp - 7)
    scale_exp = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2 - scale_exp)) * torch.exp2(scale_exp - 2)
    reciprocal_scale = (1.0 / scale_factor).to(torch.bfloat16).float()
    scale_level2 = torch.exp2(torch.floor((max_level2 * reciprocal_scale).clamp(0, 4) / 4))
    scale_level3 = torch.exp2(torch.floor((max_level3 * reciprocal_scale / scale_level2).clamp(0, 2) / 2))
    mantissa = x_unsigned / scale_level2 / scale_level3 * reciprocal_scale
    mantissa = torch.floor(mantissa * 4.0 + 0.5) / 4.0
    mantissa = mantissa.clamp(max=1.75)
    out = sign * mantissa * scale_level2 * scale_level3 * scale_factor
    out = out.flatten(-4, -1)
    if pad:
        out = out[..., :last_dim]
    return out.reshape(original_shape)


def get_quantizer(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "hif4":
        return quantize_hif4
    if name == "hif8":
        return quantize_hif8
    raise ValueError(f"Unsupported format: {name}")


def residual_quant_scores(
    matrix: torch.Tensor,
    ranks: list[int],
    quantizer: Callable[[torch.Tensor], torch.Tensor],
) -> dict[int, float]:
    matrix = matrix.float()
    denom = torch.linalg.norm(matrix).square().clamp_min(1e-12)
    max_rank = min(matrix.shape)
    effective_ranks = sorted({min(max(rank, 0), max_rank) for rank in ranks})
    if any(rank > 0 for rank in effective_ranks):
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    scores_by_effective_rank = {}
    for rank in effective_ranks:
        if rank == 0:
            residual = matrix
        else:
            head = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            residual = matrix - head
        q_residual = quantizer(residual)
        err = torch.linalg.norm(q_residual - residual).square()
        scores_by_effective_rank[rank] = float((err / denom).cpu())
    return {rank: scores_by_effective_rank[min(max(rank, 0), max_rank)] for rank in ranks}


def collect_activations(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    modules: list[tuple[str, nn.Linear]],
    max_rows: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    samples: dict[str, list[torch.Tensor]] = {name: [] for name, _ in modules}
    row_counts = {name: 0 for name, _ in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if row_counts[name] >= max_rows:
                return
            x = inputs[0].detach().float().cpu().reshape(-1, inputs[0].shape[-1])
            need = max_rows - row_counts[name]
            if x.shape[0] > need:
                x = x[:need]
            samples[name].append(x)
            row_counts[name] += x.shape[0]
        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            model(**encoded, use_cache=False)
            if all(count >= max_rows for count in row_counts.values()):
                break

    for handle in handles:
        handle.remove()
    return {name: torch.cat(chunks, dim=0) for name, chunks in samples.items() if chunks}


def compute_proxy_features(args: argparse.Namespace, formats: list[str], ranks: list[int]) -> dict[str, Any]:
    cache_path = Path(args.proxy_cache) if args.proxy_cache else Path(args.output_dir) / "proxy_mse_cache.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8-sig"))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_path = resolve_data_path(args.data)
    if not args.weight_only and data_path is None:
        raise RuntimeError("No calibration data found. Pass --data or use --weight-only.")

    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    modules = selected_linear_modules(model, args.module_name_contains, args.max_modules)
    if not modules:
        raise RuntimeError("No Linear modules selected")

    activations = {}
    if not args.weight_only:
        prompts = load_prompts(data_path, args.limit, args.seed)
        activations = collect_activations(model, tokenizer, prompts, modules, args.max_act_rows, device)

    proxy: dict[str, Any] = {"ranks": ranks, "formats": {}}
    for fmt in formats:
        print(f"[proxy] scoring {fmt}", flush=True)
        quantizer = get_quantizer(fmt)
        module_rows = []
        for index, (name, module) in enumerate(modules, start=1):
            print(f"[proxy] {fmt} module {index}/{len(modules)} {name}", flush=True)
            weight_scores = residual_quant_scores(module.weight.detach().float().cpu(), ranks, quantizer)
            if args.weight_only:
                act_scores = {rank: 0.0 for rank in ranks}
            else:
                x = activations.get(name)
                if x is None:
                    continue
                act_scores = residual_quant_scores(x, ranks, quantizer)
            module_rows.append({"name": name, "weight_scores": weight_scores, "act_scores": act_scores})
        proxy["formats"][fmt] = {"module_stats": module_rows}

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(proxy, indent=2), encoding="utf-8")
    return proxy


def aggregate_scores(module_stats: list[dict[str, Any]]) -> tuple[dict[int, float], dict[int, float]]:
    act: dict[int, list[float]] = {}
    weight: dict[int, list[float]] = {}
    for stats in module_stats:
        for rank, value in stats["act_scores"].items():
            act.setdefault(int(rank), []).append(float(value))
        for rank, value in stats["weight_scores"].items():
            weight.setdefault(int(rank), []).append(float(value))
    return (
        {rank: float(np.mean(values)) for rank, values in act.items()},
        {rank: float(np.mean(values)) for rank, values in weight.items()},
    )


def zscore(x: np.ndarray, mean: float | None = None, std: float | None = None) -> tuple[np.ndarray, float, float]:
    if mean is None:
        mean = float(x.mean())
    if std is None:
        std = float(x.std())
    if std < 1e-12:
        return x * 0.0, mean, std
    return (x - mean) / std, mean, std


def build_feature_matrix(
    rows: list[dict[str, float]],
    stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    stats = {} if stats is None else dict(stats)
    act, mean, std = zscore(np.array([row["activation_mse"] for row in rows]), *stats.get("activation_mse", (None, None)))
    stats["activation_mse"] = (mean, std)
    weight, mean, std = zscore(np.array([row["weight_mse"] for row in rows]), *stats.get("weight_mse", (None, None)))
    stats["weight_mse"] = (mean, std)
    proxy, mean, std = zscore(np.array([row["proxy_mse"] for row in rows]), *stats.get("proxy_mse", (None, None)))
    stats["proxy_mse"] = (mean, std)
    ka, mean, std = zscore(np.array([row["ka"] for row in rows], dtype=np.float64), *stats.get("ka", (None, None)))
    stats["ka"] = (mean, std)
    kb, mean, std = zscore(np.array([row["kb"] for row in rows], dtype=np.float64), *stats.get("kb", (None, None)))
    stats["kb"] = (mean, std)
    features = np.column_stack([np.ones(len(rows)), act, weight, act * weight, proxy, ka, kb, ka * kb])
    return features, stats


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_results(out_dir: Path, fmt: str, rows: list[dict[str, Any]], matrix: AccMatrix) -> None:
    import matplotlib.pyplot as plt

    ka_values = matrix.ka_values
    kb_values = matrix.kb_values
    actual = np.array([[next(r["actual_acc"] for r in rows if r["ka"] == ka and r["kb"] == kb) for ka in ka_values] for kb in kb_values])
    pred = np.array([[next(r["pred_acc"] for r in rows if r["ka"] == ka and r["kb"] == kb) for ka in ka_values] for kb in kb_values])
    error = pred - actual

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, data, title in zip(axes, [actual, pred, error], ["Actual acc", "Predicted acc", "Prediction error"]):
        image = ax.imshow(data, aspect="auto", origin="lower")
        ax.set_title(f"{fmt} {title}")
        ax.set_xticks(range(len(ka_values)), ka_values, rotation=45)
        ax.set_yticks(range(len(kb_values)), kb_values)
        ax.set_xlabel("ka / xrank")
        ax.set_ylabel("kb / wrank")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_dir / f"{fmt}_acc_prediction_heatmaps.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual.reshape(-1), pred.reshape(-1), s=18)
    lo = min(actual.min(), pred.min())
    hi = max(actual.max(), pred.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("Actual acc")
    ax.set_ylabel("Predicted acc")
    ax.set_title(f"{fmt} predicted vs actual")
    fig.savefig(out_dir / f"{fmt}_pred_vs_actual.png", dpi=200)
    plt.close(fig)


def choose_anchors(matrix: AccMatrix, base_pairs: list[tuple[int, int]], random_n: int, seed: int) -> list[tuple[int, int]]:
    valid = set(matrix.values)
    anchors = [pair for pair in base_pairs if pair in valid]
    remaining = sorted(valid - set(anchors))
    rng = random.Random(seed)
    rng.shuffle(remaining)
    anchors.extend(remaining[:random_n])
    return sorted(set(anchors), key=lambda pair: (pair[0] + pair[1], pair[0], pair[1]))


def run_format(
    fmt: str,
    acc_csv: str,
    proxy_data: dict[str, Any],
    args: argparse.Namespace,
    ranks: list[int],
    output_dir: Path,
) -> dict[str, Any]:
    matrix = load_acc_matrix(acc_csv)
    baseline_acc = args.baseline_acc if args.baseline_acc is not None else max(matrix.values.values())
    module_stats = proxy_data["formats"][fmt]["module_stats"]
    act_scores, weight_scores = aggregate_scores(module_stats)

    rows = []
    for kb in matrix.kb_values:
        for ka in matrix.ka_values:
            if ka not in ranks or kb not in ranks:
                continue
            actual_acc = matrix.values[(ka, kb)]
            activation_mse = act_scores.get(ka, 0.0)
            weight_mse = weight_scores.get(kb, 0.0)
            rows.append(
                {
                    "format": fmt,
                    "ka": ka,
                    "kb": kb,
                    "activation_mse": activation_mse,
                    "weight_mse": weight_mse,
                    "proxy_mse": activation_mse + weight_mse,
                    "actual_acc": actual_acc,
                    "actual_acc_drop": baseline_acc - actual_acc,
                    "rank_cost": args.ka_cost * ka + args.kb_cost * kb,
                }
            )

    anchors = choose_anchors(matrix, parse_pairs(args.anchor_pairs), args.random_anchors, args.seed)
    anchor_set = set(anchors)
    train_indices = [index for index, row in enumerate(rows) if (row["ka"], row["kb"]) in anchor_set]
    if len(train_indices) < 3:
        raise RuntimeError(f"{fmt}: need at least three valid anchor points")

    x_all, feature_stats = build_feature_matrix(rows)
    x_train = x_all[train_indices]
    y_train = np.array([rows[index]["actual_acc_drop"] for index in train_indices], dtype=np.float64)
    beta = fit_ridge(x_train, y_train, args.ridge)
    predictions = x_all @ beta

    for row, pred_drop in zip(rows, predictions):
        row["pred_acc_drop"] = float(pred_drop)
        row["pred_acc"] = float(baseline_acc - pred_drop)
        row["pred_error"] = float(row["pred_acc"] - row["actual_acc"])
        row["abs_pred_error"] = abs(row["pred_error"])
        row["feasible_pred"] = row["pred_acc_drop"] <= args.max_acc_drop
        row["feasible_actual"] = row["actual_acc_drop"] <= args.max_acc_drop
        row["is_anchor"] = (row["ka"], row["kb"]) in anchor_set

    feasible = [row for row in rows if row["feasible_pred"]]
    feasible.sort(key=lambda row: (row["rank_cost"], row["pred_acc_drop"], row["ka"], row["kb"]))
    errors = np.array([row["pred_error"] for row in rows], dtype=np.float64)
    abs_errors = np.abs(errors)
    best = feasible[0] if feasible else None
    summary = {
        "format": fmt,
        "acc_csv": acc_csv,
        "baseline_acc": baseline_acc,
        "max_acc_drop": args.max_acc_drop,
        "num_points": len(rows),
        "num_anchors": len(train_indices),
        "anchors": anchors,
        "feature_stats": {key: {"mean": value[0], "std": value[1]} for key, value in feature_stats.items()},
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "max_abs_error": float(abs_errors.max()),
        "num_pred_feasible": len(feasible),
        "num_actual_feasible": sum(1 for row in rows if row["feasible_actual"]),
        "best_min_cost_pred_feasible": best,
    }

    format_dir = output_dir / fmt
    format_dir.mkdir(parents=True, exist_ok=True)
    write_rows(format_dir / "all_predictions.csv", rows)
    write_rows(format_dir / "pred_feasible_points.csv", feasible)
    (format_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_results(format_dir, fmt, rows, matrix)
    return summary


def main() -> None:
    args = parse_args()
    formats = [part.strip() for part in args.formats.split(",") if part.strip()]
    ranks = [int(part) for part in args.rank_candidates.split(",") if part.strip()]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_data = compute_proxy_features(args, formats, ranks)
    acc_paths = {"hif4": args.hif4_acc_csv, "hif8": args.hif8_acc_csv}
    summaries = []
    for fmt in formats:
        summaries.append(run_format(fmt, acc_paths[fmt], proxy_data, args, ranks, output_dir))

    combined = {"model_path": args.model_path, "output_dir": str(output_dir), "summaries": summaries}
    (output_dir / "combined_summary.json").write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
