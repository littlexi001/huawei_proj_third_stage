#!/usr/bin/env python3
"""Calibration-driven ka/kp search prototype for Adaptive-Metis.

This script intentionally separates the rank-search proxy from full task
evaluation. It loads any HuggingFace CausalLM model path, builds prompts from a
small MMLU/GSM8K JSONL subset, collects representative linear-layer inputs, and
scores candidate (ka, kp) pairs by residual quantization risk.

HIF4 and HIF8 follow the PyTorch reference behavior under `Metis/`:

* HIF4 (`hifx4`) uses a 64 -> 8 -> 4 hierarchical shared-scale layout.
* HIF8 uses per-value tapered precision, with 3/2/1/0 mantissa bits selected
  from the value's exponent range.
* FP8 uses tensor-wise scaled E4M3 quantize-dequantize.
* NVFP8 uses block-wise scaled E4M3 quantize-dequantize, with the block scales
  themselves quantized to E4M3.

Examples:
  python3 scripts/adaptive_metis_rank_search.py \
    --model-path /Users/bytedance/kv_cache/fdong/Qwen3-0.6B \
    --data data/eval/mmlu_validation_128.jsonl \
    --task mmlu --format hif8 --limit 64 --max-modules 16
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn


CHOICES = ["A", "B", "C", "D"]


@dataclass
class ModuleStats:
    name: str
    weight_scores: dict[int, float]
    act_scores: dict[int, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True, help="JSONL exported by download_eval_data.py")
    parser.add_argument("--task", choices=["mmlu", "gsm8k"], required=True)
    parser.add_argument("--format", choices=["hif4", "hif8", "fp8", "nvfp8"], default="hif8")
    parser.add_argument("--rank-candidates", default="0,5,10,15,20,25,30,40,50,60,80,100")
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--max-act-rows", type=int, default=512)
    parser.add_argument("--max-modules", type=int, default=0, help="0 means all linear modules")
    parser.add_argument("--module-name-contains", default="", help="Optional comma-separated name filters")
    parser.add_argument("--output", default="data/results/adaptive_metis_rank_search.json")
    return parser.parse_args()


def load_jsonl(path: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    return rows[: min(limit, len(rows))]


def mmlu_prompt(row: dict[str, Any]) -> str:
    choices = row.get("choices")
    if not isinstance(choices, list):
        choices = [row.get(c, "") for c in CHOICES]
    lines = [f"Question: {row['question']}"]
    for label, choice in zip(CHOICES, choices):
        lines.append(f"{label}. {choice}")
    lines.append("Answer:")
    return "\n".join(lines)


def gsm8k_prompt(row: dict[str, Any]) -> str:
    return f"Question: {row['question']}\nAnswer:"


def build_prompts(rows: list[dict[str, Any]], task: str) -> list[str]:
    if task == "mmlu":
        return [mmlu_prompt(row) for row in rows]
    return [gsm8k_prompt(row) for row in rows]


def dtype_from_arg(arg: str) -> torch.dtype | str:
    if arg == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[arg]


def linear_modules(model: nn.Module, contains: str, max_modules: int) -> list[tuple[str, nn.Linear]]:
    filters = [x for x in (part.strip() for part in contains.split(",")) if x]
    out: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if name.endswith("lm_head"):
            continue
        if filters and not any(token in name for token in filters):
            continue
        out.append((name, module))
        if max_modules and len(out) >= max_modules:
            break
    return out


def quantize_hif8(x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize using the Metis HIF8 tapered-float definition."""
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


def _fp8_e4m3_qdq(x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize values already scaled into E4M3's representable range."""
    if hasattr(torch, "float8_e4m3fn"):
        return x.to(dtype=torch.float8_e4m3fn).to(dtype=torch.float32)
    # Software fallback for older PyTorch builds. It approximates finite E4M3
    # with exponent range [-6, 8] and three mantissa bits.
    sign = x.sign()
    x_abs = x.abs().clamp(max=448.0)
    tiny = x_abs < 2.0**-9
    exponent = torch.floor(torch.log2(x_abs.clamp_min(2.0**-9))).clamp(-6, 8)
    mantissa = torch.round(x_abs * torch.exp2(-exponent) * 8.0) / 8.0
    rounded = mantissa * torch.exp2(exponent)
    rounded[tiny] = 0.0
    return sign * rounded


def quantize_fp8(x: torch.Tensor) -> torch.Tensor:
    """Tensor-wise scaled FP8 E4M3 quantize-dequantize."""
    x = x.float()
    scale = x.abs().max().clamp_min(1e-12) / 448.0
    return _fp8_e4m3_qdq(x / scale) * scale


def quantize_nvfp8(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """Block-wise scaled FP8 E4M3 quantize-dequantize with FP8-quantized scales.

    This follows the NV-style scale path used elsewhere in this repository for
    NVFP block formats: per-block scales are quantized through E4M3 using one
    tensor-wise meta-scale before values are quantized by the resulting scales.
    """
    x = x.float()
    original_shape = x.shape
    last_dim = original_shape[-1]
    pad = (-last_dim) % block_size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad), value=0.0)

    grouped = x.reshape(-1, x.shape[-1]).unflatten(-1, (-1, block_size))
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 448.0
    scale_meta = scales.abs().max().clamp_min(1e-12) / 448.0
    scales = _fp8_e4m3_qdq(scales / scale_meta) * scale_meta
    out = _fp8_e4m3_qdq(grouped / scales) * scales
    out = out.flatten(-2, -1).reshape(*x.shape)
    if pad:
        out = out[..., :last_dim]
    return out.reshape(original_shape)


def quantize_hif4(x: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize using Metis HIF4's 64 -> 8 -> 4 hierarchy.

    Quantization is applied along the last tensor dimension, matching
    `QType("hifx4").dim(-1)` in the reference implementation.
    """
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
    scale_factor = (max_level1 * div7).to(torch.bfloat16).float()
    scale_factor = scale_factor.clamp(min=2.0**-48, max=49152.0)

    # BF16 rounding followed by E6M2 rounding, as in quant_hifx.
    scale_exp = torch.floor(torch.log2(scale_factor))
    scale_factor = (
        torch.round(scale_factor * torch.exp2(7 - scale_exp))
        * torch.exp2(scale_exp - 7)
    )
    scale_exp = torch.floor(torch.log2(scale_factor))
    scale_factor = (
        torch.round(scale_factor * torch.exp2(2 - scale_exp))
        * torch.exp2(scale_exp - 2)
    )

    reciprocal_scale = (1.0 / scale_factor).to(torch.bfloat16).float()
    scale_level2 = torch.exp2(torch.floor((max_level2 * reciprocal_scale).clamp(0, 4) / 4))
    scale_level3 = torch.exp2(
        torch.floor((max_level3 * reciprocal_scale / scale_level2).clamp(0, 2) / 2)
    )

    # HIF4 has sign + 3 magnitude precision bits in the reference (`man_bits=3`).
    mantissa = x_unsigned / scale_level2 / scale_level3 * reciprocal_scale
    mantissa = torch.floor(mantissa * 4.0 + 0.5) / 4.0
    mantissa = mantissa.clamp(max=1.75)

    out = sign * mantissa * scale_level2 * scale_level3 * scale_factor
    out = out.flatten(-4, -1)
    if pad:
        out = out[..., :last_dim]
    return out.reshape(original_shape)


def get_quantizer(format_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if format_name == "hif4":
        return quantize_hif4
    if format_name == "hif8":
        return quantize_hif8
    if format_name == "fp8":
        return quantize_fp8
    if format_name == "nvfp8":
        return quantize_nvfp8
    raise ValueError(f"Unsupported format: {format_name}")


def residual_quant_scores(
    a: torch.Tensor,
    ranks: list[int],
    quantizer: Callable[[torch.Tensor], torch.Tensor],
) -> dict[int, float]:
    """Score all ranks after one SVD of the input matrix."""
    a = a.float()
    denom = torch.linalg.norm(a).square().clamp_min(1e-12)
    max_rank = min(a.shape)
    effective_ranks = sorted({min(max(rank, 0), max_rank) for rank in ranks})
    need_svd = any(rank > 0 for rank in effective_ranks)
    if need_svd:
        u, s, vh = torch.linalg.svd(a, full_matrices=False)

    scores_by_effective_rank: dict[int, float] = {}
    for rank in effective_ranks:
        if rank == 0:
            residual = a
        else:
            head = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
            residual = a - head
        q_residual = quantizer(residual)
        err = torch.linalg.norm(q_residual - residual).square()
        scores_by_effective_rank[rank] = float((err / denom).cpu())

    return {
        rank: scores_by_effective_rank[min(max(rank, 0), max_rank)]
        for rank in ranks
    }


def collect_activation_samples(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    modules: list[tuple[str, nn.Linear]],
    max_length: int,
    max_rows: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    samples: dict[str, list[torch.Tensor]] = {name: [] for name, _ in modules}
    per_module_rows: dict[str, int] = {name: 0 for name, _ in modules}
    handles = []

    def make_hook(name: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...], _output: torch.Tensor) -> None:
            if per_module_rows[name] >= max_rows:
                return
            x = inputs[0].detach().float().cpu()
            x = x.reshape(-1, x.shape[-1])
            need = max_rows - per_module_rows[name]
            if x.shape[0] > need:
                x = x[:need]
            samples[name].append(x)
            per_module_rows[name] += x.shape[0]
        return hook

    for name, module in modules:
        handles.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            model(**encoded)
            if all(rows >= max_rows for rows in per_module_rows.values()):
                break

    for handle in handles:
        handle.remove()

    out: dict[str, torch.Tensor] = {}
    for name, chunks in samples.items():
        if chunks:
            out[name] = torch.cat(chunks, dim=0)
    return out


def summarize_scores(module_stats: list[ModuleStats], rank_candidates: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for ka in rank_candidates:
        for kp in rank_candidates:
            vals = []
            for stats in module_stats:
                if ka in stats.act_scores and kp in stats.weight_scores:
                    vals.append(stats.act_scores[ka] + stats.weight_scores[kp])
            if not vals:
                continue
            score = sum(vals) / len(vals)
            rows.append({"ka": ka, "kp": kp, "proxy_score": score, "num_modules": len(vals)})
    rows.sort(key=lambda x: (x["proxy_score"], x["ka"] + x["kp"]))
    return rows


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    quantizer = get_quantizer(args.format)
    ranks = [int(x) for x in args.rank_candidates.split(",") if x.strip()]
    rows = load_jsonl(args.data, args.limit, args.seed)
    prompts = build_prompts(rows, args.task)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu")
    if args.device != "auto":
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

    modules = linear_modules(model, args.module_name_contains, args.max_modules)
    if not modules:
        raise RuntimeError("No Linear modules selected")
    print(f"Selected {len(modules)} linear modules")

    activations = collect_activation_samples(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        modules=modules,
        max_length=args.max_length,
        max_rows=args.max_act_rows,
        device=device,
    )

    module_stats: list[ModuleStats] = []
    for idx, (name, module) in enumerate(modules, start=1):
        print(f"[{idx}/{len(modules)}] scoring {name}")
        w = module.weight.detach().float().cpu()
        x = activations.get(name)
        if x is None:
            continue
        weight_scores = residual_quant_scores(w, ranks=ranks, quantizer=quantizer)
        act_scores = residual_quant_scores(x, ranks=ranks, quantizer=quantizer)
        module_stats.append(ModuleStats(name=name, weight_scores=weight_scores, act_scores=act_scores))

    candidates = summarize_scores(module_stats, ranks)
    result = {
        "model_path": args.model_path,
        "data": args.data,
        "task": args.task,
        "format": args.format,
        "quantizer": "Metis reference-compatible HIF4/HIF8 plus FP8/NVFP8 PyTorch implementation",
        "rank_candidates": ranks,
        "num_examples": len(rows),
        "num_modules": len(module_stats),
        "top_candidates": candidates[:25],
        "candidates": candidates,
        "module_stats": [
            {
                "name": stats.name,
                "weight_scores": stats.weight_scores,
                "act_scores": stats.act_scores,
            }
            for stats in module_stats
        ],
    }
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")
    print("Top candidates:")
    for row in candidates[:10]:
        print(row)


if __name__ == "__main__":
    main()
