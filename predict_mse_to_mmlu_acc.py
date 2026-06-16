#!/usr/bin/env python3
"""Predict MMLU accuracy from output-level hidden-state MSE.

This script actually runs the compressed model (SVD + quantization) on calibration
data to compute the last-layer hidden-state MSE for every (ka, kb) grid point.
It then uses a small set of anchor configurations (with known MMLU accuracy from
acc_matrix.csv) to fit a ridge regression model: output_mse → MMLU accuracy drop.

Workflow:
  1. Load model, run forward pass → collect baseline (uncompressed) hidden states
  2. For each (ka, kb) grid point:
     a. SVD-decompose weights at rank=kb, quantize main & residual
     b. Run forward pass (activations decomposed at rank=ka on the fly)
     c. Compute last-hidden MSE vs baseline
     NB: This is SLOW (~144 model loads/format), cached to proxy_mse_cache.json
  3. From acc_matrix.csv, select anchor (ka,kb) points as training labels
  4. Fit ridge regression: output_mse, ka, kb → accuracy_drop
  5. Predict accuracy for all grid points, output feasible (ka,kb) configurations
"""

from __future__ import annotations

import argparse
import csv
import gc
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
    parser.add_argument("--proxy-device", default="cuda", help="Device used for MSE/SVD scoring; use cuda for speed, cpu to save VRAM")
    parser.add_argument(
        "--svd-method",
        choices=["lowrank", "exact", "randomized", "full"],
        default="lowrank",
        help="'lowrank'/'randomized' use torch.svd_lowrank; 'exact'/'full' use torch.linalg.svd",
    )
    parser.add_argument("--svd-oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=2)
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


# --- SVD low-rank approximation utilities ---

def _disabled_autocast(device_type: str):
    if device_type == "cuda" and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=False)
    from contextlib import nullcontext

    return nullcontext()


@torch.no_grad()
def low_rank_approx(x2d: torch.Tensor, rank: int, method: str) -> torch.Tensor:
    """Compute rank-r SVD low-rank approximation of a 2D tensor.

    Args:
        method: 'exact' for torch.linalg.svd, 'lowrank' for torch.svd_lowrank (randomized).
    """
    with _disabled_autocast(x2d.device.type):
        x32 = x2d.float().contiguous()
        r = min(int(rank), min(x32.shape))
        if r <= 0:
            return torch.zeros_like(x32)
        if r >= min(x32.shape):
            return x32.clone()
        if method in {"exact", "full"}:
            u, s, vh = torch.linalg.svd(x32, full_matrices=False)
            return (u[:, :r] * s[:r]) @ vh[:r, :]
        if method in {"lowrank", "randomized"}:
            u, s, v = torch.svd_lowrank(x32, q=r, niter=2)
            return (u[:, :r] * s[:r]) @ v[:, :r].T
        raise ValueError(f"Unknown SVD method: {method}")


# --- SVD + Quantize Residual Linear Layer ---

TARGET_LINEAR_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


class SVDRQuantLinear(nn.Module):
    """Linear layer with SVD low-rank decomposition + residual quantization.

    Decomposes weight W into main (low-rank) W_m and residual W_r:
        W = W_m + W_r,   W_m = low_rank_approx(W, weight_rank)

    Decomposes input activation X into main (low-rank) X_m and residual X_r:
        X = X_m + X_r,   X_m = low_rank_approx(X, act_rank)

    Forward pass computes four cross terms:
        out = X_m @ W_m^T + X_mq @ W_rq^T + X_rq @ W_mq^T + X_rq @ W_rq^T
    where _q suffix denotes quantized versions.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        weight_rank: int,
        act_rank: int,
        quantizer: Callable[[torch.Tensor], torch.Tensor],
        svd_method: str,
        compute_dtype: torch.dtype,
    ):
        super().__init__()
        self.weight_rank = int(weight_rank)
        self.act_rank = int(act_rank)
        self.quantizer = quantizer
        self.svd_method = svd_method

        device = base_linear.weight.device
        if device.type == "cpu" and compute_dtype in (torch.float16, torch.bfloat16):
            compute_dtype = torch.float32
        self.compute_dtype = compute_dtype
        self.store_dtype = compute_dtype if isinstance(compute_dtype, torch.dtype) else torch.float16
        self.out_features = base_linear.out_features

        w_fp = base_linear.weight.detach().float()
        if self.weight_rank <= 0:
            w_rq = quantizer(w_fp)
            self.register_buffer("weight_main", None, persistent=False)
            self.register_buffer("weight_main_q", None, persistent=False)
            self.register_buffer("weight_resid_q", w_rq.to(dtype=self.store_dtype, device=device), persistent=True)
        elif self.weight_rank >= min(w_fp.shape):
            w_mq = quantizer(w_fp)
            self.register_buffer("weight_main", w_fp.to(dtype=self.store_dtype, device=device), persistent=True)
            self.register_buffer("weight_main_q", w_mq.to(dtype=self.store_dtype, device=device), persistent=True)
            self.register_buffer("weight_resid_q", None, persistent=False)
        else:
            w_m = low_rank_approx(w_fp, self.weight_rank, svd_method)
            w_r = w_fp - w_m
            w_mq = quantizer(w_m)
            w_rq = quantizer(w_r)
            self.register_buffer("weight_main", w_m.to(dtype=self.store_dtype, device=device), persistent=True)
            self.register_buffer("weight_main_q", w_mq.to(dtype=self.store_dtype, device=device), persistent=True)
            self.register_buffer("weight_resid_q", w_rq.to(dtype=self.store_dtype, device=device), persistent=True)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base_linear.bias.detach().to(dtype=self.store_dtype, device=device), persistent=True)
        del w_fp
        if torch.cuda.is_available() and device.type == "cuda":
            torch.cuda.empty_cache()

    @torch.no_grad()
    def _decompose_activation(self, x: torch.Tensor):
        orig_shape = x.shape
        hidden = orig_shape[-1]
        x_fp = x.reshape(-1, hidden).float()
        if self.act_rank <= 0:
            x_m = torch.zeros_like(x_fp)
            x_r = x_fp
        elif self.act_rank >= min(x_fp.shape):
            x_m = x_fp
            x_r = torch.zeros_like(x_fp)
        else:
            x_m = low_rank_approx(x_fp, self.act_rank, self.svd_method)
            x_r = x_fp - x_m

        x_mq = self.quantizer(x_m)
        x_rq = self.quantizer(x_r)
        return (
            x_m.reshape(orig_shape).to(dtype=self.compute_dtype),
            x_mq.reshape(orig_shape).to(dtype=self.compute_dtype),
            x_rq.reshape(orig_shape).to(dtype=self.compute_dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m, x_mq, x_rq = self._decompose_activation(x)
        bias = self.bias.to(dtype=self.compute_dtype) if self.bias is not None else None

        out = x.new_zeros((*x.shape[:-1], self.out_features), dtype=self.compute_dtype)
        if bias is not None:
            out = out + bias
        if self.weight_main is not None:
            w_m = self.weight_main.to(dtype=self.compute_dtype)
            out = out + torch.nn.functional.linear(x_m, w_m)
        if self.weight_resid_q is not None:
            w_rq = self.weight_resid_q.to(dtype=self.compute_dtype)
            out = out + torch.nn.functional.linear(x_mq, w_rq)
            out = out + torch.nn.functional.linear(x_rq, w_rq)
        if self.weight_main_q is not None:
            w_mq = self.weight_main_q.to(dtype=self.compute_dtype)
            out = out + torch.nn.functional.linear(x_rq, w_mq)
        return out


def patch_model_linears(
    model: nn.Module,
    weight_rank: int,
    act_rank: int,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
    svd_method: str,
    compute_dtype: torch.dtype,
    module_name_contains: str,
) -> list[str]:
    """Replace target nn.Linear layers with SVDRQuantLinear."""
    filters = [part.strip() for part in module_name_contains.split(",") if part.strip()]
    replaced = []

    def _get_parent(root: nn.Module, module_name: str):
        parts = module_name.split(".")
        parent = root
        for p in parts[:-1]:
            parent = getattr(parent, p)
        return parent, parts[-1]

    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module_name.endswith("lm_head"):
            continue
        if not any(module_name.endswith(suf) for suf in TARGET_LINEAR_SUFFIXES):
            continue
        if filters and not any(token in module_name for token in filters):
            continue
        parent, child_name = _get_parent(model, module_name)
        setattr(
            parent,
            child_name,
            SVDRQuantLinear(
                module, weight_rank, act_rank, quantizer, svd_method, compute_dtype,
            ),
        )
        replaced.append(module_name)
    return replaced


# --- Output hidden-state collection & MSE computation ---

@torch.no_grad()
def collect_output_hidden(
    model: nn.Module,
    tokenizer: Any,
    prompts: list[str],
    device: torch.device,
    dtype: torch.dtype,
    desc: str = "",
) -> torch.Tensor:
    """Run forward pass on calibration prompts, collect last hidden states (on valid tokens)."""
    chunks: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        pbar = prompts
        if desc:
            from tqdm import tqdm as _tqdm
            pbar = _tqdm(prompts, desc=desc)
        for prompt in pbar:
            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            out = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = out.hidden_states[-1].detach().float().cpu()  # last layer hidden
            mask = encoded["attention_mask"].detach().cpu().bool()
            chunks.append(hidden[mask])  # keep only non-padding tokens
    return torch.cat(chunks, dim=0)


def mse_between_outputs(cur: torch.Tensor, ref: torch.Tensor) -> dict[str, float]:
    """Compute MSE / RMSE / relative MSE between compressed and reference hidden states."""
    if cur.shape != ref.shape:
        raise ValueError(f"Shape mismatch: cur={tuple(cur.shape)}, ref={tuple(ref.shape)}")
    diff = cur - ref
    sse = float(diff.square().sum())
    ref_norm_sq = float(ref.square().sum())
    count = diff.numel()
    mse_val = sse / max(count, 1)
    return {
        "mse": mse_val,
        "rmse": mse_val ** 0.5,
        "relative_mse": sse / max(ref_norm_sq, 1e-30),
        "max_abs": float(diff.abs().max()),
        "num_values": count,
    }


def compute_proxy_features(args: argparse.Namespace, formats: list[str], ranks: list[int]) -> dict[str, Any]:
    """Compute output-level hidden-state MSE for every (ka, kb) grid point.

    For each (ka, kb) combination:
    1. SVD-decompose all Linear weights at rank=kb, quantize main & residual
    2. Run forward pass with activation SVD-decomposition at rank=ka
    3. Compare last hidden states with uncompressed baseline → output MSE
    """
    cache_path = Path(args.proxy_cache) if args.proxy_cache else Path(args.output_dir) / "proxy_mse_cache.json"
    expected_keys = {f"{ka}_{kb}" for ka in ranks for kb in ranks}
    proxy: dict[str, Any] = {"mode": "last_hidden_output_mse", "ranks": ranks, "formats": {}}
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
        if cached.get("mode") == "last_hidden_output_mse":
            proxy = cached
            complete = True
            for fmt in formats:
                output_mse_map = proxy.get("formats", {}).get(fmt, {}).get("output_mse_map", {})
                if set(output_mse_map) & expected_keys != expected_keys:
                    complete = False
                    break
            if complete:
                print(f"[proxy] loading complete cached output MSE features from {cache_path}")
                return proxy
            print(f"[proxy] resuming incomplete output MSE cache from {cache_path}", flush=True)
        else:
            print(f"[proxy] ignoring incompatible cache at {cache_path}; expected last_hidden_output_mse mode", flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    data_path = resolve_data_path(args.data)
    if data_path is None:
        raise RuntimeError("No calibration data found. Pass --data.")

    device = torch.device(args.device)
    compute_dtype = dtype_from_arg(args.dtype)
    print(
        f"[proxy] torch_cuda_available={torch.cuda.is_available()} "
        f"cuda_device_count={torch.cuda.device_count()} "
        f"device={device} dtype={args.dtype} svd_method={args.svd_method}",
        flush=True,
    )

    # --- Load model & tokenizer once ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = load_prompts(data_path, args.limit, args.seed)
    print(f"[proxy] loaded {len(prompts)} calibration prompts", flush=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=compute_dtype if isinstance(compute_dtype, torch.dtype) else None,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    print(f"[proxy] loaded model {args.model_path}", flush=True)

    # --- Collect baseline (uncompressed) last hidden states ---
    print("[proxy] collecting baseline hidden states ...", flush=True)
    ref_hidden = collect_output_hidden(model, tokenizer, prompts, device, compute_dtype, desc="baseline")
    print(f"[proxy] baseline hidden states: shape={tuple(ref_hidden.shape)}", flush=True)

    # --- Compute output MSE for each (ka, kb) grid point ---
    # Count how many Linear layers will be patched
    candidate_names = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module_name.endswith("lm_head"):
            continue
        if not any(module_name.endswith(suf) for suf in TARGET_LINEAR_SUFFIXES):
            continue
        filters = [part.strip() for part in args.module_name_contains.split(",") if part.strip()]
        if filters and not any(token in module_name for token in filters):
            continue
        candidate_names.append(module_name)
    print(f"[proxy] will patch {len(candidate_names)} Linear layers", flush=True)

    # Free baseline model (keep ref_hidden on CPU for comparison)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    for fmt in formats:
        print(f"\n[proxy] === format: {fmt} ===", flush=True)
        quantizer = get_quantizer(fmt)
        output_mse_map: dict[str, dict[str, float]] = proxy.get("formats", {}).get(fmt, {}).get("output_mse_map", {})

        total = len(ranks) * len(ranks)
        count = 0
        for ka in ranks:
            for kb in ranks:
                count += 1
                key = f"{ka}_{kb}"
                if key in output_mse_map:
                    print(f"[proxy] {fmt} [{count}/{total}] ka={ka} kb={kb} skip cached", flush=True)
                    continue
                print(f"[proxy] {fmt} [{count}/{total}] ka={ka} kb={kb}", flush=True)

                # Reload fresh model
                model = AutoModelForCausalLM.from_pretrained(
                    args.model_path,
                    torch_dtype=compute_dtype if isinstance(compute_dtype, torch.dtype) else None,
                    trust_remote_code=True,
                    device_map=None,
                ).to(device)

                # Patch model with SVD + quantize residual layers
                replaced = patch_model_linears(
                    model,
                    weight_rank=kb,
                    act_rank=ka,
                    quantizer=quantizer,
                    svd_method=args.svd_method,
                    compute_dtype=compute_dtype if isinstance(compute_dtype, torch.dtype) else torch.float16,
                    module_name_contains=args.module_name_contains,
                )
                if count == 1:
                    print(f"[proxy] {fmt} patched {len(replaced)} modules (e.g. {replaced[:3]})", flush=True)

                # Collect compressed hidden states & compute MSE vs baseline
                cur_hidden = collect_output_hidden(
                    model, tokenizer, prompts, device, compute_dtype,
                    desc=f"{fmt} ka={ka} kb={kb}",
                )
                metrics = mse_between_outputs(cur_hidden, ref_hidden)
                output_mse_map[key] = metrics
                print(f"[proxy] {fmt} [{count}/{total}] ka={ka} kb={kb} mse={metrics['mse']:.6f} rmse={metrics['rmse']:.6f}", flush=True)

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Save incremental cache (so partial progress is preserved)
                proxy["formats"][fmt] = {"output_mse_map": output_mse_map}
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_text(json.dumps(proxy, indent=2), encoding="utf-8")

        proxy["formats"][fmt] = {"output_mse_map": output_mse_map}

    print(f"[proxy] done. cache saved to {cache_path}", flush=True)
    return proxy


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
    """Build feature matrix with output_mse (last-hidden MSE), ka, kb as features.

    Features: [1 (intercept), output_mse, ka, kb, ka*kb]
    All features are z-score normalized for numerical stability.
    """
    stats = {} if stats is None else dict(stats)
    out_mse, mean, std = zscore(np.array([row["output_mse"] for row in rows]), *stats.get("output_mse", (None, None)))
    stats["output_mse"] = (mean, std)
    ka, mean, std = zscore(np.array([row["ka"] for row in rows], dtype=np.float64), *stats.get("ka", (None, None)))
    stats["ka"] = (mean, std)
    kb, mean, std = zscore(np.array([row["kb"] for row in rows], dtype=np.float64), *stats.get("kb", (None, None)))
    stats["kb"] = (mean, std)
    features = np.column_stack([np.ones(len(rows)), out_mse, ka, kb, ka * kb])
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
    anchors = np.array([[next(r["is_anchor"] for r in rows if r["ka"] == ka and r["kb"] == kb) for ka in ka_values] for kb in kb_values])
    error = pred - actual

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    for ax, data, title in zip(axes, [actual, pred, error], ["Actual acc", "Predicted acc", "Prediction error"]):
        image = ax.imshow(data, aspect="auto", origin="lower")
        ax.set_title(f"{fmt} {title}")
        ax.set_xticks(range(len(ka_values)), ka_values, rotation=45)
        ax.set_yticks(range(len(kb_values)), kb_values)
        ax.set_xlabel("ka / xrank")
        ax.set_ylabel("kb / wrank")
        threshold = (float(np.nanmin(data)) + float(np.nanmax(data))) / 2.0
        for y_index in range(data.shape[0]):
            for x_index in range(data.shape[1]):
                value = float(data[y_index, x_index])
                text_color = "white" if value < threshold else "black"
                ax.text(
                    x_index,
                    y_index,
                    f"{value:.4f}" + ("*" if anchors[y_index, x_index] else ""),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color=text_color,
                )
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
    output_mse_map = proxy_data["formats"][fmt]["output_mse_map"]

    rows = []
    for kb in matrix.kb_values:
        for ka in matrix.ka_values:
            if ka not in ranks or kb not in ranks:
                continue
            actual_acc = matrix.values[(ka, kb)]
            key = f"{ka}_{kb}"
            if key not in output_mse_map:
                raise RuntimeError(
                    f"{fmt}: missing output MSE for ka={ka}, kb={kb}. "
                    "Delete incompatible cache or rerun to resume missing grid points."
                )
            mse_info = output_mse_map[key]
            rows.append(
                {
                    "format": fmt,
                    "ka": ka,
                    "kb": kb,
                    "output_mse": mse_info.get("mse", 0.0),
                    "output_rmse": mse_info.get("rmse", 0.0),
                    "output_relative_mse": mse_info.get("relative_mse", 0.0),
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
        is_anchor = (row["ka"], row["kb"]) in anchor_set
        model_pred_drop = float(pred_drop)
        model_pred_acc = float(baseline_acc - model_pred_drop)
        row["model_pred_acc_drop"] = model_pred_drop
        row["model_pred_acc"] = model_pred_acc
        row["model_pred_error"] = float(model_pred_acc - row["actual_acc"])
        row["abs_model_pred_error"] = abs(row["model_pred_error"])
        if is_anchor:
            row["pred_acc_drop"] = float(row["actual_acc_drop"])
            row["pred_acc"] = float(row["actual_acc"])
            row["pred_source"] = "actual_anchor"
        else:
            row["pred_acc_drop"] = model_pred_drop
            row["pred_acc"] = model_pred_acc
            row["pred_source"] = "ridge_prediction"
        row["pred_error"] = float(row["pred_acc"] - row["actual_acc"])
        row["abs_pred_error"] = abs(row["pred_error"])
        row["feasible_pred"] = row["pred_acc_drop"] <= args.max_acc_drop
        row["feasible_actual"] = row["actual_acc_drop"] <= args.max_acc_drop
        row["is_anchor"] = is_anchor

    feasible = [row for row in rows if row["feasible_pred"]]
    feasible.sort(key=lambda row: (row["rank_cost"], row["pred_acc_drop"], row["ka"], row["kb"]))
    model_errors = np.array([row["model_pred_error"] for row in rows], dtype=np.float64)
    model_abs_errors = np.abs(model_errors)
    non_anchor_errors = np.array([row["model_pred_error"] for row in rows if not row["is_anchor"]], dtype=np.float64)
    non_anchor_abs_errors = np.abs(non_anchor_errors) if len(non_anchor_errors) else np.array([], dtype=np.float64)
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
        "model_pred_mae_all": float(model_abs_errors.mean()),
        "model_pred_rmse_all": float(np.sqrt(np.mean(model_errors**2))),
        "model_pred_max_abs_error_all": float(model_abs_errors.max()),
        "model_pred_mae_non_anchor": float(non_anchor_abs_errors.mean()) if len(non_anchor_abs_errors) else None,
        "model_pred_rmse_non_anchor": float(np.sqrt(np.mean(non_anchor_errors**2))) if len(non_anchor_errors) else None,
        "model_pred_max_abs_error_non_anchor": float(non_anchor_abs_errors.max()) if len(non_anchor_abs_errors) else None,
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
