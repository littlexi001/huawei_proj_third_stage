#!/usr/bin/env python3
"""Evaluate global (ka, kp) configurations with real model forward passes.

The script is deliberately resumable because 144 configurations can be slow.
It evaluates:

* zero-shot MMLU accuracy using A/B/C/D continuation logits;
* final hidden-state relative squared error against the unquantized baseline;
* final-token logit relative squared error against the unquantized baseline.

The first `--calibration-size` examples after deterministic shuffling are
excluded, matching the sampling convention used by the proxy script. The next
`--eval-limit` examples form the held-out oracle set.
"""

from __future__ import annotations

import argparse
import json
import random
import types
import warnings
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_metis_rank_search import (
    CHOICES,
    dtype_from_arg,
    get_quantizer,
    mmlu_prompt,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--format", choices=["hif4", "hif8", "fp8", "nvfp8"], required=True)
    parser.add_argument("--rank-candidates", default="0,5,10,15,20,25,30,40,50,60,80,100")
    parser.add_argument(
        "--pairs",
        default="",
        help="Optional explicit config list such as '0:0,20:20,40:80'. If set, only these anchors are evaluated.",
    )
    parser.add_argument("--calibration-size", type=int, default=256)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument(
        "--svd-device",
        choices=["auto", "model", "cpu"],
        default="auto",
        help="Run SVD on the model device or CPU; auto tries the model device and falls back to CPU",
    )
    parser.add_argument("--svd-oversample", type=int, default=8)
    parser.add_argument("--svd-niter", type=int, default=2)
    parser.add_argument(
        "--ka-cost",
        type=float,
        default=1.0,
        help="Latency/rank-cost coefficient for one activation rank",
    )
    parser.add_argument(
        "--kp-cost",
        type=float,
        default=1.0,
        help="Latency/rank-cost coefficient for one weight rank",
    )
    parser.add_argument("--max-modules", type=int, default=0, help="Smoke-test limit; 0 evaluates all Linear modules")
    parser.add_argument("--output", required=True, help="Resumable JSONL output")
    return parser.parse_args()


def resolve_device(arg: str) -> torch.device:
    if arg != "auto":
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_model_dtype(arg: str, device: torch.device) -> torch.dtype | str:
    if arg != "auto":
        return dtype_from_arg(arg)
    # Float16 is the most widely supported and memory-efficient inference dtype
    # on Apple Silicon. Model metadata often requests BF16 under `auto`, which
    # is not supported by every macOS/PyTorch combination.
    if device.type == "mps":
        return torch.float16
    return dtype_from_arg(arg)


def clear_device_cache(device: torch.device) -> None:
    if device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def safe_svd(
    a: torch.Tensor,
    svd_device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute float32 SVD with an explicit CPU fallback for MPS."""
    source_device = a.device
    matrix = a.float()
    if svd_device == "cpu":
        matrix = matrix.cpu()
    try:
        u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
    except (NotImplementedError, RuntimeError) as exc:
        if svd_device != "auto" or source_device.type == "cpu":
            raise
        warnings.warn(
            f"SVD failed on {source_device} ({exc}); retrying on CPU. "
            "Use --svd-device cpu to select this path explicitly.",
            RuntimeWarning,
        )
        u, s, vh = torch.linalg.svd(a.float().cpu(), full_matrices=False)
    if u.device != source_device:
        u = u.to(source_device)
        s = s.to(source_device)
        vh = vh.to(source_device)
    return u, s, vh


def load_heldout(path: str, calibration_size: int, eval_limit: int, seed: int) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(seed).shuffle(rows)
    start = calibration_size
    return rows[start : start + eval_limit]


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_completed(path: Path) -> tuple[dict[str, Any] | None, set[tuple[int, int]]]:
    baseline = None
    completed: set[tuple[int, int]] = set()
    if not path.exists():
        return baseline, completed
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "baseline":
                baseline = row
            elif row.get("kind") == "config":
                completed.add((int(row["ka"]), int(row["kp"])))
    return baseline, completed


def parse_pairs(pairs: str, ranks: list[int]) -> list[tuple[int, int]]:
    if not pairs.strip():
        return [(ka, kp) for kp in ranks for ka in ranks]
    parsed = []
    for item in pairs.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            left, right = item.split(":", 1)
        elif "/" in item:
            left, right = item.split("/", 1)
        else:
            raise ValueError(f"Invalid pair {item!r}; expected ka:kp")
        parsed.append((int(left), int(right)))
    return parsed


def choice_token_ids(tokenizer: Any) -> list[int]:
    ids = []
    for choice in CHOICES:
        candidates = [f" {choice}", choice]
        encoded = None
        for text in candidates:
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) == 1:
                encoded = token_ids[0]
                break
        if encoded is None:
            raise RuntimeError(f"Choice {choice!r} is not a single token for this tokenizer")
        ids.append(encoded)
    return ids


def truncated_svd(
    a: torch.Tensor,
    max_rank: int,
    oversample: int,
    niter: int,
    svd_device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_possible = min(a.shape)
    rank = min(max_rank, max_possible)
    if rank == 0:
        return (
            a.new_empty((a.shape[0], 0), dtype=torch.float32),
            a.new_empty((0,), dtype=torch.float32),
            a.new_empty((0, a.shape[1]), dtype=torch.float32),
        )
    # `torch.svd_lowrank` is unexpectedly slower than the optimized exact SVD
    # on the target CPU/MPS environment. Compute once and retain only top-rank
    # factors; GPU runs may replace this helper with randomized SVD later.
    u, s, vh = safe_svd(a, svd_device)
    return u[:, :rank], s[:rank], vh[:rank, :]


class MetisLinearController:
    def __init__(
        self,
        module: nn.Linear,
        quantizer: Callable[[torch.Tensor], torch.Tensor],
        max_kp: int,
        oversample: int,
        niter: int,
        svd_device: str,
    ) -> None:
        self.module = module
        self.quantizer = quantizer
        self.original_forward = module.forward
        self.original_weight = module.weight.detach()
        self.ka = 0
        self.kp: int | None = None
        self.quant_weight = self.original_weight

        u, s, vh = truncated_svd(
            self.original_weight,
            max_kp,
            oversample,
            niter,
            svd_device,
        )
        self.u = u
        self.s = s
        self.vh = vh
        self.svd_device = svd_device
        module.forward = types.MethodType(self._forward, module)

    def configure(self, ka: int, kp: int) -> None:
        self.ka = ka
        effective_kp = min(kp, self.s.numel())
        if self.kp == effective_kp:
            return
        self.kp = effective_kp
        weight = self.original_weight.float()
        if effective_kp > 0:
            head = (self.u[:, :effective_kp] * self.s[:effective_kp]) @ self.vh[:effective_kp, :]
            residual = weight - head
            reconstructed = head + self.quantizer(residual)
        else:
            reconstructed = self.quantizer(weight)
        self.quant_weight = reconstructed.to(device=weight.device, dtype=self.original_weight.dtype)

    def _forward(self, _module: nn.Linear, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        matrix = x.reshape(-1, original_shape[-1]).float()
        effective_ka = min(self.ka, min(matrix.shape))
        if effective_ka > 0:
            u, s, vh = safe_svd(matrix, self.svd_device)
            head = (u[:, :effective_ka] * s[:effective_ka]) @ vh[:effective_ka, :]
            residual = matrix - head
            quant_x = head + self.quantizer(residual)
        else:
            quant_x = self.quantizer(matrix)
        quant_x = quant_x.reshape(original_shape).to(dtype=x.dtype)
        return F.linear(quant_x, self.quant_weight, self.module.bias)

    def restore(self) -> None:
        self.module.forward = self.original_forward


def install_controllers(
    model: nn.Module,
    quantizer: Callable[[torch.Tensor], torch.Tensor],
    max_kp: int,
    oversample: int,
    niter: int,
    svd_device: str,
    max_modules: int,
) -> list[MetisLinearController]:
    controllers = []
    modules = [
        module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and not name.endswith("lm_head")
    ]
    if max_modules:
        modules = modules[:max_modules]
    for index, module in enumerate(modules, start=1):
        print(f"Preparing weight SVD {index}/{len(modules)}", flush=True)
        controllers.append(
            MetisLinearController(
                module,
                quantizer,
                max_kp,
                oversample,
                niter,
                svd_device,
            )
        )
    return controllers


def run_examples(
    model: nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    token_ids: list[int],
    max_length: int,
    batch_size: int,
    device: torch.device,
    baseline_cache: list[dict[str, torch.Tensor]] | None,
) -> tuple[dict[str, float], list[dict[str, torch.Tensor]] | None]:
    correct = 0
    hidden_error = 0.0
    hidden_ref = 0.0
    logit_error = 0.0
    logit_ref = 0.0
    generated_cache = [] if baseline_cache is None else None

    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            encoded = tokenizer(
                [mmlu_prompt(row) for row in batch_rows],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)
            logits = outputs.logits[:, -1].float()
            hidden = outputs.hidden_states[-1].float()
            choice_logits = logits[:, token_ids]
            predictions = choice_logits.argmax(dim=-1).cpu().tolist()
            correct += sum(
                int(prediction == int(row["answer"]))
                for prediction, row in zip(predictions, batch_rows)
            )

            compact = {
                "hidden": hidden.cpu(),
                "logits": logits.cpu(),
                "attention_mask": encoded["attention_mask"].cpu(),
            }
            if generated_cache is not None:
                generated_cache.append(compact)
            else:
                ref = baseline_cache[batch_start // batch_size]
                hidden_cpu = compact["hidden"]
                logits_cpu = compact["logits"]
                mask = compact["attention_mask"].unsqueeze(-1).float()
                hidden_error += float(((hidden_cpu - ref["hidden"]).square() * mask).sum())
                hidden_ref += float((ref["hidden"].square() * mask).sum())
                logit_error += float((logits_cpu - ref["logits"]).square().sum())
                logit_ref += float(ref["logits"].square().sum())

    metrics = {
        "accuracy": correct / len(rows),
        "correct": correct,
        "num_examples": len(rows),
    }
    if baseline_cache is not None:
        metrics["hidden_relative_mse"] = hidden_error / max(hidden_ref, 1e-12)
        metrics["logit_relative_mse"] = logit_error / max(logit_ref, 1e-12)
    return metrics, generated_cache


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    ranks = [int(value) for value in args.rank_candidates.split(",") if value.strip()]
    pairs = parse_pairs(args.pairs, ranks)
    output = Path(args.output)
    existing_baseline, completed = read_completed(output)
    rows = load_heldout(args.data, args.calibration_size, args.eval_limit, args.seed)
    if not rows:
        raise RuntimeError("Held-out set is empty")

    device = resolve_device(args.device)
    if args.device == "mps" and device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available in this Python environment")
    model_dtype = resolve_model_dtype(args.dtype, device)
    svd_device = "model" if args.svd_device == "model" else args.svd_device
    print(f"Using device: {device}")
    print(f"Using model dtype: {model_dtype}")
    print(f"Using SVD device policy: {svd_device}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=model_dtype,
        trust_remote_code=True,
        device_map=None,
    ).to(device)
    token_ids = choice_token_ids(tokenizer)

    print("Running unquantized baseline", flush=True)
    baseline_metrics, baseline_cache = run_examples(
        model, tokenizer, rows, token_ids, args.max_length, args.batch_size, device, baseline_cache=None
    )
    if existing_baseline is None:
        append_jsonl(
            output,
            {
                "kind": "baseline",
                "format": args.format,
                "pairs": pairs if args.pairs.strip() else "cartesian",
                "calibration_size": args.calibration_size,
                "eval_limit": len(rows),
                "seed": args.seed,
                "batch_size": args.batch_size,
                "ka_cost": args.ka_cost,
                "kp_cost": args.kp_cost,
                **baseline_metrics,
            },
        )
    print("Baseline:", baseline_metrics, flush=True)

    quantizer = get_quantizer(args.format)
    controllers = install_controllers(
        model,
        quantizer=quantizer,
        max_kp=max(ranks),
        oversample=args.svd_oversample,
        niter=args.svd_niter,
        svd_device=svd_device,
        max_modules=args.max_modules,
    )
    try:
        for kp in sorted({kp for _, kp in pairs}):
            for controller in controllers:
                controller.configure(ka=0, kp=kp)
            for ka in [ka for ka, pair_kp in pairs if pair_kp == kp]:
                if (ka, kp) in completed:
                    continue
                print(f"Evaluating ka={ka}, kp={kp}", flush=True)
                for controller in controllers:
                    controller.configure(ka=ka, kp=kp)
                metrics, _ = run_examples(
                    model,
                    tokenizer,
                    rows,
                    token_ids,
                    args.max_length,
                    args.batch_size,
                    device,
                    baseline_cache=baseline_cache,
                )
                row = {
                    "kind": "config",
                    "format": args.format,
                    "ka": ka,
                    "kp": kp,
                    "rank_cost": args.ka_cost * ka + args.kp_cost * kp,
                    "accuracy_drop": baseline_metrics["accuracy"] - metrics["accuracy"],
                    **metrics,
                }
                append_jsonl(output, row)
                print(row, flush=True)
                clear_device_cache(device)
    finally:
        for controller in controllers:
            controller.restore()


if __name__ == "__main__":
    main()
