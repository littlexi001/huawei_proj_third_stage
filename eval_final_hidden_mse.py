#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from Metis.quant import BlockQuantFunc, quant_func


TARGET_LINEAR_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}

FORMAT_CONFIGS = {
    "nvfp4": {"qtype": "nvfp4e2m1bnosr", "blocksize": 16},
    "mxfp4": {"qtype": "mxfp4e2m1bnosr", "blocksize": 32},
    "nvfp8": {"qtype": "fp8e4m3b", "blocksize": 16},
    "mxfp8": {"qtype": "fp8e4m3b", "blocksize": 32},
    "hif4": {"qtype": "hif4_0418", "blocksize": 16},
    "hif8": {"qtype": "hif8_cuda", "blocksize": 16},
}


@dataclass
class EvalConfig:
    model_path: str
    text_path: str
    output_dir: str
    formats: List[str]
    rank_grid: List[int]
    svd_method_w: str
    svd_method_x: str
    max_samples: int
    max_length: int
    batch_size: int
    stride: int
    dtype: str
    store_dtype: str
    q_scalar_w: float
    q_scalar_x: float
    seed: int
    device: str
    trust_remote_code: bool


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def maybe_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(dtype_str)


def inference_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


def disabled_autocast(device_type: str):
    if device_type == "cuda" and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=False)
    from contextlib import nullcontext

    return nullcontext()


@torch.no_grad()
def quant_tensor(x: torch.Tensor, qtype: str, q_scalar: float) -> torch.Tensor:
    if qtype not in quant_func:
        raise KeyError(f"Unknown qtype: {qtype}. Available qtypes: {list(quant_func.keys())}")
    x_fp = x.float().contiguous()
    qcls = quant_func[qtype]
    s = qcls.get_scalar(x_fp) * q_scalar
    return qcls.rquant(qcls.quant(x_fp, s), s)


def parse_rank_grid(s: str) -> List[int]:
    ranks = [int(x.strip()) for x in s.split(",") if x.strip()]
    if not ranks:
        raise ValueError("rank_grid is empty")
    if any(x < 0 for x in ranks):
        raise ValueError(f"rank_grid contains negative rank: {ranks}")
    return ranks


def _full_svd_lowrank(x2d: torch.Tensor, rank: int) -> torch.Tensor:
    with disabled_autocast(x2d.device.type):
        x32 = x2d.float().contiguous()
        u, s, vh = torch.linalg.svd(x32, full_matrices=False)
        r = min(rank, s.numel())
        if r <= 0:
            return torch.zeros_like(x32)
        return (u[:, :r] * s[:r]) @ vh[:r, :]


def _randomized_svd_lowrank(x2d: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    with disabled_autocast(x2d.device.type):
        x32 = x2d.float().contiguous()
        r = min(rank, min(x32.shape))
        if r <= 0:
            return torch.zeros_like(x32)
        if r >= min(x32.shape):
            return x32.clone()
        u, s, v = torch.svd_lowrank(x32, q=r, niter=niter)
        return (u[:, :r] * s[:r]) @ v[:, :r].T


@torch.no_grad()
def low_rank_approx(x2d: torch.Tensor, rank: int, method: str) -> torch.Tensor:
    with disabled_autocast(x2d.device.type):
        x32 = x2d.float().contiguous()
        r = min(int(rank), min(x32.shape))
        if r <= 0:
            return torch.zeros_like(x32)
        if r >= min(x32.shape):
            return x32.clone()
        if method == "full":
            return _full_svd_lowrank(x32, r)
        if method == "randomized":
            return _randomized_svd_lowrank(x32, r)
        raise ValueError(f"Unknown SVD method: {method}")


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


class PureQuantLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        qtype: str,
        q_scalar_w: float,
        q_scalar_x: float,
        compute_dtype: torch.dtype,
        store_dtype: torch.dtype,
    ):
        super().__init__()
        self.qtype = qtype
        self.q_scalar_x = float(q_scalar_x)
        self.compute_dtype = compute_dtype

        device = base_linear.weight.device
        wq = quant_tensor(base_linear.weight.detach(), qtype, q_scalar_w)
        self.register_buffer("weight_q", wq.to(dtype=store_dtype, device=device), persistent=True)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base_linear.bias.detach().to(dtype=store_dtype, device=device), persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xq = quant_tensor(x, self.qtype, self.q_scalar_x).to(dtype=self.compute_dtype)
        wq = self.weight_q.to(dtype=self.compute_dtype)
        bias = self.bias.to(dtype=self.compute_dtype) if self.bias is not None else None
        return F.linear(xq, wq, bias)


class SVDRQuantLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        weight_rank: int,
        act_rank: int,
        qtype: str,
        q_scalar_w: float,
        q_scalar_x: float,
        svd_method_w: str,
        svd_method_x: str,
        compute_dtype: torch.dtype,
        store_dtype: torch.dtype,
    ):
        super().__init__()
        self.weight_rank = int(weight_rank)
        self.act_rank = int(act_rank)
        self.qtype = qtype
        self.q_scalar_x = float(q_scalar_x)
        self.svd_method_x = svd_method_x
        self.compute_dtype = compute_dtype

        device = base_linear.weight.device
        w_fp = base_linear.weight.detach().float()
        if self.weight_rank <= 0:
            w_m = torch.zeros_like(w_fp)
            w_r = w_fp
        elif self.weight_rank >= min(w_fp.shape):
            w_m = w_fp
            w_r = torch.zeros_like(w_fp)
        else:
            w_m = low_rank_approx(w_fp, self.weight_rank, svd_method_w)
            w_r = w_fp - w_m

        w_mq = quant_tensor(w_m, qtype, q_scalar_w)
        w_rq = quant_tensor(w_r, qtype, q_scalar_w)
        self.register_buffer("weight_main", w_m.to(dtype=store_dtype, device=device), persistent=True)
        self.register_buffer("weight_main_q", w_mq.to(dtype=store_dtype, device=device), persistent=True)
        self.register_buffer("weight_resid_q", w_rq.to(dtype=store_dtype, device=device), persistent=True)
        if base_linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", base_linear.bias.detach().to(dtype=store_dtype, device=device), persistent=True)

    @torch.no_grad()
    def _decompose_activation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            x_m = low_rank_approx(x_fp, self.act_rank, self.svd_method_x)
            x_r = x_fp - x_m

        x_mq = quant_tensor(x_m, self.qtype, self.q_scalar_x)
        x_rq = quant_tensor(x_r, self.qtype, self.q_scalar_x)
        return (
            x_m.reshape(orig_shape).to(dtype=self.compute_dtype),
            x_mq.reshape(orig_shape).to(dtype=self.compute_dtype),
            x_rq.reshape(orig_shape).to(dtype=self.compute_dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_m, x_mq, x_rq = self._decompose_activation(x)
        w_m = self.weight_main.to(dtype=self.compute_dtype)
        w_mq = self.weight_main_q.to(dtype=self.compute_dtype)
        w_rq = self.weight_resid_q.to(dtype=self.compute_dtype)
        bias = self.bias.to(dtype=self.compute_dtype) if self.bias is not None else None

        out = F.linear(x_m, w_m, bias)
        out = out + F.linear(x_mq, w_rq, None)
        out = out + F.linear(x_rq, w_mq, None)
        out = out + F.linear(x_rq, w_rq, None)
        return out


def patch_model_linears(
    model: nn.Module,
    qtype: str,
    blocksize: int,
    weight_rank: int,
    act_rank: int,
    q_scalar_w: float,
    q_scalar_x: float,
    svd_method_w: str,
    svd_method_x: str,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype,
) -> List[str]:
    BlockQuantFunc.block_shape = (1, int(blocksize))
    replaced = []
    for module_name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or module_name == "lm_head":
            continue
        if not any(module_name.endswith(suf) for suf in TARGET_LINEAR_SUFFIXES):
            continue
        parent, child_name = get_parent_module(model, module_name)
        setattr(
            parent,
            child_name,
            SVDRQuantLinear(
                module,
                weight_rank,
                act_rank,
                qtype,
                q_scalar_w,
                q_scalar_x,
                svd_method_w,
                svd_method_x,
                compute_dtype,
                store_dtype,
            ),
        )
        replaced.append(module_name)
    return replaced


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def build_token_batches(
    tokenizer,
    text_path: str,
    max_samples: int,
    max_length: int,
    batch_size: int,
    stride: int,
) -> List[Dict[str, torch.Tensor]]:
    text = read_text(text_path)
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"No tokens found in {text_path}")

    step = stride if stride > 0 else max_length
    samples = []
    for start in range(0, max(1, len(ids) - max_length + 1), step):
        chunk = ids[start : start + max_length]
        if len(chunk) < 2:
            continue
        samples.append(chunk)
        if len(samples) >= max_samples:
            break

    if not samples:
        samples = [ids[:max_length]]

    batches = []
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    for i in range(0, len(samples), batch_size):
        cur = samples[i : i + batch_size]
        max_len = max(len(x) for x in cur)
        input_ids = torch.full((len(cur), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros((len(cur), max_len), dtype=torch.long)
        for row, seq in enumerate(cur):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, : len(seq)] = 1
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches


def load_model(model_path: str, dtype: torch.dtype, trust_remote_code: bool):
    kwargs = dict(device_map=None, trust_remote_code=trust_remote_code)
    try:
        return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, **kwargs)


@torch.no_grad()
def collect_last_hidden(
    model: nn.Module,
    batches: Iterable[Dict[str, torch.Tensor]],
    device: torch.device,
    dtype: torch.dtype,
    desc: str,
) -> List[torch.Tensor]:
    outputs = []
    for batch in tqdm(list(batches), desc=desc):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        with inference_autocast(device, dtype):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = out.hidden_states[-1].detach().float().cpu()
        mask = attention_mask.detach().cpu().bool()
        outputs.append(hidden[mask])
    return outputs


def mse_against_reference(cur: List[torch.Tensor], ref: List[torch.Tensor]) -> Dict[str, float]:
    if len(cur) != len(ref):
        raise ValueError(f"Batch count mismatch: cur={len(cur)}, ref={len(ref)}")
    sse = 0.0
    count = 0
    max_abs = 0.0
    ref_norm_sq = 0.0
    for c, r in zip(cur, ref):
        if c.shape != r.shape:
            raise ValueError(f"Shape mismatch: cur={tuple(c.shape)}, ref={tuple(r.shape)}")
        diff = c - r
        sse += diff.square().sum().item()
        count += diff.numel()
        max_abs = max(max_abs, diff.abs().max().item())
        ref_norm_sq += r.square().sum().item()
    mse = sse / max(count, 1)
    return {
        "mse": mse,
        "rmse": mse ** 0.5,
        "relative_mse": sse / max(ref_norm_sq, 1e-30),
        "max_abs": max_abs,
        "num_values": count,
    }


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_csv(path: str, results: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = [
        "format",
        "qtype",
        "blocksize",
        "weight_rank",
        "act_rank",
        "mse",
        "rmse",
        "relative_mse",
        "max_abs",
        "num_values",
        "patched_num_modules",
        "elapsed_sec",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for key, row in results.items():
            vals = [str(row.get(field, key if field == "format" else "")) for field in fields]
            f.write(",".join(vals) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--text_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs/final_hidden_mse")
    parser.add_argument("--formats", type=str, default="nvfp4,nvfp8,mxfp4,mxfp8,hif4,hif8")
    parser.add_argument("--rank_grid", type=str, default="0,5,10,15,20,25,30,40,50,60,80,100")
    parser.add_argument("--svd_method_w", type=str, default="randomized", choices=["full", "randomized"])
    parser.add_argument("--svd_method_x", type=str, default="randomized", choices=["full", "randomized"])
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--store_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--q_scalar_w", type=float, default=1.0)
    parser.add_argument("--q_scalar_x", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true", default=True)
    args = parser.parse_args()

    formats = [x.strip() for x in args.formats.split(",") if x.strip()]
    unknown = [x for x in formats if x not in FORMAT_CONFIGS]
    if unknown:
        raise KeyError(f"Unknown formats: {unknown}. Available: {list(FORMAT_CONFIGS)}")
    rank_grid = parse_rank_grid(args.rank_grid)

    cfg = EvalConfig(
        model_path=args.model_path,
        text_path=args.text_path,
        output_dir=args.output_dir,
        formats=formats,
        rank_grid=rank_grid,
        svd_method_w=args.svd_method_w,
        svd_method_x=args.svd_method_x,
        max_samples=args.max_samples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        stride=args.stride,
        dtype=args.dtype,
        store_dtype=args.store_dtype,
        q_scalar_w=args.q_scalar_w,
        q_scalar_x=args.q_scalar_x,
        seed=args.seed,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    compute_dtype = maybe_dtype(cfg.dtype)
    store_dtype = maybe_dtype(cfg.store_dtype)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=cfg.trust_remote_code)
    batches = build_token_batches(
        tokenizer,
        cfg.text_path,
        cfg.max_samples,
        cfg.max_length,
        cfg.batch_size,
        cfg.stride,
    )
    write_json(
        os.path.join(cfg.output_dir, "config.json"),
        {**asdict(cfg), "num_batches": len(batches), "num_sequences": sum(b["input_ids"].shape[0] for b in batches)},
    )

    print("[INFO] loading baseline model")
    model = load_model(cfg.model_path, compute_dtype, cfg.trust_remote_code).to(device).eval()
    ref_hidden = collect_last_hidden(model, batches, device, compute_dtype, "baseline")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    results = {}
    for fmt in formats:
        fmt_cfg = FORMAT_CONFIGS[fmt]
        for weight_rank in rank_grid:
            for act_rank in rank_grid:
                key = f"{fmt}/wrank_{weight_rank}_xrank_{act_rank}"
                out_dir = os.path.join(cfg.output_dir, fmt, f"wrank_{weight_rank}_xrank_{act_rank}")
                summary_path = os.path.join(out_dir, "summary.json")
                if os.path.exists(summary_path) and not args.overwrite:
                    with open(summary_path, "r", encoding="utf-8") as f:
                        results[key] = json.load(f)
                    print(f"[SKIP] existing result: {summary_path}")
                    write_json(os.path.join(cfg.output_dir, "summary.json"), {"config": asdict(cfg), "results": results})
                    write_csv(os.path.join(cfg.output_dir, "summary.csv"), results)
                    continue

                print(
                    f"[INFO] evaluating {fmt}: qtype={fmt_cfg['qtype']} "
                    f"blocksize={fmt_cfg['blocksize']} wrank={weight_rank} xrank={act_rank}"
                )
                model = load_model(cfg.model_path, compute_dtype, cfg.trust_remote_code).to(device).eval()
                replaced = patch_model_linears(
                    model,
                    qtype=fmt_cfg["qtype"],
                    blocksize=fmt_cfg["blocksize"],
                    weight_rank=weight_rank,
                    act_rank=act_rank,
                    q_scalar_w=cfg.q_scalar_w,
                    q_scalar_x=cfg.q_scalar_x,
                    svd_method_w=cfg.svd_method_w,
                    svd_method_x=cfg.svd_method_x,
                    compute_dtype=compute_dtype,
                    store_dtype=store_dtype,
                )
                start = time.time()
                desc = f"{fmt} w={weight_rank} x={act_rank}"
                cur_hidden = collect_last_hidden(model, batches, device, compute_dtype, desc)
                metrics = mse_against_reference(cur_hidden, ref_hidden)
                metrics.update(
                    {
                        "format": fmt,
                        "qtype": fmt_cfg["qtype"],
                        "blocksize": fmt_cfg["blocksize"],
                        "weight_rank": weight_rank,
                        "act_rank": act_rank,
                        "patched_num_modules": len(replaced),
                        "elapsed_sec": time.time() - start,
                    }
                )
                results[key] = metrics
                write_json(os.path.join(out_dir, "summary.json"), metrics)
                write_json(os.path.join(cfg.output_dir, "summary.json"), {"config": asdict(cfg), "results": results})
                write_csv(os.path.join(cfg.output_dir, "summary.csv"), results)
                print(json.dumps(metrics, ensure_ascii=False, indent=2))
                del model, cur_hidden
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_json(os.path.join(cfg.output_dir, "summary.json"), {"config": asdict(cfg), "results": results})
    write_csv(os.path.join(cfg.output_dir, "summary.csv"), results)
    print(f"[Done] summary: {os.path.join(cfg.output_dir, 'summary.json')}")


if __name__ == "__main__":
    main()
