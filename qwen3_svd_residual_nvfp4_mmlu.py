
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen3-0.6B:
低秩主分量 + 残差 NVFP4(E2M1B, nosr) 量化 的 MMLU 评测脚本
修复版：SVD/QR/quant 强制在 FP32 + autocast disabled 下执行，避免 BF16 geqrf_cuda 报错

核心算子:
    W = W_m + W_r
    X = X_m + X_r

    y = (W_m + Quant(W_r)) * (X_m + Quant(X_r))

这里默认只对 Transformer block 中 7 个主要线性层做替换:
    self_attn.{q_proj,k_proj,v_proj,o_proj}
    mlp.{gate_proj,up_proj,down_proj}

功能:
1) 单个 (weight_rank, act_rank) 评测
2) 12x12 网格任务自动分发到多张 GPU
3) 结果按 json 保存，便于后续汇总画图

依赖:
    pip install torch transformers datasets tqdm
    以及你的 Metis 包可 import:
        from Metis.quant import quant_func
"""

import os
import re
import gc
import json
import time
import math
import queue
import shutil
import argparse
import subprocess
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Any, Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# =========================
# Metis quant
# =========================
try:
    from Metis.quant import quant_func
except Exception as e:
    quant_func = None
    _METIS_IMPORT_ERROR = repr(e)
else:
    _METIS_IMPORT_ERROR = None


def quant_tensor(x: torch.Tensor, qtype: str, q_scalar: float = 1.0) -> torch.Tensor:
    if quant_func is None:
        raise ImportError(
            "Failed to import Metis.quant.quant_func. "
            f"Original error: {_METIS_IMPORT_ERROR}"
        )

    if qtype not in quant_func:
        raise KeyError(
            f"Unknown qtype: {qtype}. "
            f"Available qtypes: {list(quant_func.keys())}"
        )

    # 量化也放到 autocast 之外，避免外层 BF16 autocast 改变中间计算 dtype。
    with disabled_autocast_context(x.device.type):
        x_fp = x.float().contiguous() if x.is_floating_point() else x.contiguous()
        qcls = quant_func[qtype]
        s = qcls.get_scalar(x_fp) * q_scalar
        xq = qcls.quant(x_fp, s)
        xq = qcls.rquant(xq, s)
    return xq


# =========================
# Utils
# =========================
TARGET_LINEAR_SUFFIXES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def json_dump(obj: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def jsonl_append(obj: Dict[str, Any], path: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def normalize_answer(ans: Any) -> str:
    if isinstance(ans, int):
        if 0 <= ans <= 3:
            return "ABCD"[ans]
    ans = str(ans).strip().upper()
    if ans in {"A", "B", "C", "D"}:
        return ans
    if ans in {"0", "1", "2", "3"}:
        return "ABCD"[int(ans)]
    raise ValueError(f"Unknown answer format: {ans}")


def list_subjects_from_dataset(ds) -> List[str]:
    subs = sorted(list(set(ds["subject"])))
    return subs


def parse_gpu_ids(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def maybe_autocast_dtype(dtype_str: str):
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float32":
        return torch.float32
    raise ValueError(dtype_str)


def disabled_autocast_context(device_type: str):
    """
    SVD / QR / quant 部分必须在 FP32 下执行。
    外层 model forward 可能开了 BF16 autocast；如果不关掉，torch.svd_lowrank
    内部的 torch.linalg.qr 会收到 BF16 矩阵，从而报：
        RuntimeError: "geqrf_cuda" not implemented for 'BFloat16'
    """
    if device_type == "cuda" and hasattr(torch, "autocast"):
        return torch.autocast(device_type="cuda", enabled=False)
    return nullcontext()


def inference_autocast_context(device: torch.device, dtype: torch.dtype):
    """只在 CUDA + fp16/bf16 时开启 autocast；float32 或 CPU 时不启用。"""
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def load_causal_lm_compat(model_path: str, dtype: torch.dtype):
    """兼容新版 transformers 的 dtype 参数和旧版 torch_dtype 参数。"""
    common_kwargs = dict(
        device_map=None,
        trust_remote_code=True,
    )
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            **common_kwargs,
        )
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            **common_kwargs,
        )


# =========================
# Low-rank approx
# =========================
def _full_svd_lowrank(x2d: torch.Tensor, rank: int) -> torch.Tensor:
    # x2d: [m, n]. 这里强制 FP32，避免 BF16 CUDA SVD/QR 不支持。
    with disabled_autocast_context(x2d.device.type):
        x32 = x2d.float().contiguous()
        U, S, Vh = torch.linalg.svd(x32, full_matrices=False)
        r = min(rank, S.numel())
        if r <= 0:
            return torch.zeros_like(x32)
        return (U[:, :r] * S[:r]) @ Vh[:r, :]


def _randomized_svd_lowrank(x2d: torch.Tensor, rank: int, niter: int = 2) -> torch.Tensor:
    # torch.svd_lowrank 内部会调用 torch.linalg.qr；必须避免 BF16。
    with disabled_autocast_context(x2d.device.type):
        x32 = x2d.float().contiguous()
        r = min(rank, min(x32.shape))
        if r <= 0:
            return torch.zeros_like(x32)
        q = min(max(r + 8, r), min(x32.shape))
        U, S, V = torch.svd_lowrank(x32, q=q, niter=niter)
        r = min(r, S.numel())
        return (U[:, :r] * S[:r]) @ V[:, :r].T


def low_rank_approx(x: torch.Tensor, rank: int, method: str = "full") -> torch.Tensor:
    """
    对二维矩阵做 rank-r 近似，返回 x_m。
    注意：无论外层模型是否使用 BF16/FP16，这里都返回 FP32 的低秩近似。
    后续再显式 cast 到 compute_dtype。
    """
    with disabled_autocast_context(x.device.type):
        x32 = x.float().contiguous()

        if rank <= 0:
            return torch.zeros_like(x32)

        m, n = x32.shape
        max_rank = min(m, n)
        if rank >= max_rank:
            return x32.clone()

        if method == "full":
            return _full_svd_lowrank(x32, rank)
        if method == "randomized":
            return _randomized_svd_lowrank(x32, rank)
        raise ValueError(f"Unknown SVD method: {method}")


# =========================
# Wrapped Linear
# =========================
class SVDRQuantLinear(nn.Module):
    """
    将原线性层替换为:
        y = linear(X_m + Q(X_r), W_m + Q(W_r), b)

    其中:
        W_m: 预先算好并缓存
        Q(W_r): 预先量化并缓存
        X_m / Q(X_r): 每次 forward 在线计算
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        weight_rank: int,
        act_rank: int,
        qtype: str = "nvfp4e2m1bnosr",
        q_scalar_w: float = 1.0,
        q_scalar_x: float = 1.0,
        svd_method_w: str = "randomized",
        svd_method_x: str = "full",
        compute_dtype: torch.dtype = torch.bfloat16,
        store_dtype: torch.dtype = torch.bfloat16,
        layer_name: str = "",
        verbose: bool = False,
    ):
        super().__init__()
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.weight_rank = int(weight_rank)
        self.act_rank = int(act_rank)
        self.qtype = qtype
        self.q_scalar_w = float(q_scalar_w)
        self.q_scalar_x = float(q_scalar_x)
        self.svd_method_w = svd_method_w
        self.svd_method_x = svd_method_x
        self.compute_dtype = compute_dtype
        self.store_dtype = store_dtype
        self.layer_name = layer_name
        self.verbose = verbose

        with torch.no_grad():
            w = base_linear.weight.data.detach()
            device = w.device
            w_fp = w.float()

            if self.weight_rank <= 0:
                w_m = torch.zeros_like(w_fp)
                w_r = w_fp
            elif self.weight_rank >= min(w_fp.shape):
                w_m = w_fp
                w_r = torch.zeros_like(w_fp)
            else:
                w_m = low_rank_approx(w_fp, self.weight_rank, method=self.svd_method_w)
                w_r = w_fp - w_m

            w_rq = quant_tensor(w_r, qtype=self.qtype, q_scalar=self.q_scalar_w)

            self.register_buffer("weight_main", w_m.to(dtype=self.store_dtype, device=device), persistent=True)
            self.register_buffer("weight_resid_q", w_rq.to(dtype=self.store_dtype, device=device), persistent=True)

            if base_linear.bias is not None:
                self.register_buffer(
                    "bias",
                    base_linear.bias.data.detach().to(dtype=self.store_dtype, device=device),
                    persistent=True
                )
            else:
                self.bias = None

        if self.verbose:
            print(
                f"[SVDRQuantLinear] {layer_name} | "
                f"W rank={self.weight_rank}, X rank={self.act_rank}, "
                f"shape=({self.out_features}, {self.in_features})"
            )

    @torch.no_grad()
    def _decompose_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [..., hidden]
        将 x reshape 成 [tokens, hidden] 后做低秩近似
        """
        orig_shape = x.shape
        assert x.dim() >= 2, f"Unexpected x.shape={orig_shape}"

        hidden = orig_shape[-1]
        x2d = x.reshape(-1, hidden)

        x_fp = x2d.float()

        if self.act_rank <= 0:
            x_m = torch.zeros_like(x_fp)
            x_r = x_fp
        elif self.act_rank >= min(x_fp.shape):
            x_m = x_fp
            x_r = torch.zeros_like(x_fp)
        else:
            x_m = low_rank_approx(x_fp, self.act_rank, method=self.svd_method_x)
            x_r = x_fp - x_m

        x_rq = quant_tensor(x_r, qtype=self.qtype, q_scalar=self.q_scalar_x)
        x_hat = x_m + x_rq
        x_hat = x_hat.reshape(orig_shape).to(dtype=self.compute_dtype)
        return x_hat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_hat = self._decompose_activation(x)
        w_hat = (self.weight_main + self.weight_resid_q).to(dtype=self.compute_dtype)
        b = self.bias.to(dtype=self.compute_dtype) if self.bias is not None else None
        out = F.linear(x_hat, w_hat, b)
        return out


def patch_model_linears(
    model: nn.Module,
    weight_rank: int,
    act_rank: int,
    qtype: str,
    q_scalar_w: float,
    q_scalar_x: float,
    svd_method_w: str,
    svd_method_x: str,
    compute_dtype: torch.dtype,
    store_dtype: torch.dtype,
    verbose: bool = False,
) -> List[str]:
    replaced = []
    named_modules = list(model.named_modules())

    for module_name, module in named_modules:
        if not isinstance(module, nn.Linear):
            continue

        if module_name == "lm_head":
            continue

        matched = False
        for suf in TARGET_LINEAR_SUFFIXES:
            if module_name.endswith(suf):
                matched = True
                break
        if not matched:
            continue

        parent, child_name = get_parent_module(model, module_name)
        wrapped = SVDRQuantLinear(
            base_linear=module,
            weight_rank=weight_rank,
            act_rank=act_rank,
            qtype=qtype,
            q_scalar_w=q_scalar_w,
            q_scalar_x=q_scalar_x,
            svd_method_w=svd_method_w,
            svd_method_x=svd_method_x,
            compute_dtype=compute_dtype,
            store_dtype=store_dtype,
            layer_name=module_name,
            verbose=verbose,
        )
        setattr(parent, child_name, wrapped)
        replaced.append(module_name)

    return replaced


# =========================
# MMLU prompt / eval
# =========================
def format_subject(subject: str) -> str:
    return subject.replace("_", " ")


def format_example(ex: Dict[str, Any], include_answer: bool = True) -> str:
    q = ex["question"].strip()
    choices = ex["choices"]
    answer = normalize_answer(ex["answer"])
    text = (
        f"Question: {q}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        f"Answer:"
    )
    if include_answer:
        text += f" {answer}\n\n"
    else:
        text += " "
    return text


def build_fewshot_prompt(subject: str, dev_examples: List[Dict[str, Any]], query_ex: Dict[str, Any]) -> str:
    header = (
        f"The following are multiple choice questions (with answers) about {format_subject(subject)}.\n\n"
    )
    shots = "".join(format_example(ex, include_answer=True) for ex in dev_examples)
    query = format_example(query_ex, include_answer=False)
    return header + shots + query


def get_abcd_token_ids(tokenizer) -> Dict[str, int]:
    out = {}
    for ch in "ABCD":
        ids = tokenizer.encode(" " + ch, add_special_tokens=False)
        if len(ids) == 0:
            ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(ids) == 0:
            raise RuntimeError(f"Failed to tokenize answer label {ch!r}")
        out[ch] = ids[-1]
    return out


@torch.no_grad()
def score_one_prompt(
    model,
    tokenizer,
    prompt: str,
    answer_token_ids: Dict[str, int],
    device: torch.device,
    max_length: int,
    autocast_dtype: torch.dtype,
) -> Dict[str, float]:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with inference_autocast_context(device, autocast_dtype):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    last_logits = logits[0, -1, :].float()

    scores = {ch: last_logits[tok_id].item() for ch, tok_id in answer_token_ids.items()}
    pred = max(scores.items(), key=lambda kv: kv[1])[0]
    scores["pred"] = pred
    return scores


@dataclass
class EvalConfig:
    model_path: str
    output_dir: str
    weight_rank: int
    act_rank: int
    qtype: str = "nvfp4e2m1bnosr"
    q_scalar_w: float = 1.0
    q_scalar_x: float = 1.0
    svd_method_w: str = "randomized"
    svd_method_x: str = "full"
    dtype: str = "bfloat16"
    store_dtype: str = "bfloat16"
    max_length: int = 2048
    ntrain: int = 5
    split: str = "validation"
    max_eval_samples_per_subject: int = -1
    subject_filter: str = ""
    seed: int = 42
    verbose: bool = False


def evaluate_mmlu(cfg: EvalConfig) -> Dict[str, Any]:
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = maybe_autocast_dtype(cfg.dtype)
    store_dtype = maybe_autocast_dtype(cfg.store_dtype)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_causal_lm_compat(cfg.model_path, compute_dtype)
    model.to(device)
    model.eval()

    replaced = patch_model_linears(
        model=model,
        weight_rank=cfg.weight_rank,
        act_rank=cfg.act_rank,
        qtype=cfg.qtype,
        q_scalar_w=cfg.q_scalar_w,
        q_scalar_x=cfg.q_scalar_x,
        svd_method_w=cfg.svd_method_w,
        svd_method_x=cfg.svd_method_x,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
        verbose=cfg.verbose,
    )

    meta = {
        "replaced_num_modules": len(replaced),
        "replaced_modules": replaced,
    }
    json_dump(meta, os.path.join(cfg.output_dir, "patched_modules.json"))

    ds = load_dataset("cais/mmlu", "all")
    dev_ds = ds["dev"]
    eval_ds = ds[cfg.split]

    subjects = list_subjects_from_dataset(eval_ds)
    if cfg.subject_filter.strip():
        pattern = re.compile(cfg.subject_filter)
        subjects = [s for s in subjects if pattern.search(s)]

    dev_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    eval_by_subject: Dict[str, List[Dict[str, Any]]] = {}
    for s in subjects:
        dev_by_subject[s] = [x for x in dev_ds if x["subject"] == s]
        cur_eval = [x for x in eval_ds if x["subject"] == s]
        if cfg.max_eval_samples_per_subject > 0:
            cur_eval = cur_eval[:cfg.max_eval_samples_per_subject]
        eval_by_subject[s] = cur_eval

    answer_token_ids = get_abcd_token_ids(tokenizer)

    per_subject = {}
    total_correct = 0
    total_count = 0
    detail_path = os.path.join(cfg.output_dir, "detail.jsonl")
    if os.path.exists(detail_path):
        os.remove(detail_path)

    start_time = time.time()

    for subject in tqdm(subjects, desc=f"Eval rW={cfg.weight_rank}, rX={cfg.act_rank}"):
        dev_examples = dev_by_subject[subject][:cfg.ntrain]
        cur_eval = eval_by_subject[subject]

        correct = 0
        count = 0

        for ex in cur_eval:
            prompt = build_fewshot_prompt(subject, dev_examples, ex)
            scores = score_one_prompt(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                answer_token_ids=answer_token_ids,
                device=device,
                max_length=cfg.max_length,
                autocast_dtype=compute_dtype,
            )
            gold = normalize_answer(ex["answer"])
            pred = scores["pred"]
            ok = int(pred == gold)

            row = {
                "subject": subject,
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "question": ex["question"],
                "choices": ex["choices"],
                "scores": {k: v for k, v in scores.items() if k in "ABCD"},
            }
            jsonl_append(row, detail_path)

            correct += ok
            count += 1

        acc = correct / max(count, 1)
        per_subject[subject] = {
            "acc": acc,
            "correct": correct,
            "count": count,
        }
        total_correct += correct
        total_count += count

    end_time = time.time()
    overall_acc = total_correct / max(total_count, 1)

    result = {
        "config": asdict(cfg),
        "device": str(device),
        "overall_acc": overall_acc,
        "total_correct": total_correct,
        "total_count": total_count,
        "per_subject": per_subject,
        "elapsed_sec": end_time - start_time,
        "patched_num_modules": len(replaced),
    }

    json_dump(result, os.path.join(cfg.output_dir, "summary.json"))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# =========================
# Grid scheduler
# =========================
def worker_output_dir(root: str, weight_rank: int, act_rank: int) -> str:
    return os.path.join(root, f"wrank_{weight_rank}_xrank_{act_rank}")


def spawn_one_worker(
    script_path: str,
    gpu_id: int,
    base_args: argparse.Namespace,
    weight_rank: int,
    act_rank: int,
) -> subprocess.Popen:
    outdir = worker_output_dir(base_args.output_dir, weight_rank, act_rank)
    ensure_dir(outdir)

    cmd = [
        shutil.which("python") or "python",
        script_path,
        "--mode", "worker",
        "--model_path", base_args.model_path,
        "--output_dir", outdir,
        "--weight_rank", str(weight_rank),
        "--act_rank", str(act_rank),
        "--qtype", base_args.qtype,
        "--q_scalar_w", str(base_args.q_scalar_w),
        "--q_scalar_x", str(base_args.q_scalar_x),
        "--svd_method_w", base_args.svd_method_w,
        "--svd_method_x", base_args.svd_method_x,
        "--dtype", base_args.dtype,
        "--store_dtype", base_args.store_dtype,
        "--max_length", str(base_args.max_length),
        "--ntrain", str(base_args.ntrain),
        "--split", base_args.split,
        "--max_eval_samples_per_subject", str(base_args.max_eval_samples_per_subject),
        "--subject_filter", base_args.subject_filter,
        "--seed", str(base_args.seed),
    ]
    if base_args.verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(outdir, "worker.log")
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=logf,
        stderr=subprocess.STDOUT,
    )
    proc._log_file_handle = logf  # type: ignore[attr-defined]
    proc._assigned_gpu = gpu_id    # type: ignore[attr-defined]
    proc._wrank = weight_rank      # type: ignore[attr-defined]
    proc._xrank = act_rank         # type: ignore[attr-defined]
    return proc


def run_grid_scheduler(args: argparse.Namespace) -> None:
    ranks = [int(x) for x in args.rank_grid.split(",") if x.strip()]
    tasks = [(wr, xr) for wr in ranks for xr in ranks]

    ensure_dir(args.output_dir)
    status_path = os.path.join(args.output_dir, "grid_status.jsonl")
    if os.path.exists(status_path):
        os.remove(status_path)

    script_path = os.path.abspath(__file__)
    free_gpus = queue.Queue()
    for gid in parse_gpu_ids(args.gpu_ids):
        free_gpus.put(gid)

    running: List[subprocess.Popen] = []
    pending = list(tasks)

    print(f"[Scheduler] total tasks = {len(pending)}")
    print(f"[Scheduler] GPUs = {args.gpu_ids}")

    while pending or running:
        # 尽量填满 GPU
        while pending and not free_gpus.empty():
            gid = free_gpus.get()
            wr, xr = pending.pop(0)
            outdir = worker_output_dir(args.output_dir, wr, xr)
            summary_path = os.path.join(outdir, "summary.json")

            if os.path.exists(summary_path):
                rec = {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "weight_rank": wr,
                    "act_rank": xr,
                    "gpu": gid,
                    "status": "skip_exists",
                    "summary_path": summary_path,
                }
                jsonl_append(rec, status_path)
                free_gpus.put(gid)
                continue

            proc = spawn_one_worker(script_path, gid, args, wr, xr)
            running.append(proc)
            rec = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "weight_rank": wr,
                "act_rank": xr,
                "gpu": gid,
                "status": "launched",
                "pid": proc.pid,
                "output_dir": outdir,
            }
            jsonl_append(rec, status_path)
            print(f"[Launch] GPU {gid} <- (w={wr}, x={xr}), pid={proc.pid}")

        # 回收完成任务
        still_running = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append(proc)
                continue

            gid = proc._assigned_gpu  # type: ignore[attr-defined]
            wr = proc._wrank          # type: ignore[attr-defined]
            xr = proc._xrank          # type: ignore[attr-defined]
            proc._log_file_handle.close()  # type: ignore[attr-defined]
            free_gpus.put(gid)

            rec = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "weight_rank": wr,
                "act_rank": xr,
                "gpu": gid,
                "status": "finished" if ret == 0 else "failed",
                "returncode": ret,
                "pid": proc.pid,
                "output_dir": worker_output_dir(args.output_dir, wr, xr),
            }
            jsonl_append(rec, status_path)
            print(f"[Done] GPU {gid} <- (w={wr}, x={xr}), returncode={ret}")

        running = still_running
        time.sleep(5)

    print("[Scheduler] all tasks finished.")


# =========================
# CLI
# =========================
def build_parser():
    p = argparse.ArgumentParser()

    p.add_argument("--mode", type=str, default="grid", choices=["grid", "worker"])

    # common
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--qtype", type=str, default="nvfp4e2m1bnosr")
    p.add_argument("--q_scalar_w", type=float, default=1.0)
    p.add_argument("--q_scalar_x", type=float, default=1.0)
    p.add_argument("--svd_method_w", type=str, default="randomized", choices=["full", "randomized"])
    p.add_argument("--svd_method_x", type=str, default="full", choices=["full", "randomized"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--store_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--ntrain", type=int, default=5)
    p.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    p.add_argument("--max_eval_samples_per_subject", type=int, default=-1)
    p.add_argument("--subject_filter", type=str, default="")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")

    # worker
    p.add_argument("--weight_rank", type=int, default=0)
    p.add_argument("--act_rank", type=int, default=0)

    # grid
    p.add_argument("--rank_grid", type=str, default="0,5,10,15,20,25,30,40,50,60,80,100")
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")

    return p


def main():
    args = build_parser().parse_args()

    if args.mode == "grid":
        run_grid_scheduler(args)
        return

    cfg = EvalConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        weight_rank=args.weight_rank,
        act_rank=args.act_rank,
        qtype=args.qtype,
        q_scalar_w=args.q_scalar_w,
        q_scalar_x=args.q_scalar_x,
        svd_method_w=args.svd_method_w,
        svd_method_x=args.svd_method_x,
        dtype=args.dtype,
        store_dtype=args.store_dtype,
        max_length=args.max_length,
        ntrain=args.ntrain,
        split=args.split,
        max_eval_samples_per_subject=args.max_eval_samples_per_subject,
        subject_filter=args.subject_filter,
        seed=args.seed,
        verbose=args.verbose,
    )
    result = evaluate_mmlu(cfg)
    print(json.dumps({
        "overall_acc": result["overall_acc"],
        "total_correct": result["total_correct"],
        "total_count": result["total_count"],
        "weight_rank": cfg.weight_rank,
        "act_rank": cfg.act_rank,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
