#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Run ka+kb low-rank/residual quantization scans on GSM8K.

This script reuses the quantized Linear wrapper from
qwen3_svd_residual_nvfp4_mmlu.py. Rank naming follows the project convention:

    ka = activation rank = act_rank
    kb = weight rank = weight_rank

Modes:
    download_data: download/cache gsm8k/main via HuggingFace datasets
    worker:        evaluate one (ka, kb) point or an unquantized baseline
    grid:          dispatch a ka+kb grid across multiple GPUs
"""

import argparse
import gc
import json
import os
import queue
import re
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from qwen3_svd_residual_nvfp4_mmlu import (
    json_dump,
    jsonl_append,
    load_causal_lm_compat,
    maybe_autocast_dtype,
    parse_gpu_ids,
    patch_model_linears,
    set_quant_blocksize,
    set_seed,
)


ANSWER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def inference_autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        text = text.split("####")[-1]
    matches = ANSWER_RE.findall(text)
    if not matches:
        return ""
    return matches[-1].replace(",", "").strip()


def format_gsm8k_example(ex: Dict[str, Any], include_answer: bool = True) -> str:
    text = f"Question: {ex['question'].strip()}\nAnswer:"
    if include_answer:
        text += f" {ex['answer'].strip()}\n\n"
    else:
        text += " "
    return text


def build_gsm8k_prompt(train_examples: List[Dict[str, Any]], query_ex: Dict[str, Any]) -> str:
    header = (
        "Solve the following grade school math problems. "
        "Give the reasoning, then put the final numeric answer after ####.\n\n"
    )
    shots = "".join(format_gsm8k_example(ex, include_answer=True) for ex in train_examples)
    query = format_gsm8k_example(query_ex, include_answer=False)
    return header + shots + query


@torch.no_grad()
def generate_one_prompt(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    autocast_dtype: torch.dtype,
) -> str:
    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with inference_autocast_context(device, autocast_dtype):
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    gen_ids = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


@dataclass
class EvalConfig:
    model_path: str
    output_dir: str
    weight_rank: int
    act_rank: int
    qtype: str = "hif4_0418"
    blocksize: int = 16
    q_scalar_w: float = 1.0
    q_scalar_x: float = 1.0
    svd_method_w: str = "randomized"
    svd_method_x: str = "randomized"
    dtype: str = "bfloat16"
    store_dtype: str = "bfloat16"
    max_length: int = 2048
    max_new_tokens: int = 256
    ntrain: int = 8
    split: str = "test"
    max_eval_samples: int = -1
    seed: int = 42
    no_quant: bool = False
    verbose: bool = False


def download_gsm8k_dataset() -> None:
    ds = load_dataset("gsm8k", "main")
    print(json.dumps({split: len(ds[split]) for split in ds.keys()}, indent=2))


def evaluate_gsm8k(cfg: EvalConfig) -> Dict[str, Any]:
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

    if cfg.no_quant:
        replaced = []
    else:
        set_quant_blocksize(cfg.blocksize)
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

    json_dump(
        {
            "no_quant": cfg.no_quant,
            "replaced_num_modules": len(replaced),
            "replaced_modules": replaced,
        },
        os.path.join(cfg.output_dir, "patched_modules.json"),
    )

    ds = load_dataset("gsm8k", "main")
    train_ds = ds["train"]
    eval_ds = ds[cfg.split]
    train_examples = [dict(x) for x in train_ds.select(range(min(cfg.ntrain, len(train_ds))))]
    eval_examples = [dict(x) for x in eval_ds]
    if cfg.max_eval_samples > 0:
        eval_examples = eval_examples[:cfg.max_eval_samples]

    detail_path = os.path.join(cfg.output_dir, "detail.jsonl")
    if os.path.exists(detail_path):
        os.remove(detail_path)

    total_correct = 0
    total_count = 0
    start_time = time.time()
    desc = "GSM8K baseline" if cfg.no_quant else f"GSM8K kb={cfg.weight_rank}, ka={cfg.act_rank}"

    for idx, ex in enumerate(tqdm(eval_examples, desc=desc)):
        prompt = build_gsm8k_prompt(train_examples, ex)
        generation = generate_one_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_length=cfg.max_length,
            max_new_tokens=cfg.max_new_tokens,
            autocast_dtype=compute_dtype,
        )
        gold = extract_gsm8k_answer(ex["answer"])
        pred = extract_gsm8k_answer(generation)
        ok = int(pred == gold)

        jsonl_append(
            {
                "index": idx,
                "gold": gold,
                "pred": pred,
                "correct": ok,
                "question": ex["question"],
                "gold_answer": ex["answer"],
                "generation": generation,
            },
            detail_path,
        )
        total_correct += ok
        total_count += 1

    elapsed = time.time() - start_time
    result = {
        "config": asdict(cfg),
        "dataset": "gsm8k/main",
        "split": cfg.split,
        "device": str(device),
        "overall_acc": total_correct / max(total_count, 1),
        "total_correct": total_correct,
        "total_count": total_count,
        "elapsed_sec": elapsed,
        "no_quant": cfg.no_quant,
        "patched_num_modules": len(replaced),
    }
    json_dump(result, os.path.join(cfg.output_dir, "summary.json"))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def worker_output_dir(root: str, weight_rank: int, act_rank: int) -> str:
    return os.path.join(root, f"kb_{weight_rank}_ka_{act_rank}")


def baseline_output_dir(root: str) -> str:
    return os.path.join(root, "baseline_unquantized")


def spawn_one_worker(
    script_path: str,
    gpu_id: int,
    base_args: argparse.Namespace,
    weight_rank: int,
    act_rank: int,
    no_quant: bool = False,
) -> subprocess.Popen:
    outdir = baseline_output_dir(base_args.output_dir) if no_quant else worker_output_dir(base_args.output_dir, weight_rank, act_rank)
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
        "--blocksize", str(base_args.blocksize),
        "--q_scalar_w", str(base_args.q_scalar_w),
        "--q_scalar_x", str(base_args.q_scalar_x),
        "--svd_method_w", base_args.svd_method_w,
        "--svd_method_x", base_args.svd_method_x,
        "--dtype", base_args.dtype,
        "--store_dtype", base_args.store_dtype,
        "--max_length", str(base_args.max_length),
        "--max_new_tokens", str(base_args.max_new_tokens),
        "--ntrain", str(base_args.ntrain),
        "--split", base_args.split,
        "--max_eval_samples", str(base_args.max_eval_samples),
        "--seed", str(base_args.seed),
    ]
    if no_quant:
        cmd.append("--no_quant")
    if base_args.verbose:
        cmd.append("--verbose")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(outdir, "worker.log")
    logf = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    proc._log_file_handle = logf  # type: ignore[attr-defined]
    proc._assigned_gpu = gpu_id  # type: ignore[attr-defined]
    proc._wrank = weight_rank  # type: ignore[attr-defined]
    proc._xrank = act_rank  # type: ignore[attr-defined]
    proc._no_quant = no_quant  # type: ignore[attr-defined]
    return proc


def run_grid_scheduler(args: argparse.Namespace) -> None:
    ranks = [int(x) for x in args.rank_grid.split(",") if x.strip()]
    tasks = [(False, kb, ka) for kb in ranks for ka in ranks]
    if args.run_unquantized_baseline:
        tasks.insert(0, (True, 0, 0))

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
        while pending and not free_gpus.empty():
            gid = free_gpus.get()
            no_quant, kb, ka = pending.pop(0)
            outdir = baseline_output_dir(args.output_dir) if no_quant else worker_output_dir(args.output_dir, kb, ka)
            summary_path = os.path.join(outdir, "summary.json")

            if os.path.exists(summary_path):
                jsonl_append(
                    {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "no_quant": no_quant,
                        "kb": kb,
                        "ka": ka,
                        "gpu": gid,
                        "status": "skip_exists",
                        "summary_path": summary_path,
                    },
                    status_path,
                )
                free_gpus.put(gid)
                continue

            proc = spawn_one_worker(script_path, gid, args, kb, ka, no_quant=no_quant)
            running.append(proc)
            label = "baseline_unquantized" if no_quant else f"kb={kb}, ka={ka}"
            jsonl_append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "no_quant": no_quant,
                    "kb": kb,
                    "ka": ka,
                    "gpu": gid,
                    "status": "launched",
                    "pid": proc.pid,
                    "output_dir": outdir,
                },
                status_path,
            )
            print(f"[Launch] GPU {gid} <- ({label}), pid={proc.pid}", flush=True)

        still_running = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                still_running.append(proc)
                continue

            gid = proc._assigned_gpu  # type: ignore[attr-defined]
            kb = proc._wrank  # type: ignore[attr-defined]
            ka = proc._xrank  # type: ignore[attr-defined]
            no_quant = proc._no_quant  # type: ignore[attr-defined]
            proc._log_file_handle.close()  # type: ignore[attr-defined]
            free_gpus.put(gid)

            outdir = baseline_output_dir(args.output_dir) if no_quant else worker_output_dir(args.output_dir, kb, ka)
            label = "baseline_unquantized" if no_quant else f"kb={kb}, ka={ka}"
            jsonl_append(
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "no_quant": no_quant,
                    "kb": kb,
                    "ka": ka,
                    "gpu": gid,
                    "status": "finished" if ret == 0 else "failed",
                    "returncode": ret,
                    "pid": proc.pid,
                    "output_dir": outdir,
                },
                status_path,
            )
            print(f"[Done] GPU {gid} <- ({label}), returncode={ret}", flush=True)

        running = still_running
        time.sleep(5)

    print("[Scheduler] all tasks finished.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="grid", choices=["grid", "worker", "download_data"])

    p.add_argument("--model_path", type=str, default="/mnt/workspace/Qwen3-8B")
    p.add_argument("--output_dir", type=str, default="./outputs/qwen3_svd_resid_gsm8k_grid_8B_hif4")
    p.add_argument("--qtype", type=str, default="hif4_0418")
    p.add_argument("--blocksize", type=int, default=16)
    p.add_argument("--q_scalar_w", type=float, default=1.0)
    p.add_argument("--q_scalar_x", type=float, default=1.0)
    p.add_argument("--svd_method_w", type=str, default="randomized", choices=["full", "randomized"])
    p.add_argument("--svd_method_x", type=str, default="randomized", choices=["full", "randomized"])
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--store_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--ntrain", type=int, default=8)
    p.add_argument("--split", type=str, default="test", choices=["train", "test"])
    p.add_argument("--max_eval_samples", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_quant", action="store_true")
    p.add_argument("--verbose", action="store_true")

    p.add_argument("--weight_rank", type=int, default=0, help="kb: weight rank")
    p.add_argument("--act_rank", type=int, default=0, help="ka: activation rank")
    p.add_argument("--rank_grid", type=str, default="0,5,10,15,20,25,30,40,50,60,80,100")
    p.add_argument("--gpu_ids", type=str, default="0,1,2,3,4,5,6,7")
    p.add_argument("--run_unquantized_baseline", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "download_data":
        download_gsm8k_dataset()
        return
    if args.mode == "grid":
        run_grid_scheduler(args)
        return

    cfg = EvalConfig(
        model_path=args.model_path,
        output_dir=args.output_dir,
        weight_rank=args.weight_rank,
        act_rank=args.act_rank,
        qtype=args.qtype,
        blocksize=args.blocksize,
        q_scalar_w=args.q_scalar_w,
        q_scalar_x=args.q_scalar_x,
        svd_method_w=args.svd_method_w,
        svd_method_x=args.svd_method_x,
        dtype=args.dtype,
        store_dtype=args.store_dtype,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        ntrain=args.ntrain,
        split=args.split,
        max_eval_samples=args.max_eval_samples,
        seed=args.seed,
        no_quant=args.no_quant,
        verbose=args.verbose,
    )
    result = evaluate_gsm8k(cfg)
    print(json.dumps({
        "overall_acc": result["overall_acc"],
        "total_correct": result["total_correct"],
        "total_count": result["total_count"],
        "kb": cfg.weight_rank,
        "ka": cfg.act_rank,
        "blocksize": cfg.blocksize,
        "no_quant": cfg.no_quant,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
