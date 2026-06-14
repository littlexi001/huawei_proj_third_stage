#!/usr/bin/env python3
"""Download and export small MMLU / GSM8K subsets for local calibration.

Examples:
  python3 scripts/download_eval_data.py --dataset mmlu --split validation --limit 256
  python3 scripts/download_eval_data.py --dataset gsm8k --split test --limit 256
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mmlu", "gsm8k", "all"], default="all")
    parser.add_argument("--split", default=None, help="mmlu: validation/test/dev; gsm8k: train/test")
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default="data/hf_cache")
    parser.add_argument("--out-dir", default="data/eval")
    return parser.parse_args()


def write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_rows(ds: Any, limit: int, seed: int) -> list[dict[str, Any]]:
    n = len(ds)
    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    indices = indices[: min(limit, n)]
    return [dict(ds[i]) for i in indices]


def download_mmlu(args: argparse.Namespace) -> Path:
    split = args.split or "validation"
    ds = load_dataset("cais/mmlu", "all", split=split, cache_dir=args.cache_dir)
    rows = sample_rows(ds, args.limit, args.seed)
    out_path = Path(args.out_dir) / f"mmlu_{split}_{len(rows)}.jsonl"
    write_jsonl(rows, out_path)
    return out_path


def download_gsm8k(args: argparse.Namespace) -> Path:
    split = args.split or "test"
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=args.cache_dir)
    rows = sample_rows(ds, args.limit, args.seed)
    out_path = Path(args.out_dir) / f"gsm8k_{split}_{len(rows)}.jsonl"
    write_jsonl(rows, out_path)
    return out_path


def main() -> None:
    args = parse_args()
    written: list[Path] = []
    if args.dataset in ("mmlu", "all"):
        mmlu_args = argparse.Namespace(**vars(args))
        if args.dataset == "all":
            mmlu_args.split = "validation"
        written.append(download_mmlu(mmlu_args))
    if args.dataset in ("gsm8k", "all"):
        gsm_args = argparse.Namespace(**vars(args))
        if args.dataset == "all":
            gsm_args.split = "test"
        written.append(download_gsm8k(gsm_args))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
