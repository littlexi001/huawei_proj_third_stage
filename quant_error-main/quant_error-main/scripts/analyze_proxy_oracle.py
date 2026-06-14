#!/usr/bin/env python3
"""Compare calibration proxy scores with held-out oracle sweep metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--max-accuracy-drop", type=float, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "pearson": float(np.corrcoef(x, y)[0, 1]),
        "spearman": float(np.corrcoef(rankdata(x), rankdata(y))[0, 1]),
    }


def main() -> None:
    args = parse_args()
    proxy_data = json.loads(Path(args.proxy).read_text(encoding="utf-8"))
    proxy = {
        (int(row["ka"]), int(row["kp"])): float(row["proxy_score"])
        for row in proxy_data["candidates"]
    }

    baseline = None
    oracle = {}
    with open(args.oracle, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["kind"] == "baseline":
                baseline = row
            elif row["kind"] == "config":
                oracle[(int(row["ka"]), int(row["kp"]))] = row

    keys = sorted(set(proxy) & set(oracle))
    if len(keys) < 3:
        raise RuntimeError("Need at least three completed oracle configurations")

    proxy_values = np.array([proxy[key] for key in keys])
    metrics = {}
    for metric in ["hidden_relative_mse", "logit_relative_mse", "accuracy_drop"]:
        values = np.array([float(oracle[key][metric]) for key in keys])
        metrics[metric] = correlation(proxy_values, values)

    feasible = [
        row for row in oracle.values()
        if float(row["accuracy_drop"]) <= args.max_accuracy_drop
    ]
    oracle_best = min(
        feasible,
        key=lambda row: (float(row["rank_cost"]), float(row["accuracy_drop"])),
    ) if feasible else None

    result = {
        "baseline": baseline,
        "num_common_configs": len(keys),
        "correlations": metrics,
        "max_accuracy_drop": args.max_accuracy_drop,
        "num_feasible_configs": len(feasible),
        "oracle_min_cost_feasible": oracle_best,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
