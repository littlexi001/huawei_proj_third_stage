#!/usr/bin/env python3
"""Fit sparse MMLU anchors from proxy MSE and select feasible ka/kp configs.

The intended workflow is:

1. Run `adaptive_metis_rank_search.py` over a dense ka/kp grid to get proxy MSE.
2. Run `adaptive_metis_oracle_sweep.py --pairs ...` on a small anchor set.
3. Use this script to predict the MMLU accuracy drop for every dense-grid
   candidate, keep the candidates under the allowed drop, and report the
   minimum-cost configuration.

Accuracy drops are fractions, so 1% means `--max-accuracy-drop 0.01`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy", required=True, help="JSON from adaptive_metis_rank_search.py")
    parser.add_argument("--anchors", required=True, help="JSONL from adaptive_metis_oracle_sweep.py")
    parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    parser.add_argument("--uncertainty-lambda", type=float, default=1.0)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--ka-cost", type=float, default=1.0)
    parser.add_argument("--kp-cost", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_proxy(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_anchor_drops(path: str) -> tuple[dict[str, Any] | None, dict[tuple[int, int], dict[str, Any]]]:
    baseline = None
    anchors = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("kind") == "baseline":
                baseline = row
            elif row.get("kind") == "config":
                anchors[(int(row["ka"]), int(row["kp"]))] = row
    return baseline, anchors


def aggregate_side_scores(proxy_data: dict[str, Any]) -> tuple[dict[int, float], dict[int, float]]:
    module_stats = proxy_data.get("module_stats") or []
    act_values: dict[int, list[float]] = {}
    weight_values: dict[int, list[float]] = {}

    for stats in module_stats:
        for rank, value in (stats.get("act_scores") or {}).items():
            act_values.setdefault(int(rank), []).append(float(value))
        for rank, value in (stats.get("weight_scores") or {}).items():
            weight_values.setdefault(int(rank), []).append(float(value))

    act_scores = {rank: float(np.mean(values)) for rank, values in act_values.items()}
    weight_scores = {rank: float(np.mean(values)) for rank, values in weight_values.items()}
    return act_scores, weight_scores


def candidate_rows(proxy_data: dict[str, Any]) -> list[dict[str, float]]:
    act_scores, weight_scores = aggregate_side_scores(proxy_data)
    rows = []
    for row in proxy_data["candidates"]:
        ka = int(row["ka"])
        kp = int(row["kp"])
        act_proxy = act_scores.get(ka, math.nan)
        weight_proxy = weight_scores.get(kp, math.nan)
        proxy_score = float(row["proxy_score"])
        if math.isnan(act_proxy) or math.isnan(weight_proxy):
            # Older proxy files may not carry module_stats. In that case fall
            # back to a one-dimensional proxy model.
            act_proxy = proxy_score
            weight_proxy = 0.0
        rows.append(
            {
                "ka": ka,
                "kp": kp,
                "activation_proxy": act_proxy,
                "weight_proxy": weight_proxy,
                "proxy_score": proxy_score,
                "num_modules": int(row.get("num_modules", 0)),
            }
        )
    return rows


def make_features(rows: list[dict[str, float]]) -> np.ndarray:
    act = np.array([row["activation_proxy"] for row in rows], dtype=np.float64)
    weight = np.array([row["weight_proxy"] for row in rows], dtype=np.float64)
    proxy = np.array([row["proxy_score"] for row in rows], dtype=np.float64)
    ka = np.array([row["ka"] for row in rows], dtype=np.float64)
    kp = np.array([row["kp"] for row in rows], dtype=np.float64)

    def zscore(x: np.ndarray) -> np.ndarray:
        std = x.std()
        if std < 1e-12:
            return x * 0.0
        return (x - x.mean()) / std

    act_z = zscore(act)
    weight_z = zscore(weight)
    proxy_z = zscore(proxy)
    ka_z = zscore(ka)
    kp_z = zscore(kp)
    return np.column_stack(
        [
            np.ones(len(rows)),
            act_z,
            weight_z,
            act_z * weight_z,
            proxy_z,
            ka_z,
            kp_z,
        ]
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + penalty, x.T @ y)


def loocv_rmse(x: np.ndarray, y: np.ndarray, ridge: float) -> float:
    if len(y) <= x.shape[1] + 1:
        return float(np.std(y)) if len(y) > 1 else 0.0
    errors = []
    for index in range(len(y)):
        mask = np.ones(len(y), dtype=bool)
        mask[index] = False
        beta = fit_ridge(x[mask], y[mask], ridge)
        errors.append(float(x[index] @ beta - y[index]))
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    args = parse_args()
    proxy_data = load_proxy(args.proxy)
    baseline, anchor_rows = load_anchor_drops(args.anchors)
    rows = candidate_rows(proxy_data)
    by_key = {(int(row["ka"]), int(row["kp"])): row for row in rows}

    train_rows = []
    train_y = []
    for key, anchor in sorted(anchor_rows.items()):
        if key not in by_key:
            continue
        train_rows.append(by_key[key])
        train_y.append(float(anchor["accuracy_drop"]))
    if len(train_rows) < 3:
        raise RuntimeError("Need at least three anchor configs that also exist in the proxy grid")

    x_train = make_features(train_rows)
    y_train = np.array(train_y, dtype=np.float64)
    beta = fit_ridge(x_train, y_train, args.ridge)
    sigma = loocv_rmse(x_train, y_train, args.ridge)

    x_all = make_features(rows)
    predictions = x_all @ beta
    output_rows = []
    for row, pred in zip(rows, predictions):
        conservative_drop = float(pred + args.uncertainty_lambda * sigma)
        rank_cost = args.ka_cost * row["ka"] + args.kp_cost * row["kp"]
        output_rows.append(
            {
                **row,
                "rank_cost": float(rank_cost),
                "predicted_accuracy_drop": float(pred),
                "uncertainty": float(sigma),
                "conservative_accuracy_drop": conservative_drop,
                "feasible": conservative_drop <= args.max_accuracy_drop,
            }
        )

    feasible = [row for row in output_rows if row["feasible"]]
    feasible.sort(
        key=lambda row: (
            row["rank_cost"],
            row["conservative_accuracy_drop"],
            row["predicted_accuracy_drop"],
            row["ka"],
            row["kp"],
        )
    )
    output_rows.sort(
        key=lambda row: (
            row["conservative_accuracy_drop"] > args.max_accuracy_drop,
            row["rank_cost"],
            row["conservative_accuracy_drop"],
            row["ka"],
            row["kp"],
        )
    )

    result = {
        "proxy": args.proxy,
        "anchors": args.anchors,
        "baseline": baseline,
        "num_anchor_configs": len(train_rows),
        "max_accuracy_drop": args.max_accuracy_drop,
        "uncertainty_lambda": args.uncertainty_lambda,
        "loocv_rmse": sigma,
        "coefficients": beta.tolist(),
        "num_candidates": len(output_rows),
        "num_feasible": len(feasible),
        "best_min_cost_feasible": feasible[0] if feasible else None,
        "top_feasible": feasible[: args.top_k],
        "all_candidates": output_rows,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in result if k != "all_candidates"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
