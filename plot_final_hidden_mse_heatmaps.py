#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_FORMAT_ORDER = ["nvfp4", "mxfp4", "hif4", "nvfp8", "mxfp8", "hif8"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv",
        type=str,
        default="./outputs/final_hidden_mse_8B_dclm/summary.csv",
        help="eval_final_hidden_mse.py produced summary.csv",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="Output directory. Default: <csv_parent>/plots",
    )
    p.add_argument(
        "--metrics",
        type=str,
        default="mse,relative_mse",
        help="Comma-separated metric names, e.g. mse,relative_mse,rmse,max_abs",
    )
    p.add_argument(
        "--formats",
        type=str,
        default="",
        help="Comma-separated formats to plot. Default: all formats in csv using a stable order.",
    )
    p.add_argument("--dpi", type=int, default=220)
    p.add_argument("--annotate", action="store_true", help="Draw numeric values in each heatmap cell.")
    p.add_argument("--log", action="store_true", help="Use log10(metric) colors. Non-positive values become NA.")
    return p.parse_args()


def read_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    required = {"format", "weight_rank", "act_rank"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return rows


def to_float(row: Dict[str, str], key: str) -> float:
    val = row.get(key, "")
    if val == "":
        return float("nan")
    return float(val)


def stable_formats(rows: List[Dict[str, str]], requested: str) -> List[str]:
    found = {row["format"] for row in rows}
    if requested.strip():
        formats = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        formats = [fmt for fmt in DEFAULT_FORMAT_ORDER if fmt in found]
        formats.extend(sorted(found - set(formats)))
    missing = [fmt for fmt in formats if fmt not in found]
    if missing:
        raise ValueError(f"Requested formats not found in csv: {missing}")
    return formats


def build_matrix(rows: List[Dict[str, str]], fmt: str, metric: str) -> Tuple[List[int], List[int], np.ndarray]:
    subset = [row for row in rows if row["format"] == fmt]
    if not subset:
        raise ValueError(f"No rows for format={fmt}")

    wranks = sorted({int(row["weight_rank"]) for row in subset})
    xranks = sorted({int(row["act_rank"]) for row in subset})
    mat = np.full((len(wranks), len(xranks)), np.nan, dtype=float)
    w_to_i = {rank: i for i, rank in enumerate(wranks)}
    x_to_j = {rank: j for j, rank in enumerate(xranks)}

    for row in subset:
        if metric not in row:
            raise ValueError(f"CSV missing metric column: {metric}")
        i = w_to_i[int(row["weight_rank"])]
        j = x_to_j[int(row["act_rank"])]
        mat[i, j] = to_float(row, metric)

    return wranks, xranks, mat


def matrix_for_color(mat: np.ndarray, use_log: bool) -> np.ndarray:
    if not use_log:
        return mat
    out = mat.copy()
    out[out <= 0] = np.nan
    return np.log10(out)


def annotate(ax, mat: np.ndarray, use_log: bool) -> None:
    finite = mat[np.isfinite(mat)]
    mid = np.nanmean(finite) if finite.size else 0.0
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                text = "NA"
                color = "black"
            else:
                text = f"{val:.3g}"
                color = "white" if val < mid else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)


def plot_metric(rows, formats, metric: str, save_dir: Path, dpi: int, annotate_values: bool, use_log: bool) -> None:
    matrices = {}
    wranks_by_fmt = {}
    xranks_by_fmt = {}
    color_values = []

    for fmt in formats:
        wranks, xranks, mat = build_matrix(rows, fmt, metric)
        cmat = matrix_for_color(mat, use_log)
        matrices[fmt] = (mat, cmat)
        wranks_by_fmt[fmt] = wranks
        xranks_by_fmt[fmt] = xranks
        finite = cmat[np.isfinite(cmat)]
        if finite.size:
            color_values.append(finite)

        np.save(save_dir / f"{fmt}_{metric}_matrix.npy", mat)
        save_matrix_csv(save_dir / f"{fmt}_{metric}_matrix.csv", wranks, xranks, mat)

    if not color_values:
        raise ValueError(f"No finite values for metric={metric}")

    all_values = np.concatenate(color_values)
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))

    n = len(formats)
    ncols = min(3, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.8 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    im = None

    for idx, fmt in enumerate(formats):
        ax = axes[idx // ncols][idx % ncols]
        mat, cmat = matrices[fmt]
        wranks = wranks_by_fmt[fmt]
        xranks = xranks_by_fmt[fmt]
        im = ax.imshow(cmat, aspect="auto", interpolation="nearest", cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(fmt)
        ax.set_xlabel("Activation rank (rX)")
        ax.set_ylabel("Weight rank (rW)")
        ax.set_xticks(np.arange(len(xranks)))
        ax.set_yticks(np.arange(len(wranks)))
        ax.set_xticklabels([str(x) for x in xranks], rotation=45, ha="right")
        ax.set_yticklabels([str(w) for w in wranks])
        if annotate_values:
            annotate(ax, mat, use_log)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    label = f"log10({metric})" if use_log else metric
    fig.suptitle(f"Final Hidden State {label} Heatmaps", fontsize=14)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.88)
        cbar.set_label(label)

    suffix = "_log" if use_log else ""
    fig.savefig(save_dir / f"heatmaps_{metric}{suffix}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(save_dir / f"heatmaps_{metric}{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)


def save_matrix_csv(path: Path, wranks: List[int], xranks: List[int], mat: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["wrank/xrank", *xranks])
        for i, wr in enumerate(wranks):
            writer.writerow([wr, *["" if np.isnan(v) else f"{v:.10g}" for v in mat[i]]])


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    rows = read_rows(csv_path)
    formats = stable_formats(rows, args.formats)
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    save_dir = Path(args.save_dir) if args.save_dir else csv_path.parent / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plot_metric(rows, formats, metric, save_dir, args.dpi, args.annotate, use_log=False)
        if args.log:
            plot_metric(rows, formats, metric, save_dir, args.dpi, args.annotate, use_log=True)

    print("[Done]")
    print(f"csv: {csv_path}")
    print(f"save_dir: {save_dir}")
    print(f"formats: {formats}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
