#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        type=str,
        required=True,
        help="实验根目录，例如 qwen3_svd_resid_nvfp4_mmlu_grid",
    )
    p.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="baseline accuracy；默认优先读取 root/baseline_unquantized/summary.json，否则使用 0.40",
    )
    p.add_argument(
        "--save_dir",
        type=str,
        default="",
        help="图像保存目录；默认保存在 root/plots",
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="输出图片 DPI",
    )
    return p.parse_args()


def extract_json_block_from_log(log_path: Path):
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r'\{[\s\S]*?"overall_acc"[\s\S]*?\}', text)
    if not matches:
        return None

    for s in reversed(matches):
        try:
            obj = json.loads(s)
            if "overall_acc" in obj:
                return obj
        except Exception:
            continue
    return None


def extract_acc_from_log(log_path: Path):
    obj = extract_json_block_from_log(log_path)
    if obj is None:
        return None
    try:
        return float(obj["overall_acc"])
    except Exception:
        return None


def collect_results(root: Path):
    pattern = re.compile(r"wrank_(\d+)_xrank_(\d+)")
    results = {}

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = pattern.fullmatch(sub.name)
        if not m:
            continue

        wr = int(m.group(1))
        xr = int(m.group(2))
        log_path = sub / "worker.log"
        if not log_path.exists():
            print(f"[WARN] missing log: {log_path}")
            continue

        acc = extract_acc_from_log(log_path)
        if acc is None:
            print(f"[WARN] failed to parse overall_acc from: {log_path}")
            continue

        results[(wr, xr)] = acc

    return results


def load_unquantized_baseline(root: Path):
    summary_path = root / "baseline_unquantized" / "summary.json"
    if not summary_path.exists():
        return None
    try:
        obj = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(obj["overall_acc"])
    except Exception as exc:
        print(f"[WARN] failed to parse baseline summary {summary_path}: {exc}")
        return None


def build_matrix(results):
    wranks = sorted({k[0] for k in results.keys()})
    xranks = sorted({k[1] for k in results.keys()})

    mat = np.full((len(wranks), len(xranks)), np.nan, dtype=float)

    w_to_i = {w: i for i, w in enumerate(wranks)}
    x_to_j = {x: j for j, x in enumerate(xranks)}

    for (w, x), acc in results.items():
        mat[w_to_i[w], x_to_j[x]] = acc

    return wranks, xranks, mat


def annotate_heatmap(ax, mat, fmt="{:.3f}", fontsize=8):
    finite_vals = mat[np.isfinite(mat)]
    mid = np.nanmean(finite_vals) if finite_vals.size > 0 else 0.0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                text = "NA"
                color = "black"
            else:
                text = fmt.format(val)
                color = "white" if val < mid else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=fontsize, color=color)


def plot_one_heatmap(
    mat,
    wranks,
    xranks,
    title,
    cbar_label,
    save_path,
    value_fmt="{:.3f}",
    cmap="viridis",
    dpi=220,
):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap)

    ax.set_xticks(np.arange(len(xranks)))
    ax.set_yticks(np.arange(len(wranks)))
    ax.set_xticklabels([str(x) for x in xranks])
    ax.set_yticklabels([str(w) for w in wranks])

    ax.set_xlabel("Activation rank (rX)")
    ax.set_ylabel("Weight rank (rW)")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    annotate_heatmap(ax, mat, fmt=value_fmt, fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_csv(path: Path, row_labels, col_labels, mat):
    with open(path, "w", encoding="utf-8") as f:
        f.write("wrank/xrank," + ",".join(map(str, col_labels)) + "\n")
        for i, r in enumerate(row_labels):
            vals = []
            for j in range(len(col_labels)):
                v = mat[i, j]
                vals.append("" if np.isnan(v) else f"{v:.6f}")
            f.write(str(r) + "," + ",".join(vals) + "\n")


def main():
    args = parse_args()
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    save_dir = Path(args.save_dir) if args.save_dir else (root / "plots")
    save_dir.mkdir(parents=True, exist_ok=True)

    results = collect_results(root)
    if not results:
        raise RuntimeError("No valid results found. Please check root path and worker.log contents.")

    wranks, xranks, acc_mat = build_matrix(results)
    baseline = args.baseline
    if baseline is None:
        baseline = load_unquantized_baseline(root)
    if baseline is None:
        baseline = 0.40
        print("[WARN] no --baseline and no baseline_unquantized/summary.json found; using 0.40")

    # rel_pct_mat = (acc_mat - args.baseline) / args.baseline * 100.0
    rel_pct_mat = (acc_mat - baseline)

    np.save(save_dir / "acc_matrix.npy", acc_mat)
    np.save(save_dir / "rel_pct_matrix.npy", rel_pct_mat)
    save_csv(save_dir / "acc_matrix.csv", wranks, xranks, acc_mat)
    save_csv(save_dir / "rel_pct_matrix.csv", wranks, xranks, rel_pct_mat)

    plot_one_heatmap(
        mat=acc_mat,
        wranks=wranks,
        xranks=xranks,
        title="MMLU overall_acc Heatmap",
        cbar_label="overall_acc",
        save_path=save_dir / "heatmap_overall_acc.png",
        value_fmt="{:.5f}",
        cmap="viridis",
        dpi=args.dpi,
    )

    plot_one_heatmap(
        mat=rel_pct_mat,
        wranks=wranks,
        xranks=xranks,
        title=f"MMLU Relative Change vs Baseline={baseline:.5f}",
        cbar_label="relative change",
        save_path=save_dir / "heatmap_relative_percent.png",
        value_fmt="{:+.3f}",
        cmap="coolwarm",
        dpi=args.dpi,
    )

    plot_one_heatmap(
        mat=acc_mat,
        wranks=wranks,
        xranks=xranks,
        title="MMLU overall_acc Heatmap",
        cbar_label="overall_acc",
        save_path=save_dir / "heatmap_overall_acc.pdf",
        value_fmt="{:.3f}",
        cmap="viridis",
        dpi=args.dpi,
    )

    plot_one_heatmap(
        mat=rel_pct_mat,
        wranks=wranks,
        xranks=xranks,
        title=f"MMLU Relative Change vs Baseline={baseline:.3f} (%)",
        cbar_label="relative change (%)",
        save_path=save_dir / "heatmap_relative_percent.pdf",
        value_fmt="{:+.1f}",
        cmap="coolwarm",
        dpi=args.dpi,
    )

    print("[Done]")
    print(f"results found: {len(results)}")
    print(f"save_dir: {save_dir}")
    print(f"baseline: {baseline:.6f}")
    print(f"wranks: {wranks}")
    print(f"xranks: {xranks}")


if __name__ == "__main__":
    main()
