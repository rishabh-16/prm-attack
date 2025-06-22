#!/usr/bin/env python3
# plot_length_deltas.py
"""
Plot the distribution of relative change in step length (augmented – original)
for one or more evaluation runs.

Example
-------
python plot_length_deltas.py chatgpt_batch_concise chatgpt_batch_verbose \
                             chatgpt_batch_rephrase \
                             --output_path length_deltas.png
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create histogram + KDE plot of relative length deltas."
    )
    p.add_argument(
        "experiments",
        nargs="+",
        help="Folder(s) that contain an attack.parquet each."
    )
    p.add_argument(
        "--output_path",
        help="Where to save the PNG (default: <first-exp>_length_deltas.png)."
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  Data helpers
# --------------------------------------------------------------------------- #
FILTER_SPLIT = "gsm8k"
PARQUET_NAME = "attack.parquet"


def load_and_filter(pq_path: Path) -> pd.DataFrame:
    """Load parquet and apply the notebook’s filters."""
    df = pd.read_parquet(pq_path)

    keep = (
        (df["equivalence"])
        & (df.apply(lambda r: len(r["aug_steps"]) == len(r["steps"]), axis=1))
        & (df["aug_steps"].str.len() > 0)
        & (df["steps"].str.len() > 0)
        & (df["final_answer_correct"])
        & (df["split"] == FILTER_SPLIT)
    )
    return df.loc[keep].copy()


def compute_length_deltas(df: pd.DataFrame) -> np.ndarray:
    """Return (augLen-origLen)/origLen for every aligned step."""
    deltas = []
    for _, row in df.iterrows():
        for o_step, a_step in zip(row["steps"], row["aug_steps"]):
            o_len = len(o_step)
            if o_len == 0:
                continue
            deltas.append((len(a_step) - o_len) / o_len)
    return np.asarray(deltas, dtype=float)


# --------------------------------------------------------------------------- #
#  Plot helpers – identical style to plot_reward_deltas.py
# --------------------------------------------------------------------------- #
DEFAULT_COLORS = ("blue", "green", "red", "orange", "purple", "brown")


def add_curve(data: np.ndarray, label: str, color: str) -> None:
    """Plot histogram + KDE on current axes (reward-deltas style)."""
    plt.hist(
        data,
        bins=50,
        density=True,
        alpha=0.30,
        edgecolor=color,
        color=color,
    )

    kde = stats.gaussian_kde(data)
    xs = np.linspace(data.min(), data.max(), 200)
    plt.plot(xs, kde(xs), lw=2, color=color, label=f"{label} (μ={data.mean():.3f})")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    exp_dirs: Sequence[Path] = [Path(d) for d in args.experiments]
    if not exp_dirs:
        raise RuntimeError("No experiment folders supplied.")

    # Decide output file
    out_path = (
        Path(args.output_path)
        if args.output_path
        else exp_dirs[0].with_name(exp_dirs[0].stem + "_length_deltas.png")
    )

    plt.figure(figsize=(10, 6))

    for idx, exp_dir in enumerate(exp_dirs):
        pq_file = exp_dir
        if not pq_file.exists():
            print(f"[WARN] {pq_file} not found – skipping.")
            continue

        try:
            df = load_and_filter(pq_file)
            print(f"{exp_dir.name}: {len(df)} samples after filtering")

            deltas = compute_length_deltas(df)
            if deltas.size == 0:
                print(f"[WARN] {exp_dir.name}: no usable steps – skipping.")
                continue

            add_curve(deltas, exp_dir.name, DEFAULT_COLORS[idx % len(DEFAULT_COLORS)])

        except Exception as exc:
            print(f"[ERROR] processing {exp_dir}: {exc}")

    # Final cosmetics – identical to reward plot
    plt.title("Distribution of Relative Step-Length Changes", fontsize=12)
    plt.xlabel("Relative change in length  (fraction of original)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.axvline(0, ls="--", lw=1, color="k", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
