#!/usr/bin/env python3
# plot_reward_deltas.py
"""
Generate a Δ-reward density plot from an evaluation parquet file.

Example
-------
python plot_reward_deltas.py \
    --parquet_path attack.parquet \
    --output_path reward_deltas.png \
    --include_first_step
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


SKYWORK_ORIG_COL = "Skywork-o1-Open-PRM-Qwen-2.5-7B"
SKYWORK_AUG_COL  = "Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B--aug_rewards"
QWEN_ORIG_COL    = "Qwen2.5-Math-PRM-7B"
QWEN_AUG_COL     = "Qwen/Qwen2.5-Math-PRM-7B--aug_rewards"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create reward-delta histogram / KDE plot from a parquet evaluation dump."
    )
    parser.add_argument("--parquet_path", required=True, help="Path to the parquet file.")
    parser.add_argument(
        "--output_path",
        help="Where to save the PNG. Defaults to <parquet basename>_reward_deltas.png.",
    )
    parser.add_argument(
        "--include_first_step",
        action="store_true",
        help="Also plot a curve where Skywork rewards are the first entry instead of the mean.",
    )
    return parser.parse_args()


def load_and_filter(path: str | Path) -> pd.DataFrame:
    """Replicate the notebook’s filtering pipeline."""
    df = pd.read_parquet(path)

    keep = (
        (df["steps"].apply(len) > 0)
        & (df["final_answer_correct"] == True)
    )
    return df.loc[keep].copy()


def get_skywork_rewards(df: pd.DataFrame, use_first_step: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of original / augmented Skywork rewards."""
    if use_first_step:
        extract = lambda x: x[0] if len(x) else np.nan
    else:
        extract = np.mean

    orig = df[SKYWORK_ORIG_COL].apply(extract).to_numpy(dtype=float)
    aug  = df[SKYWORK_AUG_COL].apply(extract).to_numpy(dtype=float)
    return orig, aug


def get_qwen_rewards(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return arrays of original / augmented Qwen rewards (min-based, as in notebook)."""
    orig = df[QWEN_ORIG_COL].apply(min).to_numpy(dtype=float)
    aug  = df[QWEN_AUG_COL].apply(min).to_numpy(dtype=float)
    return orig, aug


def add_curve(values: np.ndarray, label: str, color: str) -> None:
    """Plot histogram + KDE on current axes."""
    # histogram
    plt.hist(values, bins=50, density=True, alpha=0.3, edgecolor=color, color=color)
    # KDE
    kde = stats.gaussian_kde(values)
    xs = np.linspace(values.min(), values.max(), 200)
    plt.plot(xs, kde(xs), lw=2, color=color, label=label)


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet_path).expanduser()
    if not parquet_path.exists():
        raise FileNotFoundError(parquet_path)

    out_path = (
        Path(args.output_path)
        if args.output_path
        else parquet_path.with_name(parquet_path.stem + "_reward_deltas.png")
    )

    df = load_and_filter(parquet_path)
    if df.empty:
        raise RuntimeError("All rows were filtered out – check data/filters.")

    # Compute deltas
    sky_orig_mean, sky_aug_mean = get_skywork_rewards(df, use_first_step=False)
    qwen_orig,      qwen_aug    = get_qwen_rewards(df)

    sky_delta_mean  = sky_aug_mean - sky_orig_mean
    qwen_delta      = qwen_aug    - qwen_orig

    # Optional Skywork curve with first-step reward
    if args.include_first_step:
        sky_orig_first, sky_aug_first = get_skywork_rewards(df, use_first_step=True)
        sky_delta_first = sky_aug_first - sky_orig_first

    # ==== Plot ====
    plt.figure(figsize=(10, 6))
    add_curve(
        sky_delta_mean,
        label=f"Skywork PRM (mean, μ={sky_delta_mean.mean():.3f})",
        color="blue",
    )
    if args.include_first_step:
        add_curve(
            sky_delta_first,
            label=f"Skywork PRM [1st step] (μ={sky_delta_first.mean():.3f})",
            color="red",
        )
    add_curve(
        qwen_delta,
        label=f"Qwen PRM (μ={qwen_delta.mean():.3f})",
        color="green",
    )

    # Cosmetics
    plt.title("Distribution of Changes in Rewards", fontsize=12)
    plt.xlabel("Δ Reward (Augmented – Original)", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.axvline(0, linestyle="--", color="k", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save and show
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()
