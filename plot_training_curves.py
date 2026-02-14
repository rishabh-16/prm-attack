#!/usr/bin/env python3
"""
Standalone script to load and plot training curves from cached experiment results.

Usage examples:
    # Plot Skywork 7B batched discrete experiment with 500 tokens at end position
    python plot_training_curves.py \
        --prm_model 7B \
        --experiment batched \
        --num_adv_tokens 500 \
        --adv_position end \
        --num_train_trajectories 8

    # Plot Qwen-7B batched discrete experiment with 100 tokens at middle position
    python plot_training_curves.py \
        --prm_model Qwen-7B \
        --experiment batched \
        --num_adv_tokens 100 \
        --adv_position middle \
        --num_train_trajectories 8

    # Plot continuous optimization experiment
    python plot_training_curves.py \
        --prm_model 1.5B \
        --experiment batched \
        --continuous \
        --num_adv_tokens 1 \
        --num_train_trajectories 8
"""

import argparse
import os
import pickle
from typing import Optional, Dict, List
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Publication-quality plot settings
def setup_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    mpl.rcParams.update({
        # Font settings - use serif fonts like in papers
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times', 'serif'],
        'mathtext.fontset': 'stix',
        
        # Font sizes
        'font.size': 14,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        
        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,
        
        # Axes settings
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'grid.linewidth': 0.8,
        
        # Figure settings
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        'legend.fancybox': True,
    })


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and plot training curves from cached experiment results."
    )
    
    # Model
    parser.add_argument(
        "--prm_model", type=str, required=True,
        choices=["1.5B", "7B", "Qwen-7B"],
        help="PRM model: 1.5B, 7B (Skywork) or Qwen-7B"
    )
    
    # Experiment type
    parser.add_argument(
        "--experiment", type=str, required=True,
        choices=["single", "batched"],
        help="Experiment type: single trajectory or batched"
    )
    
    # Optimization mode
    parser.add_argument(
        "--continuous", action="store_true", default=False,
        help="Continuous optimization mode (default: discrete)"
    )
    
    # Adversarial tokens
    parser.add_argument(
        "--num_adv_tokens", type=int, default=1,
        help="Number of adversarial tokens"
    )
    parser.add_argument(
        "--adv_position", type=str, default="end",
        choices=["end", "middle"],
        help="Position of adversarial tokens"
    )
    
    # Dataset
    parser.add_argument(
        "--num_train_trajectories", type=int, default=None,
        help="Number of training trajectories (None = all)"
    )
    
    # Cache directory
    parser.add_argument(
        "--cache_dir", type=str, default="./experiment_cache",
        help="Directory containing cached results"
    )
    
    # Output
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Output path for the plot (default: auto-generated)"
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Show the plot interactively"
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="DPI for saved plot (300 recommended for publication)"
    )
    
    # Plot customization
    parser.add_argument(
        "--figsize_width", type=float, default=12,
        help="Figure width in inches"
    )
    parser.add_argument(
        "--figsize_height", type=float, default=4.5,
        help="Figure height in inches"
    )
    parser.add_argument(
        "--font_size", type=int, default=14,
        help="Base font size for labels"
    )
    parser.add_argument(
        "--title_font_size", type=int, default=16,
        help="Font size for titles"
    )
    parser.add_argument(
        "--no_titles", action="store_true", default=False,
        help="Remove plot titles"
    )
    
    return parser.parse_args()


def get_prm_type(prm_key: str) -> str:
    """Return the PRM type: 'skywork' or 'qwen'."""
    if prm_key in ["1.5B", "7B"]:
        return "skywork"
    elif prm_key == "Qwen-7B":
        return "qwen"
    else:
        raise ValueError(f"Unknown PRM key: {prm_key}")


def get_experiment_prefix(
    prm_model: str,
    experiment: str,
    continuous: bool,
    num_adv_tokens: int,
    adv_position: str,
    num_train_trajectories: Optional[int],
) -> str:
    """
    Generate the experiment prefix used for cache file naming.
    
    This matches the naming convention in prm_attack_experiments.py:
    {prm_prefix}{experiment}_{prm_model}_{mode}_{num_tokens}tok_{position}_{n_traj}traj
    """
    mode = "continuous" if continuous else "discrete"
    n_traj = num_train_trajectories or "all"
    prm_type = get_prm_type(prm_model)
    prm_prefix = "qwen_" if prm_type == "qwen" else ""
    return f"{prm_prefix}{experiment}_{prm_model}_{mode}_{num_adv_tokens}tok_{adv_position}_{n_traj}traj"


def load_result(cache_dir: str, prefix: str) -> Optional[Dict]:
    """Load the cached result file."""
    result_path = os.path.join(cache_dir, f"{prefix}_result.pkl")
    
    if not os.path.exists(result_path):
        print(f"Error: Result file not found: {result_path}")
        return None
    
    print(f"Loading result from: {result_path}")
    with open(result_path, "rb") as f:
        result = pickle.load(f)
    
    return result


def plot_training_curves(
    result: Dict,
    experiment: str,
    output_path: str,
    figsize: tuple = (12, 4.5),
    font_size: int = 14,
    title_font_size: int = 16,
    show_titles: bool = True,
    dpi: int = 300,
    show: bool = False,
) -> None:
    """
    Plot training curves from the result dictionary.
    
    Creates publication-quality plots with:
    - Reward vs Iteration (soft and discrete)
    - Entropy vs Iteration
    """
    # Setup publication style
    setup_publication_style()
    
    # Define a nice color palette
    colors = {
        'soft': '#2E86AB',      # Steel blue
        'discrete': '#E94F37',   # Vermillion red
        'entropy': '#1B998B',    # Teal green
    }
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # ---- Plot 1: Soft vs Discrete Reward ----
    ax = axes[0]
    if experiment == "single":
        soft_rewards = result.get("reward_history", [])
        discrete_rewards = result.get("discrete_reward_history", [])
        ylabel = "Reward"
    else:  # batched
        soft_rewards = result.get("avg_reward_history", [])
        discrete_rewards = result.get("avg_discrete_reward_history", [])
        ylabel = "Average Reward"
    
    iterations = np.arange(len(soft_rewards)) if soft_rewards else []
    
    if soft_rewards:
        ax.plot(iterations, soft_rewards, label="Soft (Gumbel-Softmax)", 
                color=colors['soft'], linewidth=2.5, alpha=0.9)
    if discrete_rewards:
        ax.plot(iterations, discrete_rewards, label="Discrete (Argmax)", 
                color=colors['discrete'], linewidth=2.5, alpha=0.9, linestyle='--')
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if show_titles:
        ax.set_title("Training Reward", fontweight='semibold', pad=10)
    
    # Style the legend
    legend = ax.legend(loc='lower right', framealpha=0.95)
    legend.get_frame().set_linewidth(0.8)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Format x-axis with thousands separator if needed
    if len(iterations) > 1000:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # ---- Plot 2: Entropy ----
    ax = axes[1]
    entropy_history = result.get("entropy_history", [])
    iterations_entropy = np.arange(len(entropy_history)) if entropy_history else []
    
    if entropy_history:
        ax.plot(iterations_entropy, entropy_history, 
                color=colors['entropy'], linewidth=2.5, alpha=0.9)
        
        # Add a subtle fill under the curve
        ax.fill_between(iterations_entropy, entropy_history, 
                        alpha=0.15, color=colors['entropy'])
    
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Entropy")
    if show_titles:
        ax.set_title("Token Distribution Entropy", fontweight='semibold', pad=10)
    
    # Clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    
    # Format x-axis with thousands separator if needed
    if len(iterations_entropy) > 1000:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    
    # Adjust layout
    plt.tight_layout(pad=1.5)
    
    # Save with high quality
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved plot to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def print_result_summary(result: Dict, experiment: str) -> None:
    """Print a summary of the experiment result."""
    print("\n" + "=" * 60)
    print("EXPERIMENT RESULT SUMMARY")
    print("=" * 60)
    
    if experiment == "single":
        print(f"  Initial reward: {result.get('initial_avg_reward', 'N/A')}")
        print(f"  Best soft reward: {result.get('best_reward', 'N/A')}")
        print(f"  Best discrete reward: {result.get('best_discrete_reward', 'N/A')}")
        
        reward_history = result.get("reward_history", [])
        discrete_history = result.get("discrete_reward_history", [])
    else:  # batched
        print(f"  Initial avg reward: {result.get('initial_avg_reward', 'N/A')}")
        print(f"  Best avg soft reward: {result.get('best_avg_reward', 'N/A')}")
        print(f"  Best avg discrete reward: {result.get('best_avg_discrete_reward', 'N/A')}")
        
        reward_history = result.get("avg_reward_history", [])
        discrete_history = result.get("avg_discrete_reward_history", [])
    
    if reward_history:
        print(f"\n  Soft reward history:")
        print(f"    Start: {reward_history[0]:.4f}")
        print(f"    End: {reward_history[-1]:.4f}")
        print(f"    Max: {max(reward_history):.4f}")
        print(f"    Min: {min(reward_history):.4f}")
    
    if discrete_history:
        print(f"\n  Discrete reward history:")
        print(f"    Start: {discrete_history[0]:.4f}")
        print(f"    End: {discrete_history[-1]:.4f}")
        print(f"    Max: {max(discrete_history):.4f}")
        print(f"    Min: {min(discrete_history):.4f}")
    
    # Check for Qwen-specific min step reward
    min_step_history = result.get("min_step_reward_history") or result.get("avg_min_step_reward_history")
    if min_step_history:
        print(f"\n  Min step reward history (Qwen):")
        print(f"    Start: {min_step_history[0]:.4f}")
        print(f"    End: {min_step_history[-1]:.4f}")
        print(f"    Max: {max(min_step_history):.4f}")
        print(f"    Min: {min(min_step_history):.4f}")
    
    # Print number of iterations
    num_iterations = len(reward_history) if reward_history else 0
    print(f"\n  Number of iterations: {num_iterations}")
    
    # Print best token info if available
    if result.get("best_discrete_token_ids") is not None:
        token_ids = result["best_discrete_token_ids"]
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        print(f"  Number of adversarial tokens: {len(token_ids) if isinstance(token_ids, list) else 1}")
    
    print("=" * 60 + "\n")


def main():
    args = parse_args()
    
    # Generate experiment prefix
    prefix = get_experiment_prefix(
        prm_model=args.prm_model,
        experiment=args.experiment,
        continuous=args.continuous,
        num_adv_tokens=args.num_adv_tokens,
        adv_position=args.adv_position,
        num_train_trajectories=args.num_train_trajectories,
    )
    
    print(f"Experiment prefix: {prefix}")
    
    # Load result
    result = load_result(args.cache_dir, prefix)
    if result is None:
        print("\nAvailable result files in cache directory:")
        if os.path.exists(args.cache_dir):
            for f in sorted(os.listdir(args.cache_dir)):
                if f.endswith("_result.pkl"):
                    print(f"  {f}")
        return
    
    # Print summary
    print_result_summary(result, args.experiment)
    
    # Generate output path if not specified
    if args.output_path is None:
        output_path = os.path.join(args.cache_dir, f"{prefix}_training_curves.png")
    else:
        output_path = args.output_path
    
    # Plot
    plot_training_curves(
        result=result,
        experiment=args.experiment,
        output_path=output_path,
        figsize=(args.figsize_width, args.figsize_height),
        font_size=args.font_size,
        title_font_size=args.title_font_size,
        show_titles=not args.no_titles,
        dpi=args.dpi,
        show=args.show,
    )


if __name__ == "__main__":
    main()
