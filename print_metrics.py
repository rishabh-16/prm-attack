#!/usr/bin/env python3
"""
Print metrics from a cached experiment's metrics.pkl or result.pkl file.

Usage examples:
    # Print metrics for Skywork 7B batched experiment
    python print_metrics.py \
        --prm_model 7B \
        --experiment batched \
        --num_adv_tokens 100 \
        --adv_position end \
        --num_train_trajectories 8

    # Print metrics for Qwen-7B experiment
    python print_metrics.py \
        --prm_model Qwen-7B \
        --experiment batched \
        --num_adv_tokens 100 \
        --adv_position middle \
        --num_train_trajectories 8

    # Use result.pkl instead of metrics.pkl
    python print_metrics.py \
        --prm_model 7B \
        --experiment batched \
        --num_adv_tokens 100 \
        --adv_position end \
        --num_train_trajectories 8 \
        --use_result
"""

import argparse
import os
import pickle
from typing import Optional, Dict, Any
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print metrics from cached experiment results."
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
    
    # File type
    parser.add_argument(
        "--use_result", action="store_true", default=False,
        help="Use result.pkl instead of metrics.pkl"
    )
    
    # Output options
    parser.add_argument(
        "--show_history", action="store_true", default=False,
        help="Print full history arrays (can be very long)"
    )
    parser.add_argument(
        "--history_limit", type=int, default=10,
        help="Number of history entries to show at start/end (default: 10)"
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
    """Generate the experiment prefix used for cache file naming."""
    mode = "continuous" if continuous else "discrete"
    n_traj = num_train_trajectories or "all"
    prm_type = get_prm_type(prm_model)
    prm_prefix = "qwen_" if prm_type == "qwen" else ""
    return f"{prm_prefix}{experiment}_{prm_model}_{mode}_{num_adv_tokens}tok_{adv_position}_{n_traj}traj"


def format_value(value: Any, show_full: bool = False, limit: int = 10) -> str:
    """Format a value for printing."""
    if value is None:
        return "None"
    
    if isinstance(value, (list, np.ndarray)):
        arr = np.array(value) if isinstance(value, list) else value
        
        if arr.size == 0:
            return "[] (empty)"
        
        # For nested arrays (like per_traj_reward_history)
        if arr.ndim > 1 or (arr.ndim == 1 and isinstance(arr[0], (list, np.ndarray))):
            return f"Array shape: {np.shape(arr)}"
        
        if show_full:
            return str(arr.tolist())
        
        n = len(arr)
        if n <= 2 * limit:
            return str(arr.tolist())
        
        start = arr[:limit].tolist()
        end = arr[-limit:].tolist()
        return f"[{', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in start)}, ... ({n - 2*limit} more) ..., {', '.join(f'{x:.4f}' if isinstance(x, float) else str(x) for x in end)}] (len={n})"
    
    if isinstance(value, float):
        return f"{value:.6f}"
    
    if isinstance(value, dict):
        return f"Dict with {len(value)} keys: {list(value.keys())}"
    
    # Check for torch tensors
    if hasattr(value, 'shape'):
        return f"Tensor shape: {value.shape}"
    
    return str(value)


def print_metrics(data: Dict, show_history: bool = False, history_limit: int = 10) -> None:
    """Print metrics in a formatted way."""
    print("\n" + "=" * 70)
    print("METRICS CONTENTS")
    print("=" * 70)
    
    # Categorize keys
    history_keys = []
    scalar_keys = []
    other_keys = []
    
    for key in data.keys():
        if 'history' in key.lower():
            history_keys.append(key)
        elif isinstance(data[key], (int, float, bool, str, type(None))):
            scalar_keys.append(key)
        else:
            other_keys.append(key)
    
    # Print scalar values first
    if scalar_keys:
        print("\n--- Scalar Values ---")
        for key in sorted(scalar_keys):
            print(f"  {key}: {format_value(data[key])}")
    
    # Print history summaries
    if history_keys:
        print("\n--- Training Histories ---")
        for key in sorted(history_keys):
            value = data[key]
            if value is None:
                print(f"  {key}: None")
                continue
            
            arr = np.array(value) if isinstance(value, list) else value
            
            # Handle nested arrays (per_traj_reward_history)
            if arr.ndim > 1 or (arr.ndim == 1 and len(arr) > 0 and isinstance(arr[0], (list, np.ndarray))):
                print(f"  {key}:")
                print(f"    Shape: {np.shape(arr)}")
                continue
            
            if len(arr) == 0:
                print(f"  {key}: [] (empty)")
                continue
            
            print(f"  {key}:")
            print(f"    Length: {len(arr)}")
            print(f"    Start:  {arr[0]:.6f}")
            print(f"    End:    {arr[-1]:.6f}")
            print(f"    Min:    {np.min(arr):.6f}")
            print(f"    Max:    {np.max(arr):.6f}")
            print(f"    Mean:   {np.mean(arr):.6f}")
            
            if show_history:
                print(f"    Values: {format_value(arr, show_full=True)}")
            else:
                print(f"    Preview: {format_value(arr, show_full=False, limit=history_limit)}")
    
    # Print other values
    if other_keys:
        print("\n--- Other Data ---")
        for key in sorted(other_keys):
            print(f"  {key}: {format_value(data[key])}")
    
    print("\n" + "=" * 70)


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
    
    # Determine file type
    suffix = "_result.pkl" if args.use_result else "_metrics.pkl"
    file_path = os.path.join(args.cache_dir, f"{prefix}{suffix}")
    
    if not os.path.exists(file_path):
        print(f"\nError: File not found: {file_path}")
        
        # Try the other file type
        alt_suffix = "_metrics.pkl" if args.use_result else "_result.pkl"
        alt_path = os.path.join(args.cache_dir, f"{prefix}{alt_suffix}")
        if os.path.exists(alt_path):
            print(f"  But found: {alt_path}")
            print(f"  Try {'removing' if args.use_result else 'adding'} --use_result flag")
        
        print("\nAvailable files in cache directory:")
        if os.path.exists(args.cache_dir):
            for f in sorted(os.listdir(args.cache_dir)):
                if f.endswith(".pkl"):
                    print(f"  {f}")
        return
    
    print(f"Loading: {file_path}")
    
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    
    print(f"File size: {os.path.getsize(file_path) / 1024 / 1024:.2f} MB")
    print(f"Number of keys: {len(data)}")
    print(f"Keys: {list(data.keys())}")
    
    print_metrics(data, show_history=args.show_history, history_limit=args.history_limit)


if __name__ == "__main__":
    main()
