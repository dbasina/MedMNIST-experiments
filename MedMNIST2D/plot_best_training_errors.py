
#!/usr/bin/env python3

import os
import re
import glob
import math
import argparse
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

try:
    # TensorBoard's event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as e:
    raise RuntimeError(
        "Failed to import TensorBoard EventAccumulator. "
        "Install TensorBoard with:\n  pip install tensorboard\n"
        f"Original error: {e}"
    )

AUC_RE = re.compile(r"\[AUC\]([0-9.]+)")

def parse_test_auc_from_filename(path: str) -> Optional[float]:
    """Extract AUC from filename like: ..._test_[AUC]0.987_[ACC]0.886@r50_28_seed0.csv"""
    m = AUC_RE.search(os.path.basename(path))
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def find_completed_seeds(dataset_dir: str, dataset_name: str, seed_glob: str) -> Dict[str, float]:
    """
    Return mapping: seed_dir -> best_test_auc (by scanning *_test_*.csv files in each seed).
    Only includes seeds that have at least one test CSV.
    """
    result = {}
    for seed_dir in sorted(glob.glob(os.path.join(dataset_dir, seed_glob))):
        test_csvs = glob.glob(os.path.join(seed_dir, f"{dataset_name}_test_*.csv"))
        if not test_csvs:
            continue
        best_auc = None
        for p in test_csvs:
            auc = parse_test_auc_from_filename(p)
            if auc is None:
                continue
            if (best_auc is None) or (auc > best_auc):
                best_auc = auc
        if best_auc is not None:
            result[seed_dir] = best_auc
    return result

def pick_best_seed(seed_to_auc: Dict[str, float]) -> Optional[Tuple[str, float]]:
    """Return (best_seed_dir, best_auc) or None if empty."""
    if not seed_to_auc:
        return None
    best_seed = max(seed_to_auc.items(), key=lambda kv: kv[1])
    return best_seed[0], best_seed[1]

def load_train_loss_from_tb(tb_dir: str) -> Optional[Tuple[List[int], List[float]]]:
    """
    Load per-epoch `train_loss` scalar from a TensorBoard directory.
    Returns (steps, values) or None if not found.
    """
    if not os.path.isdir(tb_dir):
        return None
    ea = EventAccumulator(tb_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if "train_loss" not in tags:
        # Fallback: some users might log as 'train/loss'
        fallback_tags = [t for t in tags if t.replace("/", "_") == "train_loss"]
        tag = fallback_tags[0] if fallback_tags else None
    else:
        tag = "train_loss"
    if not tag:
        return None

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

def main(root: str, seed_glob: str, out_path: str, max_cols: int):
    # Datasets are immediate subfolders of root
    datasets = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
    picked = []  # list of (dataset_name, best_seed_dir, best_auc)

    for ds in datasets:
        ds_dir = os.path.join(root, ds)
        seed_to_auc = find_completed_seeds(ds_dir, ds, seed_glob)
        best = pick_best_seed(seed_to_auc)
        if best is None:
            # No completed seed (no test CSV found)
            continue
        best_seed_dir, best_auc = best
        picked.append((ds, best_seed_dir, best_auc))

    if not picked:
        raise SystemExit("No completed datasets found (no *_test_*.csv files).")

    # Prepare the subplot grid
    n = len(picked)
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows), squeeze=False)

    for idx, (ds, seed_dir, best_auc) in enumerate(picked):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        tb_dir = os.path.join(seed_dir, "Tensorboard_Results")
        series = load_train_loss_from_tb(tb_dir)
        if series is None:
            ax.text(0.5, 0.5, "No train_loss logged", ha="center", va="center")
            ax.set_title(f"{ds}  |  {os.path.basename(seed_dir)}  |  best test AUC={best_auc:.3f}")
            ax.set_xlabel("epoch")
            ax.set_ylabel("train loss")
            continue

        steps, values = series
        if steps and values:
            ax.plot(steps, values, linewidth=1.5)
        ax.set_title(f"{ds}  |  {os.path.basename(seed_dir)}  |  best test AUC={best_auc:.3f}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("train loss")
        ax.grid(True, linestyle="--", alpha=0.3)

    # Hide any unused axes
    total_axes = rows * cols
    for k in range(n, total_axes):
        r, c = divmod(k, cols)
        axes[r][c].axis("off")

    plt.tight_layout()
    fig.suptitle("Training Error (train_loss) — Best Seed per Dataset", y=1.02, fontsize=14)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"Saved figure: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training error for best seed of each dataset")
    parser.add_argument("--root", type=str, default="./output",
                        help="Path to the output root containing dataset subfolders")
    parser.add_argument("--seed_glob", type=str, default="r50_28_seed*",
                        help="Glob to match seed subdirectories")
    parser.add_argument("--out", type=str, default="training_errors_best_seeds.png",
                        help="Output image filename")
    parser.add_argument("--max_cols", type=int, default=3,
                        help="Maximum subplot columns in the figure grid")
    args = parser.parse_args()
    main(args.root, args.seed_glob, args.out, args.max_cols)
