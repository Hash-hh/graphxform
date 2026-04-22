"""
Pareto-sweep evaluation for conditional GRPO (Phase A).

Sweeps a grid of lambda vectors on the simplex, generates N molecules per
(lambda, scaffold) pair, collects per-objective raw scores from aux_metrics,
and writes a CSV + plots. Reports the primary success criteria from the plan:
monotonicity of mean objective vs lambda component, extreme separation at the
corners, and Pearson correlation.

Usage:
    python eval_pareto_sweep.py --checkpoint path/to/best_model.pt \
        --grid 11 --n-per-lambda 16 --n-scaffolds 50
"""

import argparse
import copy
import csv
import os

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import ray
import torch

from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer
from molecule_evaluator import MoleculeObjectiveEvaluator


def build_lambda_grid(num_obj: int, grid: int):
    """Uniform grid on the simplex. For 2D returns `grid` points; for 3D returns
    a triangular grid with step 1/(grid-1)."""
    if num_obj == 2:
        ts = np.linspace(0.0, 1.0, grid)
        return [np.array([t, 1.0 - t]) for t in ts]
    if num_obj == 3:
        step = 1.0 / (grid - 1)
        pts = []
        for i in range(grid):
            for j in range(grid - i):
                a = i * step
                b = j * step
                c = 1.0 - a - b
                if c < -1e-9:
                    continue
                c = max(c, 0.0)
                pts.append(np.array([a, b, max(c, 0.0)]))
        return pts
    raise NotImplementedError(f"num_obj={num_obj} not supported by this sweep")


def extract_scores(mol, keys):
    aux = getattr(mol, "aux_metrics", None) or {}
    return {k: float(aux.get(k, float("nan"))) for k in keys}


def aux_keys_for_task(objective_type: str):
    if objective_type == "polypharmacy_2d":
        return ["gsk3b", "jnk3", "reward"]
    if objective_type == "safety_2d":
        return ["jnk3", "herg", "reward"]
    if objective_type == "tpp_3d":
        return ["gsk3b", "bbb", "herg", "reward"]
    raise NotImplementedError(f"Unknown objective_type={objective_type}")


def primary_axes_for_task(objective_type: str):
    """Return (x_label, y_label) for the Pareto scatter. y is the 'minimize' axis
    plotted as (1 - raw), consistent with how the reward uses it."""
    if objective_type == "polypharmacy_2d":
        return ("gsk3b", "jnk3")
    if objective_type == "safety_2d":
        return ("jnk3", "herg")
    if objective_type == "tpp_3d":
        return ("gsk3b", "herg")
    raise NotImplementedError


def run_sweep(config: MoleculeConfig, network, objective_eval, scaffolds,
              grid_points, n_per_lambda: int, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    # Force per-(lambda, scaffold) sample count.
    config = copy.deepcopy(config)
    config.gumbeldore_config = copy.deepcopy(config.gumbeldore_config)
    config.gumbeldore_config["num_samples_per_instance"] = n_per_lambda
    config.gumbeldore_config["destination_path"] = None

    dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_eval)

    weights = network.get_weights()
    aux_keys = aux_keys_for_task(config.objective_type)
    csv_path = os.path.join(output_dir, "sweep.csv")

    with open(csv_path, "w", newline="") as f:
        fieldnames = (
            ["lambda_idx", "scaffold_idx", "scaffold"]
            + [f"lambda_{i}" for i in range(config.num_objectives)]
            + ["generated_smiles"]
            + aux_keys
        )
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for li, lam in enumerate(grid_points):
            print(f"[sweep] lambda {li+1}/{len(grid_points)} = {lam}")
            for si, scaffold in enumerate(scaffolds):
                grouped = dataset.generate_dataset(
                    network_weights=weights,
                    memory_aggressive=False,
                    prompts=[scaffold],
                    return_raw_trajectories=True,
                    mode="eval",
                    fixed_lambda=lam.tolist(),
                )
                if not grouped or not grouped[0]:
                    continue
                for mol in grouped[0]:
                    row = {
                        "lambda_idx": li,
                        "scaffold_idx": si,
                        "scaffold": scaffold,
                        "generated_smiles": getattr(mol, "smiles", ""),
                    }
                    for i, v in enumerate(lam):
                        row[f"lambda_{i}"] = float(v)
                    row.update(extract_scores(mol, aux_keys))
                    writer.writerow(row)

    print(f"[sweep] wrote {csv_path}")
    return csv_path


def summarize(csv_path: str, objective_type: str, output_dir: str):
    try:
        import pandas as pd
    except ImportError:
        print("[summarize] pandas not available; skipping summary")
        return

    df = pd.read_csv(csv_path)
    x_key, y_key = primary_axes_for_task(objective_type)

    # Per-lambda means on the two primary axes.
    lam_cols = [c for c in df.columns if c.startswith("lambda_") and c != "lambda_idx"]
    agg = df.groupby("lambda_idx").agg(
        **{f"mean_{x_key}": (x_key, "mean"),
           f"mean_{y_key}": (y_key, "mean"),
           "mean_reward": ("reward", "mean"),
           **{c: (c, "first") for c in lam_cols}}
    ).reset_index()

    summary_path = os.path.join(output_dir, "sweep_summary.csv")
    agg.to_csv(summary_path, index=False)
    print(f"[summarize] wrote {summary_path}")

    if "lambda_0" in agg.columns:
        lam0 = agg["lambda_0"].to_numpy()
        mx = agg[f"mean_{x_key}"].to_numpy()
        # Monotonicity in lambda_0
        order = np.argsort(lam0)
        mx_sorted = mx[order]
        inversions = int(np.sum(np.diff(mx_sorted) < 0))
        max_inv_drop = float(-np.min(np.diff(mx_sorted))) if len(mx_sorted) > 1 else 0.0

        # Corner extremes: (1,0) vs (0,1) on primary x-axis
        def _find_row(target_lam):
            diffs = np.linalg.norm(agg[lam_cols].to_numpy() - np.array(target_lam), axis=1)
            return agg.iloc[int(np.argmin(diffs))]
        corner_x_high = _find_row([1.0, 0.0])
        corner_x_low  = _find_row([0.0, 1.0])
        extreme_sep = float(corner_x_high[f"mean_{x_key}"] - corner_x_low[f"mean_{x_key}"])

        try:
            pearson = float(np.corrcoef(lam0, mx)[0, 1])
        except Exception:
            pearson = float("nan")

        print("[summarize] primary success criteria (polypharmacy_2d):")
        print(f"    monotonicity inversions in mean_{x_key} vs lambda_0: {inversions} (max drop {max_inv_drop:.3f})")
        print(f"    extreme separation mean_{x_key}[lambda=(1,0)] - mean_{x_key}[lambda=(0,1)] = {extreme_sep:.3f} (target >= 0.25)")
        print(f"    Pearson rho(lambda_0, mean_{x_key}) = {pearson:.3f} (target > 0.7)")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(agg["lambda_0"], agg[f"mean_{x_key}"], "o-", label=f"mean {x_key}")
        axes[0].plot(agg["lambda_0"], 1.0 - agg[f"mean_{y_key}"], "s-", label=f"mean (1 - {y_key})")
        axes[0].set_xlabel("lambda_0")
        axes[0].set_ylabel("score")
        axes[0].set_title("Mean objective components vs lambda_0")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].scatter(df[x_key], 1.0 - df[y_key], c=df["lambda_0"], cmap="viridis", s=8, alpha=0.4)
        axes[1].set_xlabel(x_key)
        axes[1].set_ylabel(f"1 - {y_key}")
        axes[1].set_title("Pareto scatter (color = lambda_0)")
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        png_path = os.path.join(output_dir, "sweep.png")
        fig.savefig(png_path, dpi=150)
        print(f"[summarize] wrote {png_path}")
    except ImportError:
        print("[summarize] matplotlib not available; skipping plot")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--grid", type=int, default=11,
                        help="2D: number of sweep points. 3D: points per simplex edge (~grid*(grid+1)/2 total).")
    parser.add_argument("--n-per-lambda", type=int, default=16,
                        help="Molecules generated per (scaffold, lambda).")
    parser.add_argument("--n-scaffolds", type=int, default=50,
                        help="Number of held-out scaffolds to sweep.")
    parser.add_argument("--scaffolds-path", default=None,
                        help="Override scaffolds file (default: config.evaluation_scaffolds_path).")
    args = parser.parse_args()

    config = MoleculeConfig()
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), "pareto_sweep")
    os.makedirs(output_dir, exist_ok=True)

    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    if ray.is_initialized():
        ray.shutdown()
    ray.init(num_gpus=num_gpus, logging_level="info", ignore_reinit_error=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    network = MoleculeTransformer(config, config.training_device)
    objective_eval = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)
    checkpoint = torch.load(args.checkpoint, map_location=config.training_device)
    network.load_state_dict(checkpoint["model_weights"], strict=False)
    network.to(network.device)
    network.eval()

    scaffolds_path = args.scaffolds_path or config.evaluation_scaffolds_path
    with open(scaffolds_path, "r") as f:
        all_scaffolds = [line.strip() for line in f if line.strip()]
    scaffolds = all_scaffolds[: args.n_scaffolds]
    print(f"[sweep] {len(scaffolds)} scaffolds from {scaffolds_path}")

    grid_points = build_lambda_grid(config.num_objectives, args.grid)
    print(f"[sweep] {len(grid_points)} lambda points on the simplex")

    with torch.no_grad():
        csv_path = run_sweep(config, network, objective_eval, scaffolds,
                             grid_points, args.n_per_lambda, output_dir)
    summarize(csv_path, config.objective_type, output_dir)

    ray.shutdown()


if __name__ == "__main__":
    main()
