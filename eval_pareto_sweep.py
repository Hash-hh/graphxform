"""
pareto_sweep_diagnostic.py

Standalone script that loads a trained lambda-conditioned molecular transformer,
runs a Pareto sweep across a grid of lambda values for each validation scaffold,
and produces a CSV of raw molecule-level results plus diagnostic plots.

The script is task-general: it reads `config.objective_type` and picks the
appropriate metric names, sign conventions, and default lambda grid.
Supported objectives out of the box:
    * polypharmacy_2d : λ0·GSK3B + λ1·(1-JNK3)·gate   (max gsk3b, min jnk3)
    * safety_2d       : λ0·JNK3  + λ1·(1-hERG)·gate   (max jnk3,  min herg)
    * tpp_3d          : λ0·GSK3B + λ1·BBB + λ2·(1-hERG), gated by GSK3B
                        (max gsk3b, max bbb, min herg)

Adding a new task = add one entry to TASK_SPECS below.

Early-crash: after the very first (scaffold, lambda) pair is evaluated we
verify that every expected aux_metric key is actually populated on the
returned molecules. If not, we raise immediately instead of wasting an hour
only to discover NaNs at the end.

Usage:
    python eval_pareto_sweep.py \
        --checkpoint results/<run>/best_modelXX.pt \
        --scaffold_path scaffold_splitting/zinc_splits_optimized/run_seed_42/val_scaffolds.txt \
        --output_dir ./pareto_sweep_results \
        --num_samples_per_scaffold 16 \
        --max_scaffolds 50
"""

import argparse
import copy
import os
import gc
import itertools
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer
from molecule_evaluator import MoleculeObjectiveEvaluator


# ---------------------------------------------------------------------------
# Task specifications
# ---------------------------------------------------------------------------
@dataclass
class MetricSpec:
    key: str          # key in mol.aux_metrics
    direction: str    # "max" or "min" — direction the agent should push it
    display: str      # pretty label for plots


@dataclass
class TaskSpec:
    name: str
    metrics: List[MetricSpec]                 # ordered, length == num_objectives
    lambda_grid: List[np.ndarray] = field(default_factory=list)

    @property
    def num_objectives(self) -> int:
        return len(self.metrics)


def _simplex_grid_3d() -> List[np.ndarray]:
    """Corners + pairwise midpoints + centroid of the 3-simplex."""
    pts = []
    for i in range(3):
        v = np.zeros(3); v[i] = 1.0
        pts.append(v)
    for i, j in itertools.combinations(range(3), 2):
        v = np.zeros(3); v[i] = 0.5; v[j] = 0.5
        pts.append(v)
    pts.append(np.full(3, 1.0 / 3.0))
    return pts


LAMBDA_GRID_2D = [
    np.array([1.00, 0.00]),
    np.array([0.75, 0.25]),
    np.array([0.50, 0.50]),
    np.array([0.25, 0.75]),
    np.array([0.00, 1.00]),
]

TASK_SPECS = {
    "polypharmacy_2d": TaskSpec(
        name="polypharmacy_2d",
        metrics=[
            MetricSpec("gsk3b", "max", "GSK3B activity"),
            MetricSpec("jnk3",  "min", "JNK3 activity"),
        ],
        lambda_grid=LAMBDA_GRID_2D,
    ),
    "safety_2d": TaskSpec(
        name="safety_2d",
        metrics=[
            MetricSpec("jnk3", "max", "JNK3 activity"),
            MetricSpec("herg", "min", "hERG liability"),
        ],
        lambda_grid=LAMBDA_GRID_2D,
    ),
    "tpp_3d": TaskSpec(
        name="tpp_3d",
        metrics=[
            MetricSpec("gsk3b", "max", "GSK3B activity"),
            MetricSpec("bbb",   "max", "BBB penetration"),
            MetricSpec("herg",  "min", "hERG liability"),
        ],
        lambda_grid=_simplex_grid_3d(),
    ),
}


def get_task_spec(objective_type: str, num_objectives: int) -> TaskSpec:
    if objective_type in TASK_SPECS:
        spec = TASK_SPECS[objective_type]
        if spec.num_objectives != num_objectives:
            raise ValueError(
                f"[TaskSpec] objective_type={objective_type!r} expects "
                f"{spec.num_objectives} objectives but config has {num_objectives}."
            )
        return spec
    raise ValueError(
        f"[TaskSpec] Unsupported objective_type={objective_type!r}. "
        f"Add an entry to TASK_SPECS in eval_pareto_sweep.py. "
        f"Supported: {list(TASK_SPECS)}"
    )


def lambda_label(lam: np.ndarray) -> str:
    return "(" + ", ".join(f"{v:.2f}" for v in lam) + ")"


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def build_eval_config(base_config: MoleculeConfig, num_samples: int) -> MoleculeConfig:
    cfg = copy.deepcopy(base_config)
    cfg.num_epochs = 0
    cfg.gumbeldore_config = copy.deepcopy(cfg.gumbeldore_config)
    cfg.gumbeldore_config["destination_path"] = None
    cfg.gumbeldore_config["search_type"] = "iid_mc"
    cfg.gumbeldore_config["num_samples_per_instance"] = num_samples
    cfg.gumbeldore_config["sampling_temperature"] = 1.0
    cfg.use_wandb = False
    return cfg


def load_checkpoint_into_network(checkpoint_path: str, config: MoleculeConfig):
    network = MoleculeTransformer(config, config.training_device)
    ckpt = torch.load(checkpoint_path, map_location=config.training_device,
                      weights_only=False)
    network.load_state_dict(ckpt["model_weights"])
    network.to(network.device)
    network.eval()
    epochs_trained = ckpt.get("epochs_trained", "unknown")
    best_val = ckpt.get("best_validation_metric", "unknown")
    print(f"[Load] Loaded checkpoint. Epochs trained: {epochs_trained}, "
          f"best validation metric at save time: {best_val}")
    return network


def assert_aux_metrics_match(group, task: TaskSpec):
    """
    Early-crash sanity check. Verifies every expected metric key is present
    and finite on the first feasible molecule. Prevents the 'ran for an hour
    and all values are NaN' trap.
    """
    expected_keys = [m.key for m in task.metrics]
    for mol in group:
        if mol.objective is None or not np.isfinite(mol.objective):
            continue
        aux = getattr(mol, "aux_metrics", None)
        if not aux:
            raise RuntimeError(
                f"[Sanity] Generated molecule has no aux_metrics. "
                f"Expected task {task.name!r} to populate keys {expected_keys}. "
                f"Check MoleculeObjectiveEvaluator for objective_type={task.name!r}."
            )
        missing = [k for k in expected_keys if k not in aux]
        if missing:
            raise RuntimeError(
                f"[Sanity] aux_metrics is missing expected key(s) {missing} "
                f"for task {task.name!r}. Got keys: {list(aux.keys())}. "
                f"The script's TASK_SPECS entry and the checkpoint's "
                f"objective_type probably disagree."
            )
        bad = [k for k in expected_keys
               if aux.get(k) is None or not np.isfinite(aux[k])]
        if bad:
            raise RuntimeError(
                f"[Sanity] aux_metrics key(s) {bad} are NaN/None in the "
                f"first feasible molecule. Aborting early."
            )
        print(f"[Sanity] aux_metrics OK. Keys = {list(aux.keys())}")
        return
    print("[Sanity] Warning: first scaffold/lambda produced zero feasible "
          "molecules; deferring metric check.")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def run_sweep(
    checkpoint_path: str,
    scaffold_path: str,
    output_dir: str,
    num_samples_per_scaffold: int,
    max_scaffolds: Optional[int],
    lambda_grid: Optional[List[np.ndarray]],
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = MoleculeConfig()
    print(f"[Setup] Objective type: {config.objective_type}")
    print(f"[Setup] Num objectives: {config.num_objectives}")

    task = get_task_spec(config.objective_type, config.num_objectives)
    if lambda_grid is None:
        lambda_grid = task.lambda_grid
    metric_keys = [m.key for m in task.metrics]
    print(f"[Setup] Task metrics: "
          f"{[f'{m.key}({m.direction})' for m in task.metrics]}")
    print(f"[Setup] Lambda grid ({len(lambda_grid)} points):")
    for lam in lambda_grid:
        print(f"          {lambda_label(lam)}")

    eval_config = build_eval_config(config, num_samples=num_samples_per_scaffold)

    import ray
    if not getattr(eval_config, "disable_ray", False):
        num_gpus = len(eval_config.CUDA_VISIBLE_DEVICES.split(","))
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_gpus=num_gpus, logging_level="info",
                 ignore_reinit_error=True)

    np.random.seed(eval_config.seed)
    torch.manual_seed(eval_config.seed)

    with open(scaffold_path, "r") as f:
        scaffolds = [line.strip() for line in f if line.strip()]
    if max_scaffolds is not None and max_scaffolds < len(scaffolds):
        scaffolds = scaffolds[:max_scaffolds]
    print(f"[Setup] Evaluating on {len(scaffolds)} scaffolds "
          f"with {num_samples_per_scaffold} samples per (scaffold, lambda).")
    print(f"[Setup] Total molecules to generate: "
          f"{len(scaffolds) * num_samples_per_scaffold * len(lambda_grid)}")

    network = load_checkpoint_into_network(checkpoint_path, eval_config)
    objective_eval = MoleculeObjectiveEvaluator(
        eval_config, device=eval_config.objective_gnn_device
    )
    dataset = GumbeldoreDataset(config=eval_config,
                                objective_evaluator=objective_eval)
    weights = copy.deepcopy(network.get_weights())

    rows = []
    sanity_checked = False

    for lam_idx, lam in enumerate(lambda_grid):
        lam_lab = lambda_label(lam)
        print(f"\n[Sweep] === Lambda {lam_idx + 1}/{len(lambda_grid)}: "
              f"{lam_lab} ===")

        for scaf_idx, scaffold_smi in tqdm(
            enumerate(scaffolds), total=len(scaffolds),
            desc=f"Lambda {lam_lab}"
        ):
            try:
                grouped = dataset.generate_dataset(
                    network_weights=weights,
                    memory_aggressive=False,
                    prompts=[scaffold_smi],
                    return_raw_trajectories=True,
                    mode="eval",
                    fixed_lambda=lam,
                )
            except Exception as e:
                print(f"[Sweep] Generation failed for scaffold {scaf_idx} "
                      f"at lambda {lam_lab}: {e}")
                continue

            if not grouped or not grouped[0]:
                continue

            group = grouped[0]

            # Early crash: verify aux_metrics layout on the very first group.
            if not sanity_checked:
                assert_aux_metrics_match(group, task)
                sanity_checked = True

            for mol in group:
                if mol.objective is None or not np.isfinite(mol.objective):
                    continue

                aux = getattr(mol, "aux_metrics", {}) or {}
                row = {
                    "scaffold_idx": scaf_idx,
                    "prompt_smiles": scaffold_smi,
                    "lambda_label": lam_lab,
                    "generated_smiles": mol.smiles_string or "",
                    "weighted_reward": float(mol.objective),
                    "num_atoms": int(mol.atoms.shape[0]) - 1
                    if hasattr(mol, "atoms") else -1,
                }
                for dim_i in range(len(lam)):
                    row[f"lambda_{dim_i}"] = float(lam[dim_i])
                for mk in metric_keys:
                    v = aux.get(mk, np.nan)
                    try:
                        v = float(v)
                    except (TypeError, ValueError):
                        v = np.nan
                    row[f"{mk}_raw"] = v
                rows.append(row)

            del grouped
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ------------------------------------------------------------ save CSV
    df = pd.DataFrame(rows)
    csv_path = output_dir / "pareto_sweep_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[Save] Wrote {len(df)} rows to {csv_path}")

    if len(df) == 0:
        print("[Error] No valid molecules were generated. "
              "Check the model / config.")
        if ray.is_initialized():
            ray.shutdown()
        return

    # Summary
    agg_spec = {"n_mols": ("weighted_reward", "size")}
    for m in task.metrics:
        agg_spec[f"{m.key}_mean"] = (f"{m.key}_raw", "mean")
        agg_spec[f"{m.key}_std"]  = (f"{m.key}_raw", "std")
    agg_spec["reward_mean"] = ("weighted_reward", "mean")
    agg_spec["reward_best"] = ("weighted_reward", "max")
    summary = df.groupby("lambda_label", sort=False).agg(**agg_spec).reset_index()
    summary_path = output_dir / "pareto_sweep_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 70)
    print("SUMMARY PER LAMBDA")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)

    print_conditioning_diagnostic(summary, task)

    make_plots(df, summary, output_dir, task)
    print(f"\n[Save] Plots written to {output_dir}")

    if ray.is_initialized():
        ray.shutdown()


def print_conditioning_diagnostic(summary: pd.DataFrame, task: TaskSpec):
    """
    For each metric, compare its mean at the lambda that most-favors it vs the
    lambda that least-favors it. If conditioning is biting, the signed delta
    (in the 'wanted' direction) should be clearly positive.
    """
    print("\nDIAGNOSTIC — is lambda-conditioning actually biting?")
    lam_vecs = np.array([
        [float(x) for x in row.strip("()").split(",")]
        for row in summary["lambda_label"]
    ])
    any_strong = False
    for dim_i, m in enumerate(task.metrics):
        col = f"{m.key}_mean"
        i_hi = int(np.argmax(lam_vecs[:, dim_i]))
        i_lo = int(np.argmin(lam_vecs[:, dim_i]))
        v_hi = summary.iloc[i_hi][col]
        v_lo = summary.iloc[i_lo][col]
        if m.direction == "max":
            delta = v_hi - v_lo
        else:
            delta = v_lo - v_hi  # 'min' metric → should DROP as its λ grows
        print(f"  {m.key:>6s} ({m.direction}): "
              f"λ_hi={summary.iloc[i_hi]['lambda_label']} mean={v_hi:.3f} | "
              f"λ_lo={summary.iloc[i_lo]['lambda_label']} mean={v_lo:.3f} | "
              f"desired-direction delta={delta:+.3f}")
        if delta > 0.05:
            any_strong = True
    if any_strong:
        print("  ✅ At least one dimension shows a strong lambda response.")
    else:
        print("  ❌ No clear lambda response on any dimension — "
              "conditioning may not be biting.")


# ===========================================================================
#                                   PLOTS
# ===========================================================================
def _to_higher_is_better(values: np.ndarray, direction: str) -> np.ndarray:
    return values if direction == "max" else 1.0 - values


def _display_axis_label(m: MetricSpec) -> str:
    if m.direction == "max":
        return f"{m.display}  (raw {m.key}; higher = better)"
    return f"1 − {m.key}  ({m.display}; higher = better)"


def make_plots(df: pd.DataFrame, summary: pd.DataFrame, output_dir: Path,
               task: TaskSpec):
    unique_labels = summary["lambda_label"].tolist()
    cmap = cm.get_cmap("coolwarm")
    def color_for(i):
        return cmap(i / max(len(unique_labels) - 1, 1))

    x = np.arange(len(summary))

    # --- 1. Mean raw biology vs lambda --------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    palette = ["tab:blue", "tab:red", "tab:green", "tab:purple", "tab:orange"]
    for i, m in enumerate(task.metrics):
        arrow = "↑" if m.direction == "max" else "↓"
        ax.errorbar(x, summary[f"{m.key}_mean"], yerr=summary[f"{m.key}_std"],
                    marker="o", capsize=4,
                    label=f"{m.key} {arrow} (want {m.direction.upper()} "
                          f"when λ_{i} high)",
                    color=palette[i % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(summary["lambda_label"], rotation=20, ha="right")
    ax.set_xlabel("lambda")
    ax.set_ylabel("Mean raw oracle score")
    ax.set_title(f"Mean raw biology vs lambda — task: {task.name}\n"
                 "If conditioning works, each curve should move as its "
                 "arrow indicates.")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_biology_vs_lambda.png", dpi=150)
    plt.close(fig)

    # --- 2/3. Biology-space scatter + Pareto front (2D tasks) --------------
    if task.num_objectives == 2:
        mx, my = task.metrics[0], task.metrics[1]

        fig, ax = plt.subplots(figsize=(8, 7))
        for i, lab in enumerate(unique_labels):
            sub = df[df["lambda_label"] == lab]
            ax.scatter(
                _to_higher_is_better(sub[f"{mx.key}_raw"].values, mx.direction),
                _to_higher_is_better(sub[f"{my.key}_raw"].values, my.direction),
                alpha=0.4, s=18, color=color_for(i), label=lab,
            )
        ax.set_xlabel(_display_axis_label(mx))
        ax.set_ylabel(_display_axis_label(my))
        ax.set_title(f"All generated molecules in biology space — "
                     f"task: {task.name}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.axvline(0.5, color="gray", lw=0.5, ls="--")
        ax.legend(title="lambda", loc="lower left", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "pareto_scatter.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 7))
        K = 5
        front_points = []
        for i, lab in enumerate(unique_labels):
            sub = df[df["lambda_label"] == lab].copy()
            best = (sub.sort_values("weighted_reward", ascending=False)
                       .groupby("scaffold_idx").head(K))
            xs = _to_higher_is_better(best[f"{mx.key}_raw"].values, mx.direction)
            ys = _to_higher_is_better(best[f"{my.key}_raw"].values, my.direction)
            ax.scatter(xs, ys, alpha=0.7, s=40, color=color_for(i),
                       label=lab, edgecolors="k", lw=0.3)
            if len(xs):
                front_points.append((float(np.nanmean(xs)),
                                     float(np.nanmean(ys))))
            else:
                front_points.append((np.nan, np.nan))
        front_arr = np.array(front_points)
        ax.plot(front_arr[:, 0], front_arr[:, 1], "k--", alpha=0.5, lw=1.5,
                label="Mean front trajectory")
        for (px, py), lab in zip(front_points, unique_labels):
            if np.isfinite(px) and np.isfinite(py):
                ax.annotate(lab, (px, py), fontsize=8,
                            textcoords="offset points", xytext=(5, 5))
        ax.set_xlabel(_display_axis_label(mx))
        ax.set_ylabel(_display_axis_label(my))
        ax.set_title(f"Pareto front: top-{K} per scaffold-lambda — "
                     f"task: {task.name}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(title="lambda", fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "pareto_front.png", dpi=150)
        plt.close(fig)
    else:
        # 3D+ tasks: per-metric grouped bar chart.
        fig, axes = plt.subplots(
            1, task.num_objectives,
            figsize=(5 * task.num_objectives, 5)
        )
        if task.num_objectives == 1:
            axes = [axes]
        for i, m in enumerate(task.metrics):
            ax = axes[i]
            ax.bar(x, summary[f"{m.key}_mean"],
                   yerr=summary[f"{m.key}_std"], capsize=4,
                   color=[color_for(j) for j in range(len(unique_labels))])
            ax.set_xticks(x)
            ax.set_xticklabels(summary["lambda_label"], rotation=30,
                               ha="right", fontsize=8)
            ax.set_ylabel(f"{m.key} (raw, {m.direction})")
            ax.set_title(m.display)
            ax.grid(alpha=0.3)
        fig.suptitle(f"Per-metric summary — task: {task.name}")
        fig.tight_layout()
        fig.savefig(output_dir / "pareto_scatter.png", dpi=150)
        plt.close(fig)

    # --- 4. Weighted reward vs lambda ---------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, summary["reward_mean"], marker="o", color="tab:green",
            label="Mean weighted reward")
    ax.plot(x, summary["reward_best"], marker="^", color="tab:orange",
            label="Best weighted reward")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["lambda_label"], rotation=20, ha="right")
    ax.set_xlabel("lambda")
    reward_expr = " + ".join(
        f"λ_{i}·{m.key}" if m.direction == "max" else f"λ_{i}·(1-{m.key})"
        for i, m in enumerate(task.metrics)
    )
    ax.set_ylabel(f"Weighted reward ≈ {reward_expr}")
    ax.set_title(f"Achieved weighted reward per lambda — task: {task.name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "reward_vs_lambda.png", dpi=150)
    plt.close(fig)


# ===========================================================================
#                                   ENTRY
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Pareto sweep diagnostic for lambda-conditioned molecular RL"
    )
    parser.add_argument("--checkpoint", type=str,
                        help="Path to .pt checkpoint (e.g. best_model.pt)",
                        default="results/2026-04-23--18-28-35/best_model182.pt")
    parser.add_argument("--scaffold_path", type=str,
                        default="scaffold_splitting/zinc_splits_optimized/"
                                "run_seed_42/val_scaffolds.txt",
                        help="Path to scaffold SMILES file (one per line)")
    parser.add_argument("--output_dir", type=str,
                        default="./pareto_sweep_results",
                        help="Where to write CSV and plots")
    parser.add_argument("--num_samples_per_scaffold", type=int, default=16,
                        help="iid MC samples per (scaffold, lambda). Default 16.")
    parser.add_argument("--max_scaffolds", type=int, default=50,
                        help="Cap on number of scaffolds to evaluate "
                             "(for speed). Default 50. Set to -1 for all.")
    args = parser.parse_args()

    max_scaf = None if args.max_scaffolds < 0 else args.max_scaffolds

    run_sweep(
        checkpoint_path=args.checkpoint,
        scaffold_path=args.scaffold_path,
        output_dir=args.output_dir,
        num_samples_per_scaffold=args.num_samples_per_scaffold,
        max_scaffolds=max_scaf,
        lambda_grid=None,
    )


if __name__ == "__main__":
    main()

