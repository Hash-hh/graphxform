"""
evaluate_neurips_models.py
==========================
Inference-only evaluation of multiple trained checkpoints living under
`model/neurips/` (or any folder you pass with --models_dir).

Naming convention for checkpoints:
    {algo}_{objective}.pt
where:
    algo       in {ppo, grpo, reinforce, graphxform, tasar}
    objective  in {kinase, jnk3, gsk3}        # 'kinase' --> kinase_mpo

For each checkpoint we:
  * load the model weights into a fresh `MoleculeTransformer`,
  * build a fresh `MoleculeObjectiveEvaluator` for the inferred objective
    (so `config.objective_type` is set correctly and the right oracles
    are constructed),
  * load the test scaffolds (config.evaluation_scaffolds_path or
    --scaffolds_path), and
  * run inference `--num_runs` times (default 3) using stochastic
    sampling, with a different RNG seed per run, to produce mean / std.

We report the model objective (kinase_mpo / jnk3 / gsk3) AND, for the
kinase models, additionally log the individual GSK3B and JNK3
component scores (re-computed via the TDC oracle on the generated
SMILES).

All per-molecule rows are dumped to a per-model CSV
(`{model}_run{R}_detailed.csv`) and a single master CSV
(`neurips_eval_summary.csv`) is written with one row per (model x run)
plus aggregate "mean ± std" rows.

Usage
-----
    python evaluate_neurips_models.py \
        --models_dir model/neurips \
        --num_runs 3 \
        --num_samples 32 \
        --output_dir results/neurips_eval

You can override the test scaffolds path with --scaffolds_path; otherwise
the path stored in the (base) config is used.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime
import gc
import importlib
import os
import platform
import sys
from typing import Callable, List, Optional, Tuple

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import ray
import torch
from tqdm import tqdm

from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer
from molecule_evaluator import (
    MoleculeObjectiveEvaluator,
    OracleTracker,
    LocalOracleTracker,
)

# Lazy-imported (only when needed) to avoid heavy TDC oracle init
from objective_predictor.tdc.kinase_mpo import KinaseMPOObjective


# ---------------------------------------------------------------------------
# Naming convention helpers
# ---------------------------------------------------------------------------
# Filename rule: everything BEFORE the last '_' is the algo / variant tag,
# the substring AFTER the last '_' is the objective. Examples:
#   grxform_denovo_gsk3b.pt    -> algo='grxform_denovo'  obj='gsk3b'
#   original_graphxform_jnk3.pt-> algo='original_graphxform'  obj='jnk3'
#   ppo_gsk3b.pt               -> algo='ppo'  obj='gsk3b'
#   grpo_kinase.pt             -> algo='grpo' obj='kinase'
OBJ_ALIASES = {
    # accepted suffix -> (obj_short, MoleculeConfig.objective_type)
    "kinase":     ("kinase", "kinase_mpo"),
    "kinasempo":  ("kinase", "kinase_mpo"),
    "kinase_mpo": ("kinase", "kinase_mpo"),
    "jnk3":       ("jnk3",   "jnk3"),
    "jnk":        ("jnk3",   "jnk3"),
    "gsk3":       ("gsk3",   "gsk3"),
    "gsk3b":      ("gsk3",   "gsk3"),
    "gsk":        ("gsk3",   "gsk3"),
}


def parse_model_name(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Returns (algo_tag, obj_short, objective_type) or None if it can't be parsed.

    The objective is read from the substring AFTER the LAST '_' in the
    filename (case-insensitive). Everything before that is treated as the
    algo / variant tag and only used for naming.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    if "_" not in base:
        return None
    algo_tag, obj_token = base.rsplit("_", 1)
    key = obj_token.lower()
    if key not in OBJ_ALIASES:
        return None
    obj_short, obj_type = OBJ_ALIASES[key]
    return algo_tag, obj_short, obj_type


# ---------------------------------------------------------------------------
# Config / dataset helpers
# ---------------------------------------------------------------------------
def _build_eval_config(base_cfg_cls,
                       objective_type: str,
                       scaffolds_path: Optional[str],
                       results_path: str,
                       num_samples: int,
                       beam_width: int,
                       seed: int,
                       batch_size_per_worker: int = 16) -> MoleculeConfig:
    """Build a clean inference-only config for one model run."""
    cfg = base_cfg_cls()
    cfg.objective_type = objective_type
    cfg.num_epochs = 0
    cfg.load_checkpoint_from_path = None
    cfg.use_wandb = False
    cfg.prodrug_mode = False  # not relevant for these tasks
    cfg.results_path = results_path
    cfg.seed = seed

    # Disable EVERY lambda-related code path. These tasks are single-
    # objective, so lambda is meaningless. We:
    #   * turn off corner sampling and the extremes restriction,
    #   * unset eval_lambda,
    #   * turn off FiLM and additive-lambda conditioning in the model
    #     (no-op for checkpoints that don't have those modules anyway,
    #     because we load with strict=False),
    # and we also pass `fixed_lambda=[1.0]` to `generate_dataset` below
    # so the dataset's `_sample_lambda_vec()` is never called.
    cfg.use_corner_sampling = False
    cfg.restrict_training_lambda_to_extremes = False
    cfg.eval_lambda = None
    cfg.use_film = False
    cfg.use_lambda_additive = False

    if scaffolds_path is not None:
        cfg.evaluation_scaffolds_path = scaffolds_path

    # ---- Stochastic sampling so the 3 runs are not identical ----
    cfg.gumbeldore_config = copy.deepcopy(cfg.gumbeldore_config)
    cfg.gumbeldore_config["search_type"] = "iid_mc"
    cfg.gumbeldore_config["num_samples_per_instance"] = num_samples
    cfg.gumbeldore_config["sampling_temperature"] = 1.0
    cfg.gumbeldore_config["deterministic"] = False
    cfg.gumbeldore_config["destination_path"] = None
    cfg.gumbeldore_config["beam_width"] = beam_width
    cfg.gumbeldore_config["num_trajectories_to_keep"] = num_samples
    cfg.gumbeldore_config["num_rounds"] = 1

    # *** REAL speedup knob ***
    # batch_size_per_worker controls how many prompts the GPU worker
    # processes per `batched_iid_monte_carlo_sampling` call. Default in
    # MoleculeConfig is 1 (one prompt per GPU forward) — that's the
    # bottleneck. Bump it so the worker actually batches prompts on the
    # GPU.
    cfg.gumbeldore_config["batch_size_per_worker"] = max(1, int(batch_size_per_worker))
    cfg.gumbeldore_config["batch_size_per_cpu_worker"] = max(
        1, int(batch_size_per_worker)
    )

    # Make sure devices_for_workers contains something sensible
    if torch.cuda.is_available():
        cfg.training_device = "cuda:0"
        cfg.gumbeldore_config["devices_for_workers"] = ["cuda:0"]
    else:
        cfg.training_device = "cpu"
        cfg.gumbeldore_config["devices_for_workers"] = ["cpu"]

    return cfg


def _load_scaffolds(scaffolds_path: str) -> List[str]:
    if not scaffolds_path or not os.path.exists(scaffolds_path):
        raise FileNotFoundError(
            f"Test scaffolds file not found: {scaffolds_path}"
        )
    with open(scaffolds_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Per-objective per-molecule scoring
# ---------------------------------------------------------------------------
def _score_extras(smiles: str,
                  obj_short: str,
                  kinase_oracle: Optional[KinaseMPOObjective]
                  ) -> Tuple[Optional[float], Optional[float],
                             Optional[float], Optional[float],
                             Optional[bool]]:
    """
    Returns (gsk3, jnk3, qed, sa, success) where:
       - gsk3, jnk3 are TDC oracle probabilities
       - qed, sa are only filled for kinase
       - success is the strict kinase MPO success criterion (else None)
    For jnk3 / gsk3 single-objective models, the relevant component
    is ALSO returned so we always have one consistent column.
    """
    gsk = jnk = qed = sa = None
    success = None
    if not smiles:
        return gsk, jnk, qed, sa, success

    if obj_short == "kinase" and kinase_oracle is not None:
        ind = kinase_oracle.individual_scores(smiles)
        gsk = float(ind.get("GSK3B")) if ind.get("GSK3B") is not None else None
        jnk = float(ind.get("JNK3")) if ind.get("JNK3") is not None else None
        qed = float(ind.get("QED")) if ind.get("QED") is not None else None
        sa = float(ind.get("SA")) if ind.get("SA") is not None else None
        try:
            success = bool(kinase_oracle.is_successful(smiles))
        except Exception:
            success = None
    return gsk, jnk, qed, sa, success


# ---------------------------------------------------------------------------
# One run = generate molecules for every test scaffold once
# ---------------------------------------------------------------------------
def _run_inference_once(network: MoleculeTransformer,
                        cfg: MoleculeConfig,
                        objective_evaluator: MoleculeObjectiveEvaluator,
                        scaffolds: List[str],
                        obj_short: str,
                        kinase_oracle: Optional[KinaseMPOObjective],
                        run_idx: int,
                        per_run_csv_path: str,
                        scaffold_batch_size: int = 1) -> dict:
    """
    Generates molecules for every scaffold once, picks the best per scaffold,
    writes a detailed CSV, and returns aggregate metrics for this run.

    `scaffold_batch_size` controls how many prompts are sent to
    `generate_dataset` at once. Bigger = faster (better GPU utilisation),
    bounded by GPU memory.
    """
    eval_cfg = copy.deepcopy(cfg)
    eval_cfg.gumbeldore_config["destination_path"] = None

    dataset = GumbeldoreDataset(config=eval_cfg,
                                objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    # ---- per-mol logs ----
    fieldnames = [
        "run_idx", "scaffold_idx", "prompt_smiles", "generated_smiles",
        "objective_score", "is_successful",
        "gsk3", "jnk3", "qed", "sa",
    ]
    os.makedirs(os.path.dirname(per_run_csv_path), exist_ok=True)
    f = open(per_run_csv_path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    top1_objs: List[float] = []
    gsk_list: List[float] = []
    jnk_list: List[float] = []
    qed_list: List[float] = []
    sa_list:  List[float] = []
    successes: List[int] = []
    n_valid = 0

    bs = max(1, int(scaffold_batch_size))
    n = len(scaffolds)
    n_batches = (n + bs - 1) // bs

    # All tasks here are single-objective. Pass an explicit lambda=[1.0]
    # so `generate_dataset` never calls `_sample_lambda_vec()` — i.e.
    # zero contact with the lambda sampling machinery.
    fixed_lambda = np.array([1.0], dtype=np.float32)

    pbar = tqdm(total=n, desc=f"Run {run_idx} (bs={bs})")

    for b in range(n_batches):
        start = b * bs
        end = min(start + bs, n)
        batch_prompts = scaffolds[start:end]
        batch_indices = list(range(start, end))

        try:
            grouped = dataset.generate_dataset(
                network_weights=weights,
                memory_aggressive=False,
                prompts=batch_prompts,
                return_raw_trajectories=True,
                mode="eval",
                fixed_lambda=fixed_lambda,
            )
        except Exception as e:
            print(f"[run {run_idx}] generation failed for batch "
                  f"[{start}:{end}]: {e}")
            grouped = None

        # Normalize: if generate_dataset returned None or fewer groups
        # than prompts, pad so per-prompt indexing is safe.
        if grouped is None:
            grouped = [None] * len(batch_prompts)
        elif len(grouped) < len(batch_prompts):
            grouped = list(grouped) + [None] * (
                len(batch_prompts) - len(grouped))

        for local_i, (idx, prompt) in enumerate(zip(batch_indices,
                                                    batch_prompts)):
            group = grouped[local_i]
            pbar.update(1)

            if not group:
                writer.writerow({
                    "run_idx": run_idx, "scaffold_idx": idx,
                    "prompt_smiles": prompt,
                    "generated_smiles": "GENERATION_FAILED",
                    "objective_score": "", "is_successful": "",
                    "gsk3": "", "jnk3": "", "qed": "", "sa": "",
                })
                continue

            valid = [m for m in group
                     if getattr(m, "objective", None) is not None
                     and m.smiles_string]
            if not valid:
                writer.writerow({
                    "run_idx": run_idx, "scaffold_idx": idx,
                    "prompt_smiles": prompt,
                    "generated_smiles": "NO_VALID",
                    "objective_score": "", "is_successful": "",
                    "gsk3": "", "jnk3": "", "qed": "", "sa": "",
                })
                continue

            best = max(valid, key=lambda m: m.objective)
            smi = best.smiles_string
            obj_val = float(best.objective)

            gsk, jnk, qed, sa, success = _score_extras(
                smi, obj_short, kinase_oracle
            )

            # For pure jnk3 / gsk3 models, the model's objective IS that
            # score, so mirror it into the matching column.
            if obj_short == "jnk3" and jnk is None:
                jnk = obj_val
            if obj_short == "gsk3" and gsk is None:
                gsk = obj_val

            writer.writerow({
                "run_idx": run_idx, "scaffold_idx": idx,
                "prompt_smiles": prompt,
                "generated_smiles": smi,
                "objective_score": obj_val,
                "is_successful": success if success is not None else "",
                "gsk3": gsk if gsk is not None else "",
                "jnk3": jnk if jnk is not None else "",
                "qed":  qed if qed is not None else "",
                "sa":   sa  if sa  is not None else "",
            })

            top1_objs.append(obj_val)
            if gsk is not None: gsk_list.append(gsk)
            if jnk is not None: jnk_list.append(jnk)
            if qed is not None: qed_list.append(qed)
            if sa  is not None: sa_list.append(sa)
            if success is not None: successes.append(int(success))
            n_valid += 1

        # Flush rows in case we OOM later, and free memory before the
        # next batch.
        f.flush()
        del grouped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    pbar.close()
    f.close()

    def _mean(xs): return float(np.mean(xs)) if xs else float("nan")

    metrics = {
        "n_scaffolds":  len(scaffolds),
        "n_valid":      n_valid,
        "mean_top1_obj": _mean(top1_objs),
        "mean_gsk3":    _mean(gsk_list),
        "mean_jnk3":    _mean(jnk_list),
        "mean_qed":     _mean(qed_list),
        "mean_sa":      _mean(sa_list),
        "success_rate": (float(np.mean(successes)) if successes
                         else float("nan")),
    }
    return metrics


# ---------------------------------------------------------------------------
# Driver: evaluate one checkpoint over N runs
# ---------------------------------------------------------------------------
def evaluate_checkpoint(checkpoint_path: str,
                        algo: str,
                        obj_short: str,
                        objective_type: str,
                        base_cfg_cls,
                        scaffold_files: List[Tuple[int, str]],
                        output_dir: str,
                        num_samples: int,
                        beam_width: int,
                        oracle_tracker,
                        scaffold_batch_size: int = 1,
                        batch_size_per_worker: int = 16,
                        on_run_complete: Optional[Callable[[dict], None]] = None
                        ) -> List[dict]:
    """
    Evaluate one model. Runs inference once per scaffold file (i.e. once per
    test-set seed) and returns one metrics dict per run.
    """
    model_tag = f"{algo}_{obj_short}"
    print("\n" + "=" * 70)
    print(f"[neurips-eval] {model_tag}  ({checkpoint_path})")
    print(f"               objective_type = {objective_type}")
    print(f"               {len(scaffold_files)} scaffold seed(s): "
          f"{[s for s, _ in scaffold_files]}")
    print("=" * 70)

    model_out_dir = os.path.join(output_dir, model_tag)
    os.makedirs(model_out_dir, exist_ok=True)

    # One kinase oracle is enough; reuse across runs
    kinase_oracle = (KinaseMPOObjective()
                     if obj_short == "kinase" else None)

    run_metrics: List[dict] = []

    for scaffold_seed, scaffolds_path in scaffold_files:
        scaffolds = _load_scaffolds(scaffolds_path)
        print(f"[neurips-eval] seed={scaffold_seed}  "
              f"{len(scaffolds)} test scaffolds  ({scaffolds_path})")

        # Use the test-set seed as RNG seed too so runs are reproducible
        # but distinct across the three test sets.
        seed = int(scaffold_seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        cfg = _build_eval_config(
            base_cfg_cls,
            objective_type=objective_type,
            scaffolds_path=scaffolds_path,
            results_path=model_out_dir,
            num_samples=num_samples,
            beam_width=beam_width,
            seed=seed,
            batch_size_per_worker=batch_size_per_worker,
        )

        # Build model + objective evaluator using the run's config so the
        # right TDC oracles are instantiated for this objective_type.
        network = MoleculeTransformer(cfg, cfg.training_device)
        objective_eval = MoleculeObjectiveEvaluator(
            cfg, device=cfg.objective_gnn_device,
            oracle_tracker=oracle_tracker,
        )

        ckpt = torch.load(checkpoint_path, map_location=cfg.training_device,
                          weights_only=False)
        if isinstance(ckpt, dict) and "model_weights" in ckpt:
            state_dict = ckpt["model_weights"]
        elif isinstance(ckpt, dict) and "best_model_weights" in ckpt:
            state_dict = ckpt["best_model_weights"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            # Raw state_dict
            state_dict = ckpt

        # Always non-strict: PPO checkpoints can carry value-head /
        # critic / optimizer side modules that are not part of the
        # generator network.
        missing, unexpected = network.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[neurips-eval] missing keys ({len(missing)}): "
                  f"{missing[:5]}{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[neurips-eval] unexpected keys ({len(unexpected)}): "
                  f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")

        network.to(network.device)
        network.eval()

        per_run_csv = os.path.join(
            model_out_dir, f"{model_tag}_seed{scaffold_seed}_detailed.csv"
        )
        with torch.no_grad():
            metrics = _run_inference_once(
                network=network,
                cfg=cfg,
                objective_evaluator=objective_eval,
                scaffolds=scaffolds,
                obj_short=obj_short,
                kinase_oracle=kinase_oracle,
                run_idx=scaffold_seed,
                per_run_csv_path=per_run_csv,
                scaffold_batch_size=scaffold_batch_size,
            )
        metrics.update({
            "model_name": model_tag,
            "algo": algo,
            "objective": obj_short,
            "objective_type": objective_type,
            "checkpoint": os.path.abspath(checkpoint_path),
            "run_idx": scaffold_seed,
            "scaffold_seed": scaffold_seed,
            "scaffolds_path": scaffolds_path,
            "seed": seed,
        })
        print(f"[neurips-eval] {model_tag} seed={scaffold_seed}: "
              f"obj={metrics['mean_top1_obj']:.4f}  "
              f"succ={metrics['success_rate']}  "
              f"gsk={metrics['mean_gsk3']}  jnk={metrics['mean_jnk3']}")
        run_metrics.append(metrics)

        # Persist master CSVs immediately after this run so progress is
        # never lost (caller appends + rewrites both CSVs).
        if on_run_complete is not None:
            try:
                on_run_complete(metrics)
            except Exception as e:
                print(f"[neurips-eval] WARN: on_run_complete failed: {e}")

        # Clean up
        del network, objective_eval, ckpt, state_dict
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    return run_metrics


# ---------------------------------------------------------------------------
# Aggregation over runs
# ---------------------------------------------------------------------------
AGG_FIELDS = ("mean_top1_obj", "success_rate",
              "mean_gsk3", "mean_jnk3", "mean_qed", "mean_sa")


def _aggregate_per_model(run_rows: List[dict]) -> dict:
    """Return mean / std across runs for one model."""
    if not run_rows:
        return {}
    out = {
        "model_name": run_rows[0]["model_name"],
        "algo": run_rows[0]["algo"],
        "objective": run_rows[0]["objective"],
        "objective_type": run_rows[0]["objective_type"],
        "n_runs": len(run_rows),
        "checkpoint": run_rows[0]["checkpoint"],
    }
    for k in AGG_FIELDS:
        vals = [r[k] for r in run_rows
                if r.get(k) is not None and not (isinstance(r[k], float)
                                                 and np.isnan(r[k]))]
        out[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        out[f"{k}_std"] = (float(np.std(vals, ddof=0))
                           if len(vals) > 1 else 0.0)
    return out


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the neurips trained checkpoints in inference mode."
    )
    parser.add_argument("--models_dir", type=str, default="model/neurips",
                        help="Directory containing model checkpoint files.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional config module path (e.g. configs.my_config). "
                             "If omitted, the default MoleculeConfig is used.")
    parser.add_argument("--scaffolds_dir", type=str,
                        default="scaffold_splitting/zinc_splits_optimized",
                        help="Root directory containing run_seed_<N>/test_scaffolds.txt.")
    parser.add_argument("--scaffold_seeds", type=str, default="42,43,44",
                        help="Comma-separated list of scaffold seeds to evaluate "
                             "(one inference run per seed).")
    parser.add_argument("--scaffolds_path", type=str, default=None,
                        help="Optional override: a single test_scaffolds.txt to "
                             "evaluate against instead of --scaffolds_dir/--scaffold_seeds.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Where to write per-model CSVs and the master "
                             "summary. Defaults to "
                             "results/neurips_eval_<YYYY-MM-DD--HH-MM-SS>.")
    parser.add_argument("--num_samples", type=int, default=32,
                        help="Number of molecules sampled per scaffold per run.")
    parser.add_argument("--beam_width", type=int, default=1,
                        help="Beam width passed to gumbeldore (default 1; iid_mc).")
    parser.add_argument("--scaffold_batch_size", type=int, default=128,
                        help="How many scaffolds to send to generate_dataset() "
                             "in one call. Mostly amortises setup cost; the "
                             "actual GPU batching is controlled by "
                             "--batch_size_per_worker.")
    parser.add_argument("--batch_size_per_worker", type=int, default=16,
                        help="How many prompts the GPU worker processes per "
                             "forward pass (the *real* speedup knob). Bigger = "
                             "faster, bounded by GPU memory. Drop to 4-8 if you "
                             "OOM, raise to 32+ if you have headroom.")
    parser.add_argument("--only", type=str, default=None,
                        help="Comma-separated list of model_tags to run "
                             "(e.g. ppo_kinase,grpo_jnk3). Default: all matched.")
    parser.add_argument("--disable_ray", action="store_true",
                        help="Run without Ray (slower; helpful for debugging).")
    args = parser.parse_args()

    # Resolve config class
    if args.config:
        BaseCfg = importlib.import_module(args.config).MoleculeConfig
    else:
        BaseCfg = MoleculeConfig

    # ---- Resolve output dir (default: results/neurips_eval_<timestamp>) ----
    if args.output_dir is None:
        ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        args.output_dir = os.path.join("results", f"neurips_eval_{ts}")
    print(f"[neurips-eval] Writing outputs to: {args.output_dir}")

    # ---- Resolve scaffold seed files ----
    scaffold_files: List[Tuple[int, str]] = []
    if args.scaffolds_path is not None:
        if not os.path.exists(args.scaffolds_path):
            print(f"[ERROR] --scaffolds_path not found: {args.scaffolds_path}",
                  file=sys.stderr)
            sys.exit(1)
        scaffold_files.append((0, args.scaffolds_path))
    else:
        seeds = [int(s.strip()) for s in args.scaffold_seeds.split(",")
                 if s.strip()]
        for s in seeds:
            p = os.path.join(args.scaffolds_dir,
                             f"run_seed_{s}", "test_scaffolds.txt")
            if not os.path.exists(p):
                print(f"[WARN] scaffold file missing for seed {s}: {p}",
                      file=sys.stderr)
                continue
            scaffold_files.append((s, p))
        if not scaffold_files:
            print("[ERROR] No valid scaffold seed files found.",
                  file=sys.stderr)
            sys.exit(1)
    print(f"[neurips-eval] Using {len(scaffold_files)} scaffold seed file(s):")
    for s, p in scaffold_files:
        print(f"   - seed={s}  ->  {p}")

    # Discover models
    models_dir = args.models_dir
    if not os.path.isdir(models_dir):
        print(f"[ERROR] models_dir not found: {models_dir}", file=sys.stderr)
        sys.exit(1)

    candidates = []
    for fname in sorted(os.listdir(models_dir)):
        if not fname.lower().endswith(".pt"):
            continue
        parsed = parse_model_name(fname)
        if parsed is None:
            print(f"[skip] Cannot infer (algo, obj) from filename: {fname}")
            continue
        algo, obj_short, obj_type = parsed
        tag = f"{algo}_{obj_short}"
        if args.only:
            allowed = {s.strip() for s in args.only.split(",") if s.strip()}
            if tag not in allowed:
                continue
        candidates.append((os.path.join(models_dir, fname),
                           algo, obj_short, obj_type))

    if not candidates:
        print("[ERROR] No models matched. Check --models_dir / naming.",
              file=sys.stderr)
        sys.exit(1)

    print(f"[neurips-eval] Found {len(candidates)} models:")
    for path, algo, obj_short, obj_type in candidates:
        print(f"   - {os.path.basename(path)}  -> algo={algo}  obj={obj_short}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Probe config for ray sizing
    probe_cfg = BaseCfg()
    if args.disable_ray:
        oracle_tracker = LocalOracleTracker()
    else:
        if ray.is_initialized():
            ray.shutdown()
        num_gpus = (len(probe_cfg.CUDA_VISIBLE_DEVICES.split(","))
                    if torch.cuda.is_available() else 0)
        ray_args = dict(num_gpus=num_gpus, logging_level="info",
                        ignore_reinit_error=True)
        if platform.system() == "Windows":
            ray_args["include_dashboard"] = False
            ray_args["_temp_dir"] = "C:/ray_tmp"
            ray_args["address"] = "local"
        ray.init(**ray_args)
        print("Ray resources:", ray.available_resources())
        oracle_tracker = OracleTracker.remote()

    all_run_rows: List[dict] = []
    all_summary_rows: List[dict] = []

    # Map (model_tag) -> index in all_summary_rows so we can update the
    # aggregate row in-place each time a new run for that model finishes.
    summary_idx_by_tag: dict = {}

    # Group rows by model so we can recompute per-model aggregate after
    # every individual run.
    rows_by_tag: dict = {}

    def _flush(latest_run: Optional[dict] = None):
        """Persist both master CSVs after each individual run."""
        if latest_run is not None:
            tag = latest_run["model_name"]
            all_run_rows.append(latest_run)
            rows_by_tag.setdefault(tag, []).append(latest_run)
            agg = _aggregate_per_model(rows_by_tag[tag])
            if tag in summary_idx_by_tag:
                all_summary_rows[summary_idx_by_tag[tag]] = agg
            else:
                summary_idx_by_tag[tag] = len(all_summary_rows)
                all_summary_rows.append(agg)
        _write_master_csvs(args.output_dir, all_run_rows, all_summary_rows)

    try:
        for path, algo, obj_short, obj_type in candidates:
            try:
                evaluate_checkpoint(
                    checkpoint_path=path,
                    algo=algo,
                    obj_short=obj_short,
                    objective_type=obj_type,
                    base_cfg_cls=BaseCfg,
                    scaffold_files=scaffold_files,
                    output_dir=args.output_dir,
                    num_samples=args.num_samples,
                    beam_width=args.beam_width,
                    oracle_tracker=oracle_tracker,
                    scaffold_batch_size=args.scaffold_batch_size,
                    batch_size_per_worker=args.batch_size_per_worker,
                    on_run_complete=_flush,
                )
            except Exception as e:
                print(f"[ERROR] Failed evaluating {path}: {e}")
                import traceback; traceback.print_exc()
                # Make sure whatever progress we have is on disk
                _flush()
                continue
    finally:
        # Final flush guard
        _flush()
        if not args.disable_ray and ray.is_initialized():
            ray.shutdown()

    print("\n[neurips-eval] DONE.")
    print(f"  Per-run CSV   : {os.path.join(args.output_dir, 'neurips_eval_runs.csv')}")
    print(f"  Summary CSV   : {os.path.join(args.output_dir, 'neurips_eval_summary.csv')}")


def _write_master_csvs(output_dir: str,
                       run_rows: List[dict],
                       summary_rows: List[dict]):
    if run_rows:
        run_csv = os.path.join(output_dir, "neurips_eval_runs.csv")
        run_fields = [
            "model_name", "algo", "objective", "objective_type",
            "scaffold_seed", "run_idx", "seed", "scaffolds_path",
            "n_scaffolds", "n_valid",
            "mean_top1_obj", "success_rate",
            "mean_gsk3", "mean_jnk3", "mean_qed", "mean_sa",
            "checkpoint",
        ]
        with open(run_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=run_fields, extrasaction="ignore")
            w.writeheader()
            for r in run_rows:
                w.writerow(r)

    if summary_rows:
        sum_csv = os.path.join(output_dir, "neurips_eval_summary.csv")
        sum_fields = [
            "model_name", "algo", "objective", "objective_type", "n_runs",
            "mean_top1_obj_mean", "mean_top1_obj_std",
            "success_rate_mean",  "success_rate_std",
            "mean_gsk3_mean",     "mean_gsk3_std",
            "mean_jnk3_mean",     "mean_jnk3_std",
            "mean_qed_mean",      "mean_qed_std",
            "mean_sa_mean",       "mean_sa_std",
            "checkpoint",
        ]
        with open(sum_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)


if __name__ == "__main__":
    main()

