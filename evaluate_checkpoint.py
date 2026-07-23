"""
evaluate_checkpoint.py
Standalone script to evaluate trained model checkpoints without retraining.
Usage:
    python evaluate_checkpoint.py --checkpoint path/to/best_model.pt
    python evaluate_checkpoint.py --checkpoint path/to/best_model.pt --config configs.my_config
    python evaluate_checkpoint.py --checkpoint path/to/best_model.pt --batch_size 250
    python evaluate_checkpoint.py --checkpoint path/to/best_model.pt --subset_percent 1
"""

import argparse
import copy
import importlib
import os
import gc

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

import ray
import torch
import numpy as np
from tqdm import tqdm

from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer
from molecule_evaluator import MoleculeObjectiveEvaluator


def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator, batch_size: int = 500,
             subset_percent: float = 100.0):
    """Batched evaluation to avoid OOM."""
    config = copy.deepcopy(config)
    config.gumbeldore_config["destination_path"] = None

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )

    test_prompts = None
    use_batched_eval = False
    EVAL_BATCH_SIZE = batch_size

    if getattr(config, 'evaluation_scaffolds_path', None):
        path = config.evaluation_scaffolds_path
        if os.path.exists(path):
            print(f"[Eval] Loading Test Scaffolds from: {path}")
            with open(path, 'r') as f:
                test_prompts = [line.strip() for line in f if line.strip()]

            # Subset to first N% if requested (deterministic, no randomness)
            if subset_percent < 100.0:
                original_count = len(test_prompts)
                subset_count = max(1, int(len(test_prompts) * subset_percent / 100.0))
                test_prompts = test_prompts[:subset_count]
                print(f"[Eval] Using first {subset_percent}% of scaffolds: {subset_count}/{original_count}")

            if len(test_prompts) > EVAL_BATCH_SIZE:
                use_batched_eval = True
                print(f"[Eval] Large test set ({len(test_prompts)} scaffolds). Using batched evaluation.")
        else:
            print(f"[Eval] WARNING: Config path {path} not found.")
    elif config.prodrug_mode:
        test_prompts = config.prodrug_parents_test
        # Also apply subset to prodrug mode
        if subset_percent < 100.0 and test_prompts:
            original_count = len(test_prompts)
            subset_count = max(1, int(len(test_prompts) * subset_percent / 100.0))
            test_prompts = test_prompts[:subset_count]
            print(f"[Eval] Using first {subset_percent}% of scaffolds: {subset_count}/{original_count}")

    SUCCESS_THRESHOLD = 0.5
    scaffold_metrics = []
    all_valid_mols = []

    if use_batched_eval and test_prompts is not None:
        num_batches = (len(test_prompts) + EVAL_BATCH_SIZE - 1) // EVAL_BATCH_SIZE
        print(f"[Eval] Processing {len(test_prompts)} scaffolds in {num_batches} batches...")

        for batch_idx in tqdm(range(num_batches), desc="Eval Batches"):
            start_idx = batch_idx * EVAL_BATCH_SIZE
            end_idx = min(start_idx + EVAL_BATCH_SIZE, len(test_prompts))
            batch_prompts = test_prompts[start_idx:end_idx]

            grouped_trajectories = gumbeldore_dataset.generate_dataset(
                copy.deepcopy(network.get_weights()),
                memory_aggressive=False,
                prompts=batch_prompts,
                return_raw_trajectories=True
            )

            for group in grouped_trajectories:
                if not group:
                    continue
                objs = [m.objective for m in group if m.objective is not None]
                valid_mols = [m for m in group if m.objective is not None]
                all_valid_mols.extend(valid_mols)

                if not objs:
                    scaffold_metrics.append({"solved": 0.0, "top1": 0.0, "mean": 0.0})
                    continue

                best_score = max(objs)
                mean_score = np.mean(objs)
                is_solved = objective_evaluator.kinase_mpo_objective.is_successful(smi)
                scaffold_metrics.append({"solved": is_solved, "top1": best_score, "mean": mean_score})

            del grouped_trajectories
            torch.cuda.empty_cache()
            gc.collect()
    else:
        grouped_trajectories = gumbeldore_dataset.generate_dataset(
            copy.deepcopy(network.get_weights()),
            memory_aggressive=False,
            prompts=test_prompts,
            return_raw_trajectories=True
        )

        for group in grouped_trajectories:
            if not group:
                continue
            objs = [m.objective for m in group if m.objective is not None]
            valid_mols = [m for m in group if m.objective is not None]
            all_valid_mols.extend(valid_mols)

            if not objs:
                scaffold_metrics.append({"solved": 0.0, "top1": 0.0, "mean": 0.0})
                continue

            best_score = max(objs)
            mean_score = np.mean(objs)
            is_solved = 1.0 if best_score > SUCCESS_THRESHOLD else 0.0
            scaffold_metrics.append({"solved": is_solved, "top1": best_score, "mean": mean_score})

    if not scaffold_metrics:
        print("[Eval] Warning: No valid molecules generated.")
        return {}, []

    avg_success_rate = np.mean([m["solved"] for m in scaffold_metrics])
    avg_top1_score = np.mean([m["top1"] for m in scaffold_metrics])
    avg_mean_score = np.mean([m["mean"] for m in scaffold_metrics])

    metrics_out = {
        f"{eval_type}_success_rate": avg_success_rate,
        f"{eval_type}_mean_top1_obj": avg_top1_score,
        f"{eval_type}_global_mean_obj": avg_mean_score,
        f"{eval_type}_num_scaffolds_evaluated": len(scaffold_metrics)
    }

    print("=" * 30)
    print(f"EVALUATION REPORT ({eval_type})")
    print(f"Scaffolds Processed: {len(scaffold_metrics)}")
    print(f"Success Rate (> {SUCCESS_THRESHOLD}): {avg_success_rate * 100:.2f}%")
    print(f"Mean Top-1 Score: {avg_top1_score:.4f}")
    print("=" * 30)

    all_valid_mols.sort(key=lambda x: x.objective, reverse=True)
    top_20_objects = all_valid_mols[:20]

    top_20_text_lines = []
    for i, m in enumerate(top_20_objects):
        smi = m.smiles_string if m.smiles_string else "Invalid"
        top_20_text_lines.append(f"{i + 1:02d}: {smi}  obj={m.objective:.4f}")

    return metrics_out, top_20_text_lines


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained checkpoint')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (e.g., best_model.pt)')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional config module path (e.g., configs.my_config)')
    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size for evaluation (default: 500)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results (default: same as checkpoint)')
    parser.add_argument('--subset_percent', type=float, default=100.0,
                        help='Percentage of test scaffolds to use (1-100, default: 100). Takes first N%% deterministically.')
    args = parser.parse_args()

    # Load config
    if args.config is not None:
        ConfigClass = importlib.import_module(args.config).MoleculeConfig
    else:
        ConfigClass = MoleculeConfig
    config = ConfigClass()

    # Set num_epochs to 0 to skip training
    config.num_epochs = 0

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Evaluating checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")

    # Initialize Ray (skip entirely when the master switch is off)
    use_ray = getattr(config, 'use_ray', True)
    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    if use_ray:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(num_gpus=num_gpus, logging_level="info", ignore_reinit_error=True)
    else:
        print("[evaluate_checkpoint] Ray disabled (config.use_ray=False). Running serially in-process.")

    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load model
    network = MoleculeTransformer(config, config.training_device)
    objective_eval = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

    checkpoint = torch.load(args.checkpoint, map_location=config.training_device)
    network.load_state_dict(checkpoint["model_weights"])
    print(f"Loaded checkpoint trained for {checkpoint.get('epochs_trained', 'unknown')} epochs")
    print(f"Best validation metric from checkpoint: {checkpoint.get('best_validation_metric', 'N/A')}")

    network.to(network.device)
    network.eval()

    # Run evaluation
    torch.cuda.empty_cache()
    with torch.no_grad():
        test_metrics, test_top20 = evaluate('test', config, network, objective_eval,
                                            batch_size=args.batch_size,
                                            subset_percent=args.subset_percent)

    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Epochs trained: {checkpoint.get('epochs_trained', 'unknown')}\n")
        f.write(f"Subset percent: {args.subset_percent}%\n\n")
        f.write("METRICS:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v}\n")
        f.write("\nTOP 20 MOLECULES:\n")
        for line in test_top20:
            f.write(f"  {line}\n")

    print(f"\nResults saved to: {results_file}")
    print("\nMetrics:", test_metrics)

    if use_ray:
        ray.shutdown()


if __name__ == '__main__':
    main()
