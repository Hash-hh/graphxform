import argparse
import copy
import importlib
import os
import time
from typing import List, Optional
from operator import attrgetter

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle
import csv

from logger import Logger
from molecule_dataset import RandomMoleculeDataset
from prodrug_test import get_prodrug_test_parents


os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import torch
import numpy as np
import wandb
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_evaluator import MoleculeObjectiveEvaluator, OracleTracker, LocalOracleTracker
from rl_updates import dr_grpo_update, TrajectoryRecord

os.environ["RAY_raylet_start_wait_time_s"] = "120"  # Increase from default 60s


# ==============================================================================
# UNWEIGHTED BIOLOGY SCORER & FORMATTING
# ==============================================================================
def get_unweighted_score(mol, objective_type):
    if not hasattr(mol, 'aux_metrics') or not mol.aux_metrics:
        return mol.objective if mol.objective is not None else float("-inf")

    m = mol.aux_metrics
    if objective_type == 'polypharmacy_2d':
        return m.get('gsk3b', 0.0) + m.get('jnk3', 0.0)
    elif objective_type == 'safety_2d':
        return m.get('jnk3', 0.0) + (1.0 - m.get('herg', 1.0))
    elif objective_type == 'tpp_3d':
        return m.get('gsk3b', 0.0) + m.get('bbb', 0.0) + (1.0 - m.get('herg', 1.0))
    else:
        return mol.objective if mol.objective is not None else float("-inf")


def get_oracle_count(oracle_tracker, config):
    """Get oracle count, handling both Ray and local trackers."""
    if getattr(config, 'disable_ray', False):
        return oracle_tracker.get_count()
    return ray.get(oracle_tracker.get_count.remote())


def format_aux_metrics(aux: dict, objective_type: str) -> str:
    if objective_type == 'tpp_3d':
        return f"GSK={aux.get('gsk3b', 0):.3f}, BBB={aux.get('bbb', 0):.3f}, hERG={aux.get('herg', 1):.3f}"
    elif objective_type == 'safety_2d':
        return f"JNK={aux.get('jnk3', 0):.3f}, hERG={aux.get('herg', 1):.3f}"
    elif objective_type == 'polypharmacy_2d':
        return f"GSK={aux.get('gsk3b', 0):.3f}, JNK={aux.get('jnk3', 0):.3f}"
    return f"Raw={aux}"


# ==============================================================================

def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def _make_eval_config(config_orig, beam_width=1, num_keep=1000):
    """Create a deterministic beam search config for validation/evaluation."""
    config = copy.deepcopy(config_orig)
    config.gumbeldore_config["search_type"] = "beam_search"
    config.gumbeldore_config["beam_width"] = beam_width
    config.gumbeldore_config["deterministic"] = True
    config.gumbeldore_config["num_trajectories_to_keep"] = num_keep
    config.gumbeldore_config["destination_path"] = None
    return config


def _load_scaffolds(config, scaffold_attr, eval_type="Eval"):
    """Load scaffold prompts from config path or prodrug parents."""
    if config.prodrug_mode:
        parents = get_prodrug_test_parents()
        print(f"[{eval_type}] Using {len(parents)} Prodrug test parents from prodrug_test.py.")
        return [smi for _, smi in parents]
    path = getattr(config, scaffold_attr, None)
    if path and os.path.exists(path):
        print(f"[{eval_type}] Loading Scaffolds from: {path}")
        with open(path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def validate_epoch(config: MoleculeConfig, network: MoleculeTransformer,
                   objective_evaluator: MoleculeObjectiveEvaluator):
    val_scaffolds = _load_scaffolds(config, 'validation_scaffolds_path', "Val")
    if not val_scaffolds:
        return float("-inf"), 0.0

    # --- SPEED FIX: subsample validation scaffolds ---
    MAX_VAL_SCAFFOLDS = 20  # ~1 min instead of ~5 min per epoch
    if len(val_scaffolds) > MAX_VAL_SCAFFOLDS:
        # Deterministic subsample: always pick the same ones so scores are comparable
        rng = np.random.RandomState(config.seed)
        indices = rng.choice(len(val_scaffolds), size=MAX_VAL_SCAFFOLDS, replace=False)
        val_scaffolds = [val_scaffolds[i] for i in sorted(indices)]
        print(f"[Val] Subsampled to {len(val_scaffolds)} scaffolds for speed.")

    val_config = _make_eval_config(config, beam_width=1, num_keep=1)
    dataset = GumbeldoreDataset(config=val_config, objective_evaluator=objective_evaluator)

    num_obj = getattr(config, 'num_objectives', 1)
    uniform_lambda = np.ones(num_obj) / num_obj if num_obj > 1 else None

    print(f"[Val] Validating on {len(val_scaffolds)} scaffolds (Uniform Lambda: {uniform_lambda})...")

    grouped_results = dataset.generate_dataset(
        network_weights=copy.deepcopy(network.get_weights()),
        memory_aggressive=False,
        prompts=val_scaffolds,
        return_raw_trajectories=True,
        mode="eval",
        fixed_lambda=uniform_lambda
    )

    scores = []
    success_count = 0
    total_mols = 0
    check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator, 'kinase_mpo_objective')

    for group in grouped_results:
        if group and group[0].objective is not None:
            mol = group[0]
            total_mols += 1

            val_ = mol.objective if mol.objective is not None else float("-inf")
            if val_ > float("-inf"):
                scores.append(val_)

            if check_success and mol.smiles_string:
                if objective_evaluator.kinase_mpo_objective.is_successful(mol.smiles_string):
                    success_count += 1

    if not scores:
        return float("-inf"), 0.0

    mean_val_score = np.mean(scores)
    val_success_rate = (success_count / total_mols) if total_mols > 0 else 0.0

    print(f"[Val] Mean Uniform Weighted Score: {mean_val_score:.4f} | Success Rate: {val_success_rate:.2%}")
    return mean_val_score, val_success_rate

def validate_supervised(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Validates in supervised mode. Returns (mean_score, success_rate, []).
    """
    config = _make_eval_config(config_orig, beam_width=1, num_keep=1000)
    validitation_prompts = _load_scaffolds(config, 'validation_scaffolds_path', eval_type)

    if not validitation_prompts:
        print(f"[{eval_type}] Warning: No scaffolds found. Skipping evaluation.")
        return {}, ["No scaffolds found"]

    print(f"[{eval_type}] Found {len(validitation_prompts)} scaffolds. Processing one by one...")

    dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    scores = []
    success_count = 0
    total_mols = 0

    # Generate K candidates for this SINGLE prompt
    grouped_results = [dataset.generate_dataset(
        network_weights=weights,
        memory_aggressive=False,
        prompts=validitation_prompts,
        return_raw_trajectories=True,
        mode="eval"
    )]

    smiles = list(grouped_results[0]['top_20_molecules'][0].keys())

    for idx, mol in tqdm(enumerate(smiles), total=len(grouped_results), desc=f"Evaluating ({eval_type})"):


        # Check if we should calculate success rate (Only available for Kinase MPO)
        check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator,
                                                                            'kinase_mpo_objective')
        if check_success and mol:
            # Access the underlying KinaseMPOObjective instance directly
            if objective_evaluator.kinase_mpo_objective.is_successful(mol):
                success_count += 1

            scores.append(grouped_results[0]["top_20_molecules"][0][mol])

        total_mols += 1

    del grouped_results

    # Final Aggregation
    metrics_out = {
        f"{eval_type}_success_rate": success_count/total_mols,
        f"{eval_type}_mean_top1_obj": np.mean(scores),
    }

    print("=" * 30)
    print(f"EVALUATION REPORT ({eval_type})")
    print(f"Success Rate: {metrics_out[f'{eval_type}_success_rate'] * 100:.2f}%")
    print(f"Mean Top-1: {metrics_out[f'{eval_type}_mean_top1_obj']:.4f}")
    print("=" * 30)

    return np.mean(scores), success_count/total_mols, []


def train_for_one_epoch_supervised(epoch: int,
                                   config: MoleculeConfig,
                                   network: MoleculeTransformer,
                                   network_weights: dict,
                                   optimizer: torch.optim.Optimizer,
                                   objective_evaluator: MoleculeObjectiveEvaluator,
                                   best_objective: float,
                                   oracle_tracker_):
    """
    Original supervised fine-tuning path (dataset generation + cross-entropy on heads).
    """
    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator,
        oracle_tracker=oracle_tracker_
    )
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights,
        best_objective=best_objective,
        memory_aggressive=False,
        mode="train"
    )
    print("Generated molecules")
    print(f"Mean obj. over fresh best mols: {metrics['mean_best_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best mols: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
    print(f"Mean obj. over all time top 20 mols: {metrics['mean_top_20_obj']:.3f}")
    print(f"All time best mol: {list(metrics['top_20_molecules'][0].values())[0]:.3f}")
    torch.cuda.empty_cache()
    time.sleep(1)
    print("---- Loading dataset")
    dataset = RandomMoleculeDataset(config,
                                    config.gumbeldore_config["destination_path"],
                                    batch_size=config.batch_size_training,
                                    custom_num_batches=config.num_batches_per_epoch)

    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=config.num_dataloader_workers,
                            pin_memory=True,
                            persistent_workers=True)

    # Train for one epoch
    network.train()

    # freeze layers except the last (original behavior)

    if config.freeze_all_except_final_layer:
        for parameter in network.parameters():
            parameter.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

        # Apply feasibility masks (True = infeasible)
        logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
        logits_one[input_data["feasibility_mask_level_one"]] = float("-inf")
        logits_two[input_data["feasibility_mask_level_two"]] = float("-inf")

        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
        loss_zero = criterion(logits_zero, target_zero)
        loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
        loss_one = criterion(logits_one, target_one)
        loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one
        loss_two = criterion(logits_two, target_two)
        loss_two = torch.tensor(0.) if torch.isnan(loss_two) else loss_two
        loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches

    top_20_molecules = metrics["top_20_molecules"]

    # Detailed logging kinase MPO
    if config.objective_type == 'kinase_mpo':
        gsk3b_scores = []
        jnk3_scores = []
        qed_scores = []
        sa_scores = []

        for smiles in list(top_20_molecules[0].keys()):
            individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(smiles)
            gsk3b_scores.append(individual_scores.get('GSK3B'))
            jnk3_scores.append(individual_scores.get('JNK3'))
            qed_scores.append(individual_scores.get('QED'))
            sa_scores.append(individual_scores.get('SA'))

        metrics["gsk3b_scores"] = np.mean(gsk3b_scores).item()
        metrics["jnk3_scores"] = np.mean(jnk3_scores).item()
        metrics["qed_scores"] = np.mean(qed_scores).item()
        metrics["sa_scores"] = np.mean(sa_scores).item()

    del metrics["top_20_molecules"]

    # Retrieve the global count to log it
    current_oracle_count = 0
    if oracle_tracker_ is not None:
        current_oracle_count = get_oracle_count(oracle_tracker_, config)
        metrics["num_unique_oracle_calls"] = current_oracle_count

    return metrics, top_20_molecules

def train_for_one_epoch_rl(epoch: int,
                           config: MoleculeConfig,
                           network: MoleculeTransformer,
                           network_weights: dict,
                           optimizer: torch.optim.Optimizer,
                           objective_evaluator: MoleculeObjectiveEvaluator,
                           gumbeldore_dataset: GumbeldoreDataset,
                           novelty_memory: Optional[dict] = None,
                           oracle_tracker_=None):
    """
    RL fine-tuning epoch:
      1. Generate trajectories (terminated molecules) with current policy.
      2. Run policy gradient update via dr_grpo_update.
      3. Produce logging artifacts similar in spirit to supervised path.
    """
    print(f"[RL] Generating trajectories (epoch {epoch + 1})...")
    # gumbeldore_dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)

    if config.prodrug_mode:
        # Use training set
        current_prompts = config.prodrug_parents_train
    else:
        current_prompts = None  # Let generate_dataset use defaults

    # Return raw terminated trajectories (list of MoleculeDesign)
    trajectories = gumbeldore_dataset.generate_dataset(
        network_weights=network_weights,
        best_objective=None,
        memory_aggressive=False,
        prompts=current_prompts,
        mode="train"
    )

    if not trajectories or not any(trajectories):
        print("[RL] WARNING: No trajectories generated this epoch. Skipping update.")
        return {
            "num_trajectories": 0,
            "policy_loss": 0.0,
            "baseline": 0.0,
            "mean_reward": float("-inf"),
            "best_reward": float("-inf"),
            "mean_advantage": 0.0,
            "std_advantage": 0.0,
            "fraction_pos_adv": 0.0
        }, ["No molecules"]

    # Freeze backbone (match supervised style)
    if config.freeze_all_except_final_layer:
        for p in network.parameters():
            p.requires_grad = False
        network.virtual_atom_linear.weight.requires_grad = True
        network.virtual_atom_linear.bias.requires_grad = True
        network.bond_atom_linear.weight.requires_grad = True
        network.bond_atom_linear.bias.requires_grad = True

    network.train()

    print("training ...")
    metrics = dr_grpo_update(
        model=network,
        optimizer=optimizer,
        designs_groups=trajectories,
        config=config,
        device=torch.device(config.training_device),
        logger=None,
        novelty_memory=novelty_memory
    )
    print("dr GRPO update done.")
    metrics["best_gen_obj"] = metrics.get("best_objective", float("-inf"))
    metrics["mean_best_gen_obj"] = metrics.get("mean_reward", float("-inf"))
    metrics.setdefault("loss_level_zero", 0.0)
    metrics.setdefault("loss_level_one", 0.0)
    metrics.setdefault("loss_level_two", 0.0)

    # -------------------------------------------------------------------------
    # RAW BIOLOGY LOGGING
    # -------------------------------------------------------------------------
    mol_map = {}
    all_unweighted_scores = []  # NEW: Track every molecule

    for group in trajectories:
        for m in group:
            if m.objective is None or not m.smiles_string:
                continue

            unweighted = get_unweighted_score(m, config.objective_type)
            all_unweighted_scores.append(unweighted)

            if m.smiles_string not in mol_map or mol_map[m.smiles_string]["unweighted"] < unweighted:
                mol_map[m.smiles_string] = {
                    "smiles": m.smiles_string,
                    "obj": m.objective,
                    "unweighted": unweighted,
                    "aux": getattr(m, 'aux_metrics', {})
                }

    if all_unweighted_scores:
        metrics["mean_unweighted_all"] = np.mean(all_unweighted_scores)
    else:
        metrics["mean_unweighted_all"] = float("-inf")

    unique_mols = list(mol_map.values())
    # Sort strictly by the UNWEIGHTED BIOLOGY SUM
    unique_mols.sort(key=lambda x: x["unweighted"], reverse=True)
    top20 = unique_mols[:20]

    if top20:
        metrics["mean_top_20_unweighted"] = np.mean([entry["unweighted"] for entry in top20])
        metrics["best_gen_unweighted"] = top20[0]["unweighted"]
    else:
        metrics["mean_top_20_unweighted"] = float("-inf")
        metrics["best_gen_unweighted"] = float("-inf")

    top_20_text_lines = []
    for i, entry in enumerate(top20):
        aux = entry.get("aux", {})
        raw_str = format_aux_metrics(aux, config.objective_type)

        line = f"{i + 1:02d}: {entry['smiles']} | Unweighted Sum={entry['unweighted']:.4f} | {raw_str} (Weighted_Reward={entry['obj']:.4f})"
        top_20_text_lines.append(line)

    if not top20:
        top_20_text_lines.append("No terminated molecules")

    # Retrieve the global count to log it
    current_oracle_count = 0
    if oracle_tracker_ is not None:
        current_oracle_count = get_oracle_count(oracle_tracker_, config)
        metrics["num_unique_oracle_calls"] = current_oracle_count

    # Detailed logging kinase MPO
    if config.objective_type == 'kinase_mpo':
        gsk3b_scores = []
        jnk3_scores = []
        qed_scores = []
        sa_scores = []

        for entry in unique_mols:
            individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(entry['smiles'])
            gsk3b_scores.append(individual_scores.get('GSK3B'))
            jnk3_scores.append(individual_scores.get('JNK3'))
            qed_scores.append(individual_scores.get('QED'))
            sa_scores.append(individual_scores.get('SA'))

        metrics["gsk3b_scores"] = np.mean(gsk3b_scores).item()
        metrics["jnk3_scores"] = np.mean(jnk3_scores).item()
        metrics["qed_scores"] = np.mean(qed_scores).item()
        metrics["sa_scores"] = np.mean(sa_scores).item()

    return metrics, top20, top_20_text_lines


def evaluate(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    config = _make_eval_config(config_orig)
    test_prompts = _load_scaffolds(config, 'evaluation_scaffolds_path', eval_type)

    if not test_prompts:
        return {}, ["No scaffolds found"]

    eval_config = copy.deepcopy(config)
    eval_config.gumbeldore_config["destination_path"] = None

    os.makedirs(config.results_path, exist_ok=True)
    csv_path = os.path.join(config.results_path, f"{eval_type}_detailed_logs.csv")

    dataset = GumbeldoreDataset(config=eval_config, objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    # --- THE FIX: Define the Lambda Sweep based on dimensions ---
    num_obj = getattr(config, 'num_objectives', 1)
    if num_obj == 2:
        test_lambdas = [
            np.array([1.0, 0.0]),
            np.array([0.75, 0.25]),
            np.array([0.5, 0.5]),
            np.array([0.25, 0.75]),
            np.array([0.0, 1.0])
        ]
    elif num_obj == 3:
        test_lambdas = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.5, 0.5, 0.0]),
            np.array([0.5, 0.0, 0.5]),
            np.array([0.0, 0.5, 0.5]),
            np.array([0.33, 0.33, 0.34])
        ]
    else:
        test_lambdas = [None]  # Backward compatibility for Single Objective

    with open(csv_path, mode='w', newline='') as csv_file:
        # Added 'target_lambda' to the CSV headers
        fieldnames = ['scaffold_idx', 'prompt_smiles', 'target_lambda', 'generated_smiles', 'weighted_objective',
                      'unweighted_sum', 'is_successful', 'gsk3b', 'jnk3', 'herg', 'bbb', 'qed', 'sa']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        scores = []
        success_count = 0
        total_mols = 0

        for idx, prompt in tqdm(enumerate(test_prompts), total=len(test_prompts), desc=f"Evaluating ({eval_type})"):
            # --- THE FIX: Loop over every lambda in the sweep for this scaffold ---
            for current_lambda in test_lambdas:
                smi = ""
                obj_val = 0.0
                is_successful = False
                gsk, jnk, herg, bbb, qed, sa = 0, 0, 0, 0, 0, 0
                val_ = 0.0

                grouped_results = dataset.generate_dataset(
                    network_weights=weights,
                    memory_aggressive=False,
                    prompts=[prompt],
                    return_raw_trajectories=True,
                    mode="eval",
                    fixed_lambda=current_lambda  # Inject the specific sweep target
                )

                if not grouped_results or not grouped_results[0]:
                    writer.writerow({
                        'scaffold_idx': idx,
                        'prompt_smiles': prompt,
                        'target_lambda': str(current_lambda.tolist()) if current_lambda is not None else "N/A",
                        'generated_smiles': "GENERATION_FAILED",
                        'weighted_objective': 0.0,
                        'unweighted_sum': 0.0,
                        'is_successful': False,
                        'gsk3b': 0,
                        'jnk3': 0,
                        'herg': 0,
                        'bbb': 0,
                        'qed': 0,
                        'sa': 0
                    })
                    continue

                group = grouped_results[0]
                check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator, 'kinase_mpo_objective')

                ordered_group = sorted(
                    group,
                    key=lambda x: x.objective if x.objective is not None else float("-inf"),
                    reverse=True
                )

                best_mol_to_log = ordered_group[0]  # Fallback to the top-scoring failed molecule
                best_is_successful = False

                for mol in ordered_group:
                    is_successful = False
                    obj_val = mol.objective if mol.objective is not None else float("-inf")
                    smi = mol.smiles_string if mol.smiles_string else ""

                    if check_success and mol:
                        if objective_evaluator.kinase_mpo_objective.is_successful(mol.smiles_string):
                            is_successful = True
                            success_count += 1

                    if not is_successful and config.objective_type == 'kinase_mpo':
                        continue

                    best_mol_to_log = mol
                    best_is_successful = is_successful
                    break

                val_ = best_mol_to_log.objective if best_mol_to_log.objective is not None else float("-inf")
                smi = best_mol_to_log.smiles_string if best_mol_to_log.smiles_string else ""

                if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                    individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(smi)
                    gsk = individual_scores.get('GSK3B')
                    jnk = individual_scores.get('JNK3')
                    qed = individual_scores.get('QED')
                    sa = individual_scores.get('SA')

                # For MOO tasks, pull aux metrics
                if hasattr(best_mol_to_log, 'aux_metrics') and best_mol_to_log.aux_metrics:
                    gsk = best_mol_to_log.aux_metrics.get('gsk3b', gsk)
                    jnk = best_mol_to_log.aux_metrics.get('jnk3', jnk)
                    herg = best_mol_to_log.aux_metrics.get('herg', herg)
                    bbb = best_mol_to_log.aux_metrics.get('bbb', bbb)

                # Always write one row per scaffold/lambda pair
                writer.writerow({
                    'scaffold_idx': idx,
                    'prompt_smiles': prompt,
                    'target_lambda': str(current_lambda.tolist()) if current_lambda is not None else "N/A",
                    'generated_smiles': smi,
                    'weighted_objective': val_ if val_ > float("-inf") else 0.0,
                    'unweighted_sum': get_unweighted_score(best_mol_to_log, config.objective_type),
                    'is_successful': best_is_successful,
                    'gsk3b': gsk,
                    'jnk3': jnk,
                    'herg': herg,
                    'bbb': bbb,
                    'qed': qed,
                    'sa': sa
                })

                total_mols += 1
                scores.append(val_)
                del grouped_results

    metrics_out = {
        f"{eval_type}_success_rate": success_count / total_mols if total_mols > 0 else 0.0,
        f"{eval_type}_mean_top1_obj": np.mean(scores) if scores else float("-inf"),
    }

    print("=" * 30)
    print(f"EVALUATION SWEEP REPORT ({eval_type})")
    print(f"Detailed logs saved to: {csv_path}")
    print("=" * 30)

    return metrics_out




def evaluate_supervised(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Evaluates on test scaffolds one-by-one. Supervised/TASAR version.
    Saves a detailed CSV of every generated molecule.
    """
    config = _make_eval_config(config_orig, beam_width=getattr(config_orig, 'fixed_test_beam_width', 1))
    test_prompts = _load_scaffolds(config, 'evaluation_scaffolds_path', eval_type)

    if not test_prompts:
        print(f"[{eval_type}] Warning: No scaffolds found. Skipping evaluation.")
        return {}, ["No scaffolds found"]

    print(f"[{eval_type}] Found {len(test_prompts)} scaffolds. Processing one by one...")

    eval_config = copy.deepcopy(config)
    eval_config.gumbeldore_config["destination_path"] = None

    os.makedirs(config.results_path, exist_ok=True)
    csv_path = os.path.join(config.results_path, f"{eval_type}_detailed_logs.csv")
    print(f"[{eval_type}] saving detailed logs to: {csv_path}")

    if getattr(eval_config, 'fixed_test_beam_width', None) is not None:
        eval_config.gumbeldore_config["beam_width"] = eval_config.fixed_test_beam_width

    print(f"[{eval_type}] using beam width:", eval_config.gumbeldore_config["beam_width"])

    dataset = GumbeldoreDataset(config=eval_config, objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    # 3. Open CSV and Start Loop
    # We use 'w' mode to overwrite if restarting, or 'a' could be used if careful.
    with open(csv_path, mode='w', newline='') as csv_file:
        print("Opened CSV for writing:", csv_path)
        fieldnames = ['scaffold_idx', 'prompt_smiles', 'generated_smiles', 'objective_score', 'is_successful',
                      'gsk3b', 'jnk3', 'qed', 'sa']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        scores = []
        success_count = 0
        total_mols = 0

        for idx, prompt in tqdm(enumerate(test_prompts), total=len(test_prompts), desc=f"Evaluating ({eval_type})"):

            is_successful = False  # Reset for each prompt
            if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                gsk = None
                jnk = None
                qed = None
                sa = None

            # Generate K candidates for this SINGLE prompt
            grouped_results = [dataset.generate_dataset(
                network_weights=weights,
                memory_aggressive=False,
                prompts=[prompt],
                return_raw_trajectories=True,
                mode="eval"
            )]

            # Check if generation returned anything
            if not grouped_results:
                # Log failure in CSV
                writer.writerow({
                    'scaffold_idx': idx,
                    'prompt_smiles': prompt,
                    'generated_smiles': "GENERATION_FAILED",
                    'objective_score': 0.0,
                    'is_successful': False,
                    'gsk3b': 0,
                    'jnk3': 0,
                    'qed': 0,
                    'sa': 0
                })
                continue


            # Check if we should calculate success rate (Only available for Kinase MPO)
            check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator,
                                                                                'kinase_mpo_objective')

            best_mol = list(grouped_results[0]["top_20_molecules"][0].keys())[0]
            if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(best_mol)
                gsk = individual_scores.get('GSK3B')
                jnk = individual_scores.get('JNK3')
                qed = individual_scores.get('QED')
                sa = individual_scores.get('SA')


            if grouped_results and grouped_results[0]["best_gen_obj"] is not None:
                for mol in list(grouped_results[0]["top_20_molecules"][0].keys()):  # loop over all the beam leaves

                    # --- Success Rate Handling ---
                    if check_success and mol:
                        # Access the underlying KinaseMPOObjective instance directly
                        if objective_evaluator.kinase_mpo_objective.is_successful(mol):
                            is_successful = True
                            success_count += 1

                        # --- Score Handling ---
                        val_ = grouped_results[0]["top_20_molecules"][0][mol]
                        if val_ == float("-inf"):
                            val_ = 0.0
                        scores.append(val_)

                        if not is_successful:
                            continue

                        # individual metrics for each scaffold
                        if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                            individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(mol)
                            gsk = individual_scores.get('GSK3B')
                            jnk = individual_scores.get('JNK3')
                            qed = individual_scores.get('QED')
                            sa = individual_scores.get('SA')

                        break

            total_mols += 1


            if not scores:
                print("[Val] No valid molecules generated.")
                return float("-inf"), 0.0


            # --- Logging ---

            # Write EVERY beam to the CSV
            writer.writerow({
                'scaffold_idx': idx,
                'prompt_smiles': prompt,
                'generated_smiles': mol,
                'objective_score': val_,
                'is_successful': is_successful,
                'gsk3b': gsk,
                'jnk3': jnk,
                'qed': qed,
                'sa': sa
            })


            # Memory cleanup
            del grouped_results


    # Final Aggregation
    metrics_out = {
        f"{eval_type}_success_rate": success_count/total_mols,
        f"{eval_type}_mean_top1_obj": np.mean(scores),
    }

    print("=" * 30)
    print(f"EVALUATION REPORT ({eval_type})")
    print(f"Detailed logs saved to: {csv_path}")
    print(f"Success Rate: {metrics_out[f'{eval_type}_success_rate'] * 100:.2f}%")
    print(f"Mean Top-1: {metrics_out[f'{eval_type}_mean_top1_obj']:.4f}")
    print("=" * 30)

    return metrics_out




def _safe_filename(s: str) -> str:
    keep = "-_."
    return "".join(c if c.isalnum() or c in keep else "_" for c in s)


def _plot_prodrug_bbb_summary(parent_records, all_beam_records, plots_dir):
    """Generate publication-style plots for the prodrug-BBB evaluation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not parent_records:
        return

    parent_bbb = np.array([r['parent_bbb'] for r in parent_records])
    best_bbb = np.array([r['best_bbb'] for r in parent_records])
    parent_total = np.array([r['parent_total'] for r in parent_records])
    best_total = np.array([r['best_total'] for r in parent_records])
    parent_qed = np.array([r['parent_qed'] for r in parent_records])
    best_qed = np.array([r['best_qed'] for r in parent_records])
    parent_mw = np.array([r['parent_mw'] for r in parent_records])
    best_mw = np.array([r['best_mw'] for r in parent_records])
    names = [r['parent_name'] for r in parent_records]

    # 1. BBB before/after scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(parent_bbb, best_bbb, c='steelblue', s=60, edgecolor='k', alpha=0.85)
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label='y = x')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel('Parent BBB probability')
    ax.set_ylabel('Best generated BBB probability')
    ax.set_title('BBB before vs. after (per parent)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'bbb_before_after_scatter.png'), dpi=150)
    plt.close(fig)

    # 2. Total reward before/after scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(parent_total, best_total, c='seagreen', s=60, edgecolor='k', alpha=0.85)
    lo = float(min(parent_total.min(), best_total.min())) - 0.05
    hi = float(max(parent_total.max(), best_total.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1, label='y = x')
    ax.set_xlabel('Parent total reward')
    ax.set_ylabel('Best generated total reward')
    ax.set_title('BBB Objective total reward: parent vs. best')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'total_reward_before_after.png'), dpi=150)
    plt.close(fig)

    # 3. Improvement per parent (sorted bar chart)
    delta = best_total - parent_total
    order = np.argsort(delta)
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(names)), 6))
    ax.bar(range(len(delta)), delta[order],
           color=['firebrick' if d < 0 else 'steelblue' for d in delta[order]])
    ax.set_xticks(range(len(delta)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=8)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_ylabel(r'$\Delta$ total reward (best - parent)')
    ax.set_title('Improvement per parent')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'improvement_per_parent.png'), dpi=150)
    plt.close(fig)

    # 4. Per-parent beam distribution boxplot (with parent score overlay)
    by_parent = {}
    for rec in all_beam_records:
        by_parent.setdefault(rec['parent_idx'], []).append(rec['g_total'] if rec['g_total'] is not None else 0.0)
    parent_idx_sorted = sorted(by_parent.keys())
    if parent_idx_sorted:
        data = [by_parent[i] for i in parent_idx_sorted]
        idx_to_record = {r['parent_idx']: r for r in parent_records}
        name_sorted = [idx_to_record[i]['parent_name'] for i in parent_idx_sorted]
        parent_overlay = [idx_to_record[i]['parent_total'] for i in parent_idx_sorted]
        fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(data)), 6))
        ax.boxplot(data, showfliers=False)
        ax.set_xticks(range(1, len(name_sorted) + 1))
        ax.set_xticklabels(name_sorted, rotation=90, fontsize=8)
        ax.scatter(range(1, len(parent_idx_sorted) + 1), parent_overlay,
                   c='red', s=24, label='parent', zorder=3)
        ax.set_ylabel('Total reward (BBBObjective)')
        ax.set_title('Per-parent beam distribution (32 beams) vs. parent score')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, 'beam_total_reward_box.png'), dpi=150)
        plt.close(fig)

    # 5. QED before/after scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(parent_qed, best_qed, c='goldenrod', s=60, edgecolor='k', alpha=0.85)
    ax.plot([0, 1], [0, 1], 'r--', lw=1, label='y = x')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1])
    ax.set_xlabel('Parent QED')
    ax.set_ylabel('Best generated QED')
    ax.set_title('QED before vs. after (per parent)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'qed_before_after_scatter.png'), dpi=150)
    plt.close(fig)

    # 6. MW before/after scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(parent_mw, best_mw, c='purple', s=60, edgecolor='k', alpha=0.85)
    lim_hi = float(max(parent_mw.max(), best_mw.max())) + 50.0
    ax.plot([0, lim_hi], [0, lim_hi], 'r--', lw=1, label='y = x')
    ax.set_xlabel('Parent molecular weight (Da)')
    ax.set_ylabel('Best generated molecular weight (Da)')
    ax.set_title('Molecular weight before vs. after')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'mw_before_after_scatter.png'), dpi=150)
    plt.close(fig)

    # 7. Histogram of delta BBB
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(best_bbb - parent_bbb, bins=20, color='teal', edgecolor='k')
    ax.axvline(0, color='k', lw=1)
    ax.set_xlabel(r'$\Delta$ BBB probability (best - parent)')
    ax.set_ylabel('Number of parents')
    ax.set_title('Distribution of BBB improvement')
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'delta_bbb_histogram.png'), dpi=150)
    plt.close(fig)

    # 8. Stacked component comparison (parent vs best) for each parent
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(names)), 6))
    x = np.arange(len(names))
    width = 0.4
    ax.bar(x - width / 2, parent_bbb, width, color='lightcoral', label='Parent BBB prob')
    ax.bar(x + width / 2, best_bbb, width, color='steelblue', label='Best gen BBB prob')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90, fontsize=8)
    ax.set_ylim([0, 1])
    ax.set_ylabel('BBB probability')
    ax.set_title('Per-parent BBB probability: parent vs. best generated')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, 'per_parent_bbb_bars.png'), dpi=150)
    plt.close(fig)


def evaluate_prodrug_bbb(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
                         objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Detailed inference-time evaluation for the prodrug_bbb objective.

    For each parent molecule loaded from prodrug_test.py:
      - Score the parent itself with the full BBBObjective decomposition
        (BBB probability, QED, MW, gates, total reward).
      - Run deterministic beam search (default beam_width = 32) to generate
        prodrug candidates conditioned on the parent SMILES.
      - Score every beam with the same BBBObjective decomposition.
      - Pick the best beam (highest total_reward) and persist:
          * prodrug_bbb_eval/test_all_beams.csv          -- every beam, all sub-scores
          * prodrug_bbb_eval/test_best_per_parent.csv    -- the selected best per parent
          * prodrug_bbb_eval/test_parents_smiles.txt     -- input parents (name, SMILES)
          * prodrug_bbb_eval/test_generated_best_smiles.txt -- best-of-beam outputs
          * prodrug_bbb_eval/best_molecule_images/*.png  -- parent vs. best side-by-side
          * prodrug_bbb_eval/plots/*.png                 -- paper-ready summary plots
    """
    parents = get_prodrug_test_parents()  # List[(name, SMILES)]
    if not parents:
        print(f"[{eval_type}] No parents found in prodrug_test.py. Skipping.")
        return {}

    beam_width = int(getattr(config_orig, 'fixed_test_beam_width', 32) or 32)
    config = _make_eval_config(config_orig, beam_width=beam_width, num_keep=beam_width)
    config.gumbeldore_config["search_type"] = "beam_search"
    config.gumbeldore_config["beam_width"] = beam_width
    config.gumbeldore_config["num_trajectories_to_keep"] = beam_width
    config.gumbeldore_config["deterministic"] = True
    config.gumbeldore_config["destination_path"] = None

    out_dir = os.path.join(config.results_path, "prodrug_bbb_eval")
    images_dir = os.path.join(out_dir, "best_molecule_images")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    csv_all_path = os.path.join(out_dir, f"{eval_type}_all_beams.csv")
    csv_best_path = os.path.join(out_dir, f"{eval_type}_best_per_parent.csv")
    parents_smiles_path = os.path.join(out_dir, f"{eval_type}_parents_smiles.txt")
    generated_best_smiles_path = os.path.join(out_dir, f"{eval_type}_generated_best_smiles.txt")
    log_path = os.path.join(out_dir, f"{eval_type}_evaluation_log.txt")

    print("=" * 70)
    print(f"[Prodrug-BBB Eval] {len(parents)} parents from prodrug_test.py")
    print(f"[Prodrug-BBB Eval] Beam width = {beam_width} (deterministic beam search)")
    print(f"[Prodrug-BBB Eval] Outputs -> {out_dir}")
    print("=" * 70)

    from rdkit import Chem
    from rdkit.Chem import Draw

    bbb_obj = objective_evaluator.bbb_objective

    dataset = GumbeldoreDataset(config=config, objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    with open(parents_smiles_path, 'w') as f:
        for name, smi in parents:
            f.write(f"{name}\t{smi}\n")

    detail_fields = [
        'parent_idx', 'parent_name', 'parent_smiles',
        'parent_bbb_prob', 'parent_qed', 'parent_mw',
        'parent_qed_gate', 'parent_purity', 'parent_total_reward',
        'beam_idx', 'is_best',
        'generated_smiles',
        'gen_bbb_prob', 'gen_qed', 'gen_mw',
        'gen_qed_gate', 'gen_purity', 'gen_total_reward',
        'reward_bbb', 'reward_qed_gate', 'reward_purity',
        'd_bbb_prob', 'd_qed', 'd_mw', 'd_total_reward',
    ]
    best_fields = [
        'parent_idx', 'parent_name', 'parent_smiles',
        'parent_bbb_prob', 'parent_qed', 'parent_mw',
        'parent_qed_gate', 'parent_purity', 'parent_total_reward',
        'best_beam_idx',
        'generated_smiles',
        'gen_bbb_prob', 'gen_qed', 'gen_mw',
        'gen_qed_gate', 'gen_purity', 'gen_total_reward',
        'reward_bbb', 'reward_qed_gate', 'reward_purity',
        'd_bbb_prob', 'd_qed', 'd_mw', 'd_total_reward',
        'num_beams_generated', 'num_unique_smiles',
    ]

    parent_records = []
    generated_best_records = []
    all_beam_records = []
    log_lines = []

    csv_all_file = open(csv_all_path, mode='w', newline='')
    csv_best_file = open(csv_best_path, mode='w', newline='')
    writer_all = csv.DictWriter(csv_all_file, fieldnames=detail_fields)
    writer_best = csv.DictWriter(csv_best_file, fieldnames=best_fields)
    writer_all.writeheader()
    writer_best.writeheader()

    try:
        for p_idx, (parent_name, parent_smi) in tqdm(
                enumerate(parents), total=len(parents), desc=f"Evaluating ({eval_type})"
        ):
            parent_mol = Chem.MolFromSmiles(parent_smi)
            if parent_mol is None:
                msg = f"  [Skip] Could not parse parent {parent_name}: {parent_smi}"
                print(msg); log_lines.append(msg)
                continue

            # Note: scoring the parent against itself produces purity=0
            # (no junction bonds when generated == parent). Expected and correct
            # -- the parent is not a prodrug of itself. parent_total_reward
            # will therefore be ~0 for every parent; the meaningful comparison
            # in the paper is parent_bbb_prob vs gen_bbb_prob.
            parent_score = bbb_obj.calculate(parent_mol, parent_mol)
            p_metrics = parent_score['metrics']
            parent_bbb_prob = float(p_metrics['bbb_prob'])
            parent_qed = float(p_metrics['qed'])
            parent_mw = float(p_metrics['mw'])
            parent_qed_gate = float(p_metrics['qed_gate'])
            parent_purity = float(p_metrics['purity'])
            parent_total = float(parent_score['total_reward'])

            header = (
                f"\n[Parent {p_idx + 1:02d}/{len(parents)}] {parent_name}\n"
                f"   SMILES: {parent_smi}\n"
                f"   BBB={parent_bbb_prob:.4f}  QED={parent_qed:.4f}  MW={parent_mw:.2f}  "
                f"QED_gate={parent_qed_gate:.3f}  Purity={parent_purity:.3f}  "
                f"Total={parent_total:.4f}"
            )
            print(header); log_lines.append(header)

            grouped_results = dataset.generate_dataset(
                network_weights=weights,
                memory_aggressive=False,
                prompts=[parent_smi],
                return_raw_trajectories=True,
                mode="eval",
            )

            beams = grouped_results[0] if grouped_results else []
            if not beams:
                msg = f"   [Warn] No beams generated for {parent_name}."
                print(msg); log_lines.append(msg)
                writer_best.writerow({
                    'parent_idx': p_idx,
                    'parent_name': parent_name,
                    'parent_smiles': parent_smi,
                    'parent_bbb_prob': parent_bbb_prob,
                    'parent_qed': parent_qed,
                    'parent_mw': parent_mw,
                    'parent_qed_gate': parent_qed_gate,
                    'parent_purity': parent_purity,
                    'parent_total_reward': parent_total,
                    'best_beam_idx': -1,
                    'generated_smiles': "GENERATION_FAILED",
                    'gen_bbb_prob': 0.0,
                    'gen_qed': 0.0,
                    'gen_mw': 0.0,
                    'gen_qed_gate': 0.0,
                    'gen_purity': 0.0,
                    'gen_total_reward': 0.0,
                    'reward_bbb': 0.0,
                    'reward_qed_gate': 0.0,
                    'reward_purity': 0.0,
                    'd_bbb_prob': -parent_bbb_prob,
                    'd_qed': -parent_qed,
                    'd_mw': -parent_mw,
                    'd_total_reward': -parent_total,
                    'num_beams_generated': 0,
                    'num_unique_smiles': 0,
                })
                continue

            beam_records = []
            seen_smiles = set()
            for b_idx, mol in enumerate(beams):
                gen_smi = mol.smiles_string or ""
                if not gen_smi:
                    continue
                gen_rdkit = mol.rdkit_mol
                if gen_rdkit is None:
                    try:
                        gen_rdkit = Chem.MolFromSmiles(gen_smi)
                    except Exception:
                        gen_rdkit = None
                if gen_rdkit is None:
                    continue

                if hasattr(mol, 'aux_metrics') and mol.aux_metrics:
                    aux = mol.aux_metrics
                    g_bbb = float(aux.get('bbb_prob', 0.0))
                    g_qed = float(aux.get('qed', 0.0))
                    g_mw = float(aux.get('mw', 0.0))
                    g_qg = float(aux.get('qed_gate', 0.0))
                    g_pur = float(aux.get('purity', 0.0))
                    rew_bbb = float(aux.get('reward_bbb', g_bbb))
                    rew_qg = float(aux.get('reward_qed_gate', g_qg))
                    rew_pur = float(aux.get('reward_purity', g_pur))
                    g_total = float(mol.objective) if mol.objective is not None else (g_bbb * rew_qg * rew_pur)
                else:
                    score = bbb_obj.calculate(gen_rdkit, parent_mol)
                    aux = score['metrics']
                    g_bbb = float(aux['bbb_prob'])
                    g_qed = float(aux['qed'])
                    g_mw = float(aux['mw'])
                    g_qg = float(aux['qed_gate'])
                    g_pur = float(aux['purity'])
                    rew_bbb = float(score['reward_bbb'])
                    rew_qg = float(score['reward_qed_gate'])
                    rew_pur = float(score['reward_purity'])
                    g_total = float(score['total_reward'])

                seen_smiles.add(gen_smi)
                beam_records.append({
                    'beam_idx': b_idx,
                    'gen_smiles': gen_smi,
                    'g_bbb': g_bbb,
                    'g_qed': g_qed,
                    'g_mw': g_mw,
                    'g_qg': g_qg,
                    'g_pur': g_pur,
                    'rew_bbb': rew_bbb,
                    'rew_qg': rew_qg,
                    'rew_pur': rew_pur,
                    'g_total': g_total,
                    'rdkit_mol': gen_rdkit,
                })

            if not beam_records:
                msg = f"   [Warn] No valid beams parsed for {parent_name}."
                print(msg); log_lines.append(msg)
                continue

            beam_records.sort(key=lambda x: x['g_total'], reverse=True)
            best = beam_records[0]
            best_idx = best['beam_idx']

            best_msg = (
                f"   {len(beam_records)} valid beams ({len(seen_smiles)} unique). "
                f"Best beam idx={best_idx} | BBB={best['g_bbb']:.4f} QED={best['g_qed']:.4f} "
                f"MW={best['g_mw']:.2f} Purity={best['g_pur']:.2f} Total={best['g_total']:.4f} "
                f"(\u0394BBB={best['g_bbb'] - parent_bbb_prob:+.4f})"
            )
            print(best_msg); log_lines.append(best_msg)

            pure_count = sum(1 for r in beam_records if r['g_pur'] == 1.0)
            partial_count = sum(1 for r in beam_records if 0 < r['g_pur'] < 1.0)
            log_lines.append(
                f"   Pure prodrug beams: {pure_count}/{len(beam_records)}  |  "
                f"Partial: {partial_count}/{len(beam_records)}"
            )
            log_lines.append("   Per-beam SMILES (sorted by total reward):")
            for rank, rec in enumerate(beam_records):
                log_lines.append(
                    f"     {rank + 1:02d}. beam_idx={rec['beam_idx']:02d}  total={rec['g_total']:.4f}  "
                    f"BBB={rec['g_bbb']:.4f}  QED={rec['g_qed']:.4f}  MW={rec['g_mw']:.2f}  "
                    f"Purity={rec['g_pur']:.2f}  SMILES={rec['gen_smiles']}"
                )

            for rec in beam_records:
                row = {
                    'parent_idx': p_idx,
                    'parent_name': parent_name,
                    'parent_smiles': parent_smi,
                    'parent_bbb_prob': parent_bbb_prob,
                    'parent_qed': parent_qed,
                    'parent_mw': parent_mw,
                    'parent_qed_gate': parent_qed_gate,
                    'parent_purity': parent_purity,
                    'parent_total_reward': parent_total,
                    'beam_idx': rec['beam_idx'],
                    'is_best': rec['beam_idx'] == best_idx,
                    'generated_smiles': rec['gen_smiles'],
                    'gen_bbb_prob': rec['g_bbb'],
                    'gen_qed': rec['g_qed'],
                    'gen_mw': rec['g_mw'],
                    'gen_qed_gate': rec['g_qg'],
                    'gen_purity': rec['g_pur'],
                    'gen_total_reward': rec['g_total'],
                    'reward_bbb': rec['rew_bbb'],
                    'reward_qed_gate': rec['rew_qg'],
                    'reward_purity': rec['rew_pur'],
                    'd_bbb_prob': rec['g_bbb'] - parent_bbb_prob,
                    'd_qed': rec['g_qed'] - parent_qed,
                    'd_mw': rec['g_mw'] - parent_mw,
                    'd_total_reward': rec['g_total'] - parent_total,
                }
                writer_all.writerow(row)
                all_beam_records.append({
                    'parent_idx': p_idx,
                    'parent_name': parent_name,
                    'parent_total': parent_total,
                    'parent_bbb': parent_bbb_prob,
                    'g_total': rec['g_total'],
                    'g_bbb': rec['g_bbb'],
                    'g_qed': rec['g_qed'],
                    'g_mw': rec['g_mw'],
                    'g_pur': rec['g_pur'],
                })

            writer_best.writerow({
                'parent_idx': p_idx,
                'parent_name': parent_name,
                'parent_smiles': parent_smi,
                'parent_bbb_prob': parent_bbb_prob,
                'parent_qed': parent_qed,
                'parent_mw': parent_mw,
                'parent_qed_gate': parent_qed_gate,
                'parent_purity': parent_purity,
                'parent_total_reward': parent_total,
                'best_beam_idx': best_idx,
                'generated_smiles': best['gen_smiles'],
                'gen_bbb_prob': best['g_bbb'],
                'gen_qed': best['g_qed'],
                'gen_mw': best['g_mw'],
                'gen_qed_gate': best['g_qg'],
                'gen_purity': best['g_pur'],
                'gen_total_reward': best['g_total'],
                'reward_bbb': best['rew_bbb'],
                'reward_qed_gate': best['rew_qg'],
                'reward_purity': best['rew_pur'],
                'd_bbb_prob': best['g_bbb'] - parent_bbb_prob,
                'd_qed': best['g_qed'] - parent_qed,
                'd_mw': best['g_mw'] - parent_mw,
                'd_total_reward': best['g_total'] - parent_total,
                'num_beams_generated': len(beams),
                'num_unique_smiles': len(seen_smiles),
            })

            parent_records.append({
                'parent_idx': p_idx,
                'parent_name': parent_name,
                'parent_smiles': parent_smi,
                'parent_total': parent_total,
                'parent_bbb': parent_bbb_prob,
                'parent_qed': parent_qed,
                'parent_mw': parent_mw,
                'best_total': best['g_total'],
                'best_bbb': best['g_bbb'],
                'best_qed': best['g_qed'],
                'best_mw': best['g_mw'],
                'best_purity': best['g_pur'],
                'best_smiles': best['gen_smiles'],
            })
            generated_best_records.append((parent_name, best['gen_smiles']))

            try:
                img = Draw.MolsToGridImage(
                    [parent_mol, best['rdkit_mol']],
                    molsPerRow=2,
                    subImgSize=(420, 420),
                    legends=[
                        f"Parent: {parent_name}\nBBB={parent_bbb_prob:.3f}  QED={parent_qed:.3f}  MW={parent_mw:.1f}",
                        f"Best gen (beam {best_idx})\nBBB={best['g_bbb']:.3f}  QED={best['g_qed']:.3f}  MW={best['g_mw']:.1f}\nPurity={best['g_pur']:.2f}  Total={best['g_total']:.3f}",
                    ],
                )
                img_path = os.path.join(images_dir, f"{p_idx:02d}_{_safe_filename(parent_name)}.png")
                img.save(img_path)
            except Exception as e:
                print(f"   [Warn] Could not draw {parent_name}: {e}")

            del grouped_results
    finally:
        csv_all_file.close()
        csv_best_file.close()

    with open(generated_best_smiles_path, 'w', encoding='utf-8') as f:
        for name, smi in generated_best_records:
            f.write(f"{name}\t{smi}\n")

    safe_log_lines = [str(x) if x is not None else "" for x in log_lines]
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(safe_log_lines))

    try:
        _plot_prodrug_bbb_summary(parent_records, all_beam_records, plots_dir)
    except Exception as e:
        print(f"[Plots] Warning: {e}")

    if not parent_records:
        print(f"[{eval_type}] No parents successfully evaluated.")
        return {}

    parent_totals = np.array([r['parent_total'] for r in parent_records])
    best_totals = np.array([r['best_total'] for r in parent_records])
    parent_bbbs = np.array([r['parent_bbb'] for r in parent_records])
    best_bbbs = np.array([r['best_bbb'] for r in parent_records])
    parent_qeds = np.array([r['parent_qed'] for r in parent_records])
    best_qeds = np.array([r['best_qed'] for r in parent_records])
    best_purities = np.array([r['best_purity'] for r in parent_records])
    improved_total = int((best_totals > parent_totals).sum())
    improved_bbb = int((best_bbbs > parent_bbbs).sum())
    pure_best = int((best_purities == 1.0).sum())
    partial_best = int(((best_purities > 0) & (best_purities < 1.0)).sum())

    print("=" * 70)
    print(f"PRODRUG-BBB EVAL SUMMARY ({eval_type})")
    print(f"Parents evaluated:           {len(parent_records)}")
    print(f"Mean parent total reward:    {parent_totals.mean():.4f}")
    print(f"Mean best-gen total reward:  {best_totals.mean():.4f}")
    print(f"Mean parent BBB prob:        {parent_bbbs.mean():.4f}")
    print(f"Mean best-gen BBB prob:      {best_bbbs.mean():.4f}")
    print(f"Mean parent QED:             {parent_qeds.mean():.4f}")
    print(f"Mean best-gen QED:           {best_qeds.mean():.4f}")
    print(f"Mean best-gen purity:        {best_purities.mean():.4f}")
    print(f"Parents with improved total: {improved_total}/{len(parent_records)}")
    print(f"Parents with improved BBB:   {improved_bbb}/{len(parent_records)}")
    print(f"Parents with PURE best:      {pure_best}/{len(parent_records)}")
    print(f"Parents with partial best:   {partial_best}/{len(parent_records)}")
    print(f"All-beams CSV:    {csv_all_path}")
    print(f"Best-per-parent:  {csv_best_path}")
    print(f"Parent SMILES:    {parents_smiles_path}")
    print(f"Best gen SMILES:  {generated_best_smiles_path}")
    print(f"Best images:      {images_dir}")
    print(f"Plots:            {plots_dir}")
    print(f"Detail log:       {log_path}")
    print("=" * 70)

    return {
        f"{eval_type}_num_parents": len(parent_records),
        f"{eval_type}_mean_parent_bbb": float(parent_bbbs.mean()),
        f"{eval_type}_mean_best_bbb": float(best_bbbs.mean()),
        f"{eval_type}_mean_parent_total": float(parent_totals.mean()),
        f"{eval_type}_mean_best_total": float(best_totals.mean()),
        f"{eval_type}_mean_parent_qed": float(parent_qeds.mean()),
        f"{eval_type}_mean_best_qed": float(best_qeds.mean()),
        f"{eval_type}_mean_best_purity": float(best_purities.mean()),
        f"{eval_type}_num_improved_total": improved_total,
        f"{eval_type}_num_improved_bbb": improved_bbb,
        f"{eval_type}_num_pure_best": pure_best,
        f"{eval_type}_num_partial_best": partial_best,
    }

if __name__ == '__main__':
    print(">> Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config (e.g. 'experiments.exp_01')")

    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--rl_entropy_beta', type=float, default=None)
    parser.add_argument('--ppo_epochs', type=int, default=None)
    parser.add_argument('--rl_ppo_clip_epsilon', type=float, default=None)

    args = parser.parse_args()
    if args.config is not None:
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig
    else:
        from config import MoleculeConfig

    config = MoleculeConfig()
    if args.learning_rate is not None:
        config.optimizer["lr"] = args.learning_rate
    if args.rl_entropy_beta is not None:
        config.rl_entropy_beta = args.rl_entropy_beta
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.rl_ppo_clip_epsilon is not None:
        config.rl_ppo_clip_epsilon = args.rl_ppo_clip_epsilon

    print("Starting experiment on task:", config.objective_type)

    # --- WANDB INITIALIZATION ---
    if hasattr(config, 'use_wandb') and config.use_wandb:
        # Convert the config object to a dictionary for wandb
        config_dict = {k: v for k, v in config.__dict__.items() if not k.startswith('__')}
        # For nested dictionaries like 'optimizer', wandb prefers a flat structure
        flat_config = {}
        for k, v in config_dict.items():
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    flat_config[f"{k}.{sub_k}"] = sub_v
            else:
                flat_config[k] = v

        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name,
            config=flat_config
        )

        config.optimizer["lr"] = wandb.config.get('optimizer.lr', config.optimizer["lr"]) # Example
        config.rl_entropy_beta = wandb.config.get('rl_entropy_beta', config.rl_entropy_beta)
        config.ppo_epochs = wandb.config.get('ppo_epochs', config.ppo_epochs)
        config.rl_ppo_clip_epsilon = wandb.config.get('rl_ppo_clip_epsilon', config.rl_ppo_clip_epsilon)

        wandb.config.update({"task": config.objective_type}, allow_val_change=True)

    # --- Ray initialization ---
    if not getattr(config, 'disable_ray', False):
        num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))

        if ray.is_initialized():
            ray.shutdown()

        import platform
        is_local_windows = platform.system() == "Windows"

        ray_init_args = {
            "num_gpus": num_gpus,
            "logging_level": "info",
            "ignore_reinit_error": True,
        }

        if is_local_windows:
            ray_init_args["include_dashboard"] = False
            ray_init_args["_temp_dir"] = "C:/ray_tmp"
            import socket
            ray_init_args["address"] = "local"

        ray.init(**ray_init_args)
        print(ray.available_resources())
        oracle_tracker = OracleTracker.remote()
    else:
        print("[DEBUG MODE] Ray disabled. Running sequentially.")
        oracle_tracker = LocalOracleTracker()

    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    # Seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # ------------------------------------------------------------------
    # Top-K (all-time) SMILES archive (purely observational)
    # Lines format: <objective>\t<SMILES>
    # ------------------------------------------------------------------
    TOP_K_OBS = 10
    topk_archive_path = os.path.join(config.results_path, "top10_all_time_smiles.txt")
    topk_smiles_scores = {}  # {smiles: best_objective}


    def update_topk_archive_from_epoch(epoch_top20, rl_mode: bool):
        """
        epoch_top20:
          - RL mode: list[str] lines like '01: SMILES  obj=0.1234'
          - Supervised mode: list with a single dict {smiles: obj, ...}
        """
        if rl_mode:
            # In RL mode, epoch_top20 is now a list of dictionaries
            for entry in epoch_top20:
                smiles = entry['smiles']
                score = entry['unweighted']
                prev = topk_smiles_scores.get(smiles)
                if (prev is None) or (score > prev):
                    topk_smiles_scores[smiles] = score
        else:
            if not epoch_top20:
                return
            d = epoch_top20[0]
            for smiles, obj_val in d.items():
                prev = topk_smiles_scores.get(smiles)
                if (prev is None) or (obj_val > prev):
                    topk_smiles_scores[smiles] = obj_val

        # Truncate to top K by objective
        if len(topk_smiles_scores) > TOP_K_OBS:
            sorted_items = sorted(topk_smiles_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K_OBS]
            topk_smiles_scores.clear()
            topk_smiles_scores.update(sorted_items)

        # Persist: objective<TAB>SMILES (descending objective)
        os.makedirs(config.results_path, exist_ok=True)
        with open(topk_archive_path, "w") as f:
            for smiles, score in sorted(topk_smiles_scores.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{score:.6f}\t{smiles}\n")


    # Policy network
    network = MoleculeTransformer(config, config.training_device)
    objective_eval = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device,
                                                oracle_tracker=oracle_tracker)

    # Load checkpoint if needed
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path, weights_only=False)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "epochs_trained": 0,
            "validation_metric": float("-inf"),
            "best_validation_metric": float("-inf"),
            "best_validation_mean_score": float("-inf")
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"], strict=False)
        # network.load_state_dict(checkpoint["model_weights"])

    # Init new best_validation_mean_score if loading old checkpoint
    if "best_validation_mean_score" not in checkpoint:
        checkpoint["best_validation_mean_score"] = float("-inf")

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    if config.num_epochs > 0:
        print(f"Starting training for {config.num_epochs} epochs.")

        best_model_weights = checkpoint["best_model_weights"]
        best_validation_metric = checkpoint["best_validation_metric"]
        best_val_mean_score = checkpoint["best_validation_mean_score"]

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                checkpoint["epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        start_time_counter = None
        if config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {config.wall_clock_limit / 3600} hours")
            start_time_counter = time.perf_counter()

        rl_mode_active = getattr(config, "use_dr_grpo", False)

        if getattr(config, "rl_use_novelty_bonus") and rl_mode_active:
            print("Novelty bonus enabled.")
            novelty_memory = {}
        else:
            novelty_memory = None

        gumbeldore_dset = GumbeldoreDataset(config=config, objective_evaluator=objective_eval,
                                            oracle_tracker=oracle_tracker)

        for epoch in range(config.num_epochs):
            print("------")
            network_weights = copy.deepcopy(network.get_weights())

            if novelty_memory is not None:
                print(f"Start of Epoch {epoch + 1}: Novelty memory contains {len(novelty_memory)} unique SMILES.")

            if rl_mode_active:
                generated_loggable_dict, top20_dicts, top20_text = train_for_one_epoch_rl(
                    epoch, config, network, network_weights, optimizer, objective_eval, gumbeldore_dset,
                    novelty_memory=novelty_memory, oracle_tracker_=oracle_tracker
                )
                val_metric = generated_loggable_dict.get("best_gen_unweighted", float("-inf"))

            else:  # Original Supervised-only mode
                generated_loggable_dict, top20_text = train_for_one_epoch_supervised(
                    epoch, config, network, network_weights, optimizer, objective_eval, best_validation_metric,
                    oracle_tracker_=oracle_tracker
                )
                val_metric = generated_loggable_dict["best_gen_obj"]

            print("Num Unique Oracle Calls so far: ", generated_loggable_dict["num_unique_oracle_calls"])

            # --- VALIDATION STEP ---
            current_val_mean_score = float("-inf")
            current_val_success_rate = 0.0  # Initialize success rate

            if config.use_validation_for_ckpt and not config.prodrug_mode and config.use_dr_grpo:
                # Unpack the two return values
                current_val_mean_score, current_val_success_rate = validate_epoch(config, network, objective_eval)

                generated_loggable_dict["validation_mean_score"] = current_val_mean_score
                generated_loggable_dict["validation_success_rate"] = current_val_success_rate  # Log to file

            elif config.use_validation_for_ckpt and not config.prodrug_mode and not config.use_dr_grpo:
                # Unpack the two return values
                current_val_mean_score, current_val_success_rate, individual_scores = validate_supervised('validation', config, network, objective_eval)

                generated_loggable_dict["validation_mean_score"] = current_val_mean_score
                generated_loggable_dict["validation_success_rate"] = current_val_success_rate  # Log to file

            # -----------------------------

            # Update all-time top-K SMILES archive
            try:
                if rl_mode_active:  # top20_text is a list of strings
                    # update_topk_archive_from_epoch(top20_text, rl_mode=True)
                    update_topk_archive_from_epoch(top20_dicts, rl_mode=True)
                else:  # top20_text is a list containing one dictionary
                    update_topk_archive_from_epoch(top20_text, rl_mode=False)
            except Exception as e:
                print(f"[TopK Archive] Warning: failed to update archive this epoch: {e}")

            checkpoint["epochs_trained"] += 1
            scheduler.step()

            print(f">> Epoch {checkpoint['epochs_trained']}. "
                  f"Best (gen/rl) objective: {val_metric:.4f}")
            if rl_mode_active:
                mean_r = generated_loggable_dict.get('mean_reward', float('nan'))
                policy_l = generated_loggable_dict.get('policy_loss', float('nan'))
                print(f"   RL Stats: Mean Reward={mean_r:.4f}, Policy Loss={policy_l:.6f}")

            logger.log_metrics(generated_loggable_dict, step=epoch)

            # --- CHECKPOINT SAVING LOGIC ---
            saved_new_best = False

            if config.use_validation_for_ckpt and not config.prodrug_mode:
                # New Logic: Save based on Validation Mean Score
                if current_val_mean_score > best_val_mean_score:
                    print(
                        f">> New best VALIDATION score: {current_val_mean_score:.4f} (prev: {best_val_mean_score:.4f}). Saving new best model.")
                    best_val_mean_score = current_val_mean_score
                    checkpoint["best_validation_mean_score"] = best_val_mean_score
                    saved_new_best = True
            else:
                # Old Logic: Save based on best single molecule seen in training
                if val_metric > best_validation_metric:
                    print(f">> New best TRAINING molecule found: {val_metric:.4f}. Saving new best model.")
                    best_validation_metric = val_metric
                    checkpoint["best_validation_metric"] = best_validation_metric
                    saved_new_best = True

            if val_metric > best_validation_metric:
                print(f">> New best TRAINING molecule found: {val_metric:.4f}")
                best_validation_metric = val_metric
                checkpoint["best_validation_metric"] = best_validation_metric

            if saved_new_best:
                checkpoint["best_model_weights"] = copy.deepcopy(network.get_weights())
                save_checkpoint(checkpoint, "best_model.pt", config)
            # -------------------------------

            # WandB logging
            if hasattr(config, 'use_wandb') and config.use_wandb:
                wandb_log = {
                    "System/epoch": checkpoint["epochs_trained"],
                    "System/learning_rate": scheduler.get_last_lr()[0],

                    # 🟢 UNWEIGHTED (Raw Biological Progress)
                    "Unweighted/best_all_time": best_validation_metric,
                    "Unweighted/best_current_epoch": val_metric,
                    "Unweighted/mean_top_20_current_epoch": generated_loggable_dict.get('mean_top_20_unweighted',
                                                                                        float("nan")),
                    "Unweighted/mean_all_current_epoch": generated_loggable_dict.get('mean_unweighted_all',
                                                                                     float("nan")),
                }

                # Fetch and log Oracle calls
                if "num_unique_oracle_calls" not in generated_loggable_dict:
                    generated_loggable_dict["num_unique_oracle_calls"] = get_oracle_count(oracle_tracker, config)
                wandb_log["System/num_unique_oracle_calls"] = generated_loggable_dict["num_unique_oracle_calls"]

                # Validation metrics (Also Unweighted)
                if config.use_validation_for_ckpt and not config.prodrug_mode:
                    wandb_log["Unweighted/validation_mean_score"] = current_val_mean_score
                    wandb_log["Unweighted/validation_best_mean_score"] = best_val_mean_score
                    wandb_log["Unweighted/validation_success_rate"] = current_val_success_rate

                # Add specific RL metrics if in RL mode
                if rl_mode_active:
                    # 🔴 WEIGHTED (Lambda-scaled math used by GRPO)
                    wandb_log["Weighted/best_current_epoch"] = generated_loggable_dict.get('best_reward',
                                                                                           float("nan"))
                    wandb_log["Weighted/mean_all_current_epoch"] = generated_loggable_dict.get('mean_reward',
                                                                                               float("nan"))
                    wandb_log["Weighted/baseline"] = generated_loggable_dict.get('baseline', float("nan"))

                    # 🔵 RL internals
                    rl_keys = ['mean_advantage', 'std_advantage', 'policy_loss', 'mean_entropy', 'mean_traj_length',
                               'num_trajectories']
                    for key in rl_keys:
                        if key in generated_loggable_dict:
                            wandb_log[f"RL/{key}"] = generated_loggable_dict[key]

                    # # Prodrug specific tracking
                    # for key, val in generated_loggable_dict.items():
                    #     if key.startswith("prodrug/"):
                    #         wandb_log[key] = val

                # Track specific biological components if available (e.g. Kinase MPO)
                # Dynamically track all raw biological components
                for key, val in generated_loggable_dict.items():
                    if key.startswith("Biology_Raw/"):
                        wandb_log[key] = val
                    # Catch the older Kinase MPO keys
                    elif key in ['gsk3b_scores', 'jnk3_scores', 'qed_scores', 'sa_scores']:
                        wandb_log[f"Biology_Raw/{key}"] = val

                wandb.log(wandb_log)

            if rl_mode_active:
                logger.text_artifact(os.path.join(config.results_path, f"epoch_{epoch + 1}_train_top_20_molecules.txt"),
                                     "\n".join(top20_text))
            else:
                logger.text_artifact(os.path.join(config.results_path, f"epoch_{epoch + 1}_train_top_20_molecules.txt"),
                                     top20_text)

            # Update and save the 'last' model checkpoint
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(dict_to_cpu(optimizer.state_dict()))
            checkpoint["validation_metric"] = val_metric
            save_checkpoint(checkpoint, "last_model.pt", config)

            if start_time_counter is not None and time.perf_counter() - start_time_counter > config.wall_clock_limit:
                print("Time exceeded. Stopping training.")
                break

    if config.num_epochs == 0:
        print(f"Testing with loaded model.")
    else:
        print(f"Testing with best model.")
        best_ckpt_path = os.path.join(config.results_path, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(best_ckpt_path, weights_only=False)
            network.load_state_dict(checkpoint["model_weights"])
        else:
            print("WARNING: best_model.pt not found; using last model.")

    if checkpoint["model_weights"] is None and config.num_epochs == 0:
        print("WARNING! No training performed and no checkpoint loaded. Evaluating random model.")

    torch.cuda.empty_cache()
    with torch.no_grad():
        if config.objective_type == 'prodrug_bbb':
            test_loggable_dict = evaluate_prodrug_bbb('test', config, network, objective_eval)
        elif config.use_dr_grpo:
            test_loggable_dict = evaluate('test', config, network, objective_eval)
        else:
            test_loggable_dict = evaluate_supervised('test', config, network, objective_eval)
    print(">> TEST")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=0, step_desc="test")

    # WanB finish
    if hasattr(config, 'use_wandb') and config.use_wandb:
        wandb.finish()

    if not getattr(config, 'disable_ray', False):
        print("Finished. Shutting down ray.")
        ray.shutdown()
    else:
        print("Finished.")
