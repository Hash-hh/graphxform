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

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import torch
import numpy as np
import wandb
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_evaluator import MoleculeObjectiveEvaluator, OracleTracker
from rl_updates import dr_grpo_update, TrajectoryRecord

os.environ["RAY_raylet_start_wait_time_s"] = "120"  # Increase from default 60s

def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def validate_epoch(config: MoleculeConfig, network: MoleculeTransformer,
                   objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Runs deterministic validation on the scaffolds defined in config.validation_scaffolds_path.
    Returns the mean objective score and success rate.
    """
    val_path = getattr(config, 'validation_scaffolds_path', None)
    if not val_path or not os.path.exists(val_path):
        print(f"[Val] Validation path not found or not set: {val_path}")
        return float("-inf"), 0.0

    # Load Scaffolds
    with open(val_path, 'r') as f:
        val_scaffolds = [line.strip() for line in f if line.strip()]

    if not val_scaffolds:
        print("[Val] Validation file empty.")
        return float("-inf"), 0.0

    # Create a clean config for Greedy/Deterministic Search
    val_config = copy.deepcopy(config)
    val_config.gumbeldore_config["search_type"] = "beam_search"
    val_config.gumbeldore_config["beam_width"] = 1  # 1 molecule per scaffold
    val_config.gumbeldore_config["deterministic"] = True  # Deterministic transition
    val_config.gumbeldore_config["num_trajectories_to_keep"] = 1
    val_config.gumbeldore_config["destination_path"] = None  # Don't save to disk

    dataset = GumbeldoreDataset(config=val_config, objective_evaluator=objective_evaluator)

    print(f"[Val] Validating on {len(val_scaffolds)} scaffolds (Greedy Decoding)...")

    # Generate
    grouped_results = dataset.generate_dataset(
        network_weights=copy.deepcopy(network.get_weights()),
        memory_aggressive=False,
        prompts=val_scaffolds,
        return_raw_trajectories=True,
        mode="eval"
    )

    scores = []
    success_count = 0
    total_mols = 0

    # Check if we should calculate success rate (Only available for Kinase MPO)
    check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator, 'kinase_mpo_objective')

    for group in grouped_results:
        if group and group[0].objective is not None:
            mol = group[0]
            total_mols += 1

            # --- Score Handling ---
            val_ = mol.objective
            if val_ == float("-inf"):
                val_ = 0.0
            scores.append(val_)

            # --- Success Rate Handling ---
            if check_success and mol.smiles_string:
                # Access the underlying KinaseMPOObjective instance directly
                if objective_evaluator.kinase_mpo_objective.is_successful(mol.smiles_string):
                    success_count += 1

    if not scores:
        print("[Val] No valid molecules generated.")
        return float("-inf"), 0.0

    valid_scores = [s for s in scores if s > float("-inf")]

    if not valid_scores:
        print("[Val] All generated molecules were invalid (-inf).")
        return float("-inf"), 0.0

    if len(valid_scores) < len(scores):
        print(
            f"[Val] Warning: {len(scores) - len(valid_scores)} out of {len(scores)} molecules were invalid (-inf) and excluded from mean score calculation.")

    mean_val_score = np.mean(valid_scores)
    val_success_rate = (success_count / total_mols) if total_mols > 0 else 0.0

    print(
        f"[Val] Mean Score: {mean_val_score:.4f} | Success Rate: {val_success_rate:.2%} (over {total_mols} molecules)")
    return mean_val_score, val_success_rate


def validate_supervised(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Evaluates the model on validate scaffolds one-by-one for TASAR etc.
    Saves a detailed CSV of EVERY generated molecule for post-hoc analysis.
    """

    # Update config for evaluation
    # Create a clean config for Greedy/Deterministic Search
    config = copy.deepcopy(config_orig)
    config.gumbeldore_config["search_type"] = "beam_search"
    config.gumbeldore_config["beam_width"] = 1  # 1 molecule per scaffold
    config.gumbeldore_config["deterministic"] = True  # Deterministic transition
    config.gumbeldore_config["num_trajectories_to_keep"] = 1000
    config.gumbeldore_config["destination_path"] = None  # Don't save to disk

    # 1. Load Scaffolds
    validitation_prompts = []

    # Priority A: Check explicit path in config
    path = getattr(config, 'validation_scaffolds_path', None)
    if path and os.path.exists(path):
        print(f"[{eval_type}] Loading Scaffolds from: {path}")
        with open(path, 'r') as f:
            validitation_prompts = [line.strip() for line in f if line.strip()]

    # Priority B: Check Prodrug mode
    elif config.prodrug_mode:
        print(f"[{eval_type}] Using Prodrug test parents.")
        validitation_prompts = config.prodrug_parents_test

    if not validitation_prompts:
        print(f"[{eval_type}] Warning: No scaffolds found. Skipping evaluation.")
        return {}, ["No scaffolds found"]

    print(f"[{eval_type}] Found {len(validitation_prompts)} scaffolds. Processing one by one...")

    # 2. Setup Config for Evaluation
    eval_config = copy.deepcopy(config)
    eval_config.gumbeldore_config["destination_path"] = None  # Disable internal pickling

    dataset = GumbeldoreDataset(config=eval_config, objective_evaluator=objective_evaluator)
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

            # # individual metrics for each scaffold
            # if hasattr(objective_evaluator, 'kinase_mpo_objective'):
            #     individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(mol)

        total_mols += 1


        # if not scores:
        #     print("[Val] No valid molecules generated.")
        #     return float("-inf"), 0.0


    # --- Logging ---

    # Memory cleanup
    del grouped_results
    # import gc; gc.collect() # Uncomment if memory is extremely tight


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

    return np.mean(scores), success_count/total_mols, []  #individual_scores  TODO: Also log individual scores


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
        current_oracle_count = ray.get(oracle_tracker_.get_count.remote())
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

    # Build top 20 text artifact
    mol_map = {}
    for group in trajectories:
        for m in group:
            if m.objective is None:
                continue
            if not m.smiles_string:  # Good to add this check
                continue
            if m.smiles_string not in mol_map or mol_map[m.smiles_string]["obj"] < m.objective:
                mol_map[m.smiles_string] = {
                    "smiles": m.smiles_string,
                    "obj": m.objective
                }
    unique_mols = list(mol_map.values())
    unique_mols.sort(key=lambda x: x["obj"], reverse=True)
    top20 = unique_mols[:20]

    if top20:
        mean_top20_obj = sum(entry["obj"] for entry in top20) / len(top20)
        metrics["mean_top_20_obj"] = mean_top20_obj
    else:
        metrics["mean_top_20_obj"] = float("-inf")


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


    top_20_text_lines = []
    for i, entry in enumerate(top20):
        top_20_text_lines.append(f"{i + 1:02d}: {entry['smiles']}  obj={entry['obj']:.4f}")
    if not top20:
        top_20_text_lines.append("No terminated molecules")

    # Retrieve the global count to log it
    current_oracle_count = 0
    if oracle_tracker_ is not None:
        current_oracle_count = ray.get(oracle_tracker_.get_count.remote())
        metrics["num_unique_oracle_calls"] = current_oracle_count

    return metrics, top_20_text_lines


def evaluate(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Evaluates the model on test scaffolds one-by-one.
    Saves a detailed CSV of EVERY generated molecule for post-hoc analysis.
    RL (GRPO) version.
    """

    # Update config for evaluation
    # Create a clean config for Greedy/Deterministic Search
    config = copy.deepcopy(config_orig)
    config.gumbeldore_config["search_type"] = "beam_search"
    config.gumbeldore_config["beam_width"] = config.fixed_test_beam_width  # 1 molecule per scaffold
    config.gumbeldore_config["deterministic"] = True  # Deterministic transition
    config.gumbeldore_config["num_trajectories_to_keep"] = 1000
    config.gumbeldore_config["destination_path"] = None  # Don't save to disk

    # 1. Load Scaffolds
    test_prompts = []

    # Priority A: Check explicit path in config
    path = getattr(config, 'evaluation_scaffolds_path', None)
    if path and os.path.exists(path):
        print(f"[{eval_type}] Loading Scaffolds from: {path}")
        with open(path, 'r') as f:
            test_prompts = [line.strip() for line in f if line.strip()]

    # Priority B: Check Prodrug mode
    elif config.prodrug_mode:
        print(f"[{eval_type}] Using Prodrug test parents.")
        test_prompts = config.prodrug_parents_test

    if not test_prompts:
        print(f"[{eval_type}] Warning: No scaffolds found. Skipping evaluation.")
        return {}, ["No scaffolds found"]

    print(f"[{eval_type}] Found {len(test_prompts)} scaffolds. Processing one by one...")

    # 2. Setup Config for Evaluation
    eval_config = copy.deepcopy(config)
    eval_config.gumbeldore_config["destination_path"] = None  # Disable internal pickling

    # Create the CSV Log File
    os.makedirs(config.results_path, exist_ok=True)
    csv_filename = f"{eval_type}_detailed_logs.csv"
    csv_path = os.path.join(config.results_path, csv_filename)
    print(f"[{eval_type}] saving detailed logs to: {csv_path}")

    if getattr(eval_config, 'fixed_test_beam_width', None) is not None:
        eval_config.gumbeldore_config["beam_width"] = eval_config.fixed_test_beam_width

    print(f"[{eval_type}] using beam width:", eval_config.gumbeldore_config["beam_width"])

    dataset = GumbeldoreDataset(config=eval_config, objective_evaluator=objective_evaluator)
    weights = copy.deepcopy(network.get_weights())

    # 3. Open CSV and Start Loop
    # We use 'w' mode to overwrite if restarting, or 'a' could be used if careful.
    with open(csv_path, mode='w', newline='') as csv_file:
        fieldnames = ['scaffold_idx', 'prompt_smiles', 'generated_smiles', 'objective_score', 'is_successful',
                      'gsk3b', 'jnk3', 'qed', 'sa']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        scores = []
        success_count = 0
        total_mols = 0

        for idx, prompt in tqdm(enumerate(test_prompts), total=len(test_prompts), desc=f"Evaluating ({eval_type})"):

            # Initialize defaults
            smi = ""
            obj_val = 0.0
            is_successful = False
            gsk, jnk, qed, sa = 0, 0, 0, 0
            val_ = 0.0

            # Generate K candidates for this SINGLE prompt
            grouped_results = dataset.generate_dataset(
                network_weights=weights,
                memory_aggressive=False,
                prompts=[prompt],
                return_raw_trajectories=True,
                mode="eval"
            )

            # Check if generation returned anything
            if not grouped_results or not grouped_results[0]:
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

            # Check if we should calculate success rate (Only available for Kinase MPO) -- checkability xD
            check_success = (config.objective_type == 'kinase_mpo') and hasattr(objective_evaluator,
                                                                                'kinase_mpo_objective')

            group = grouped_results[0]  # List of MoleculeDesign objects

            # --- Detailed Logging to CSV ---

            best_mol = max(group, key=attrgetter('objective'))

            if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(best_mol.smiles_string)
                val_ = best_mol.objective
                gsk = individual_scores.get('GSK3B')
                jnk = individual_scores.get('JNK3')
                qed = individual_scores.get('QED')
                sa = individual_scores.get('SA')

            # Sort the objects
            ordered_group = sorted(
                group,
                key=lambda x: x.objective if x.objective is not None else float("-inf"),
                reverse=True
            )

            for mol in ordered_group:  # beam leaves
                is_successful = False
                obj_val = mol.objective if mol.objective is not None else float("-inf")
                smi = mol.smiles_string if mol.smiles_string else ""

                if check_success and mol:
                    if objective_evaluator.kinase_mpo_objective.is_successful(mol.smiles_string):
                        is_successful = True
                        success_count += 1

                if not is_successful:
                    continue

                # best mol individual scores
                val_ = mol.objective


                # individual metrics for each scaffold
                if hasattr(objective_evaluator, 'kinase_mpo_objective'):
                    individual_scores = objective_evaluator.kinase_mpo_objective.individual_scores(mol.smiles_string)
                    gsk = individual_scores.get('GSK3B')
                    jnk = individual_scores.get('JNK3')
                    qed = individual_scores.get('QED')
                    sa = individual_scores.get('SA')

                break

            total_mols += 1
            scores.append(val_)

            if not scores:
                print("[Val] No valid molecules generated.")
                return float("-inf"), 0.0


            # Write EVERY beam leaf to the CSV
            writer.writerow({
                'scaffold_idx': idx,
                'prompt_smiles': prompt,
                'generated_smiles': smi,
                'objective_score': obj_val if obj_val > float("-inf") else 0.0,
                'is_successful': is_successful,
                'gsk3b': gsk,
                'jnk3': jnk,
                'qed': qed,
                'sa': sa
            })


            # Memory cleanup
            del grouped_results
            # import gc; gc.collect() # Uncomment if memory is extremely tight


    # 4. Final Aggregation

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




def evaluate_supervised(eval_type: str, config_orig: MoleculeConfig, network: MoleculeTransformer,
             objective_evaluator: MoleculeObjectiveEvaluator):
    """
    Evaluates the model on test scaffolds one-by-one for TASAR etc.
    Saves a detailed CSV of EVERY generated molecule for post-hoc analysis.
    """

    # Update config for evaluation
    # Create a clean config for Greedy/Deterministic Search
    config = copy.deepcopy(config_orig)
    config.gumbeldore_config["search_type"] = "beam_search"
    config.gumbeldore_config["beam_width"] = config.fixed_test_beam_width  # 1 molecule per scaffold
    config.gumbeldore_config["deterministic"] = True  # Deterministic transition
    config.gumbeldore_config["num_trajectories_to_keep"] = 1000
    config.gumbeldore_config["destination_path"] = None  # Don't save to disk

    # 1. Load Scaffolds
    test_prompts = []

    # Priority A: Check explicit path in config
    path = getattr(config, 'evaluation_scaffolds_path', None)
    if path and os.path.exists(path):
        print(f"[{eval_type}] Loading Scaffolds from: {path}")
        with open(path, 'r') as f:
            test_prompts = [line.strip() for line in f if line.strip()]

    # Priority B: Check Prodrug mode
    elif config.prodrug_mode:
        print(f"[{eval_type}] Using Prodrug test parents.")
        test_prompts = config.prodrug_parents_test

    if not test_prompts:
        print(f"[{eval_type}] Warning: No scaffolds found. Skipping evaluation.")
        return {}, ["No scaffolds found"]

    print(f"[{eval_type}] Found {len(test_prompts)} scaffolds. Processing one by one...")

    # 2. Setup Config for Evaluation
    eval_config = copy.deepcopy(config)
    eval_config.gumbeldore_config["destination_path"] = None  # Disable internal pickling

    # Create the CSV Log File
    os.makedirs(config.results_path, exist_ok=True)
    csv_filename = f"{eval_type}_detailed_logs.csv"
    csv_path = os.path.join(config.results_path, csv_filename)
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
            # import gc; gc.collect() # Uncomment if memory is extremely tight


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




if __name__ == '__main__':
    print(">> Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config (e.g. 'experiments.exp_01')")

    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--rl_entropy_beta', type=float, default=None)
    parser.add_argument('--ppo_epochs', type=int, default=None)
    parser.add_argument('--rl_ppo_clip_epsilon', type=float, default=None)
    # parser.add_argument('--fragment_top_k', type=int, default=None)

    args = parser.parse_args()
    if args.config is not None:
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig
    else:
        from config import MoleculeConfig

    config = MoleculeConfig()

    # --- LOAD FRAGMENT VOCABULARY FOR RAY WORKERS ---
    if config.use_fragment_action_space and config.fragment_vocabulary is None:
        print("Loading fragment vocabulary for workers...")
        from core.fragment import load_fragment_vocabulary
        config.fragment_vocabulary = load_fragment_vocabulary(
            config.fragment_vocabulary_path,
            top_k=config.fragment_top_k,
        )
        config.K = len(config.fragment_vocabulary)
        print(f"Vocabulary loaded with {config.K} fragments.")
    # ------------------------------------------------

    if args.learning_rate is not None:
        config.optimizer["lr"] = args.learning_rate
    if args.rl_entropy_beta is not None:
        config.rl_entropy_beta = args.rl_entropy_beta
    if args.ppo_epochs is not None:
        config.ppo_epochs = args.ppo_epochs
    if args.rl_ppo_clip_epsilon is not None:
        config.rl_ppo_clip_epsilon = args.rl_ppo_clip_epsilon
    # if args.fragment_top_k is not None:
    #     config.fragment_top_k = args.fragment_top_k

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
        # config.fragment_top_k = wandb.config.get('fragment_top_k', config.fragment_top_k)

        wandb.config.update({"task": config.objective_type}, allow_val_change=True)
        # wandb.config.update({"task": config.objective_type})  # Log the task separately for easy filtering

    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))

    if ray.is_initialized():
        ray.shutdown()  # In case ray was already running and messing things up

    import platform

    is_local_windows = platform.system() == "Windows"

    ray_init_args = {
        "num_gpus": num_gpus,
        "logging_level": "info",
        "ignore_reinit_error": True,
        # "local_mode": True  <-- DELETE THIS. Local mode hides concurrency bugs.
    }

    if is_local_windows:
        # Windows-specific fixes to stop the crashing
        ray_init_args["include_dashboard"] = False  # Dashboard often crashes on Windows
        ray_init_args["_temp_dir"] = "C:/ray_tmp"  # Short path avoids path length errors

        # This fixes the VPN/Network blocking issue
        import socket

        ray_init_args["address"] = "local"
    # else:
    #     # Cluster settings (Linux)
    #     ray_init_args["include_dashboard"] = True  # Useful on cluster
    #     # On Slurm, Ray usually auto-detects the address, or you start it via script

    ray.init(**ray_init_args)

    print(ray.available_resources())

    # Create the Global Oracle Tracker Actor
    oracle_tracker = OracleTracker.remote()

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
            for line in epoch_top20:
                if "obj=" not in line:
                    continue
                try:
                    after_rank = line.split(":", 1)[1].strip()
                    tokens = after_rank.split()
                    if not tokens:
                        continue
                    obj_token = None
                    for t in reversed(tokens):
                        if t.startswith("obj="):
                            obj_token = t
                            break
                    if obj_token is None:
                        continue
                    obj_val = float(obj_token.split("obj=")[1])
                    obj_index = tokens.index(obj_token)
                    smiles = " ".join(tokens[:obj_index]).strip()
                    if not smiles:
                        continue
                    prev = topk_smiles_scores.get(smiles)
                    if (prev is None) or (obj_val > prev):
                        topk_smiles_scores[smiles] = obj_val
                except Exception:
                    continue
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
        checkpoint = torch.load(config.load_checkpoint_from_path)
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
        network.load_state_dict(checkpoint["model_weights"])

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
                generated_loggable_dict, top20_text = train_for_one_epoch_rl(
                    epoch, config, network, network_weights, optimizer, objective_eval, gumbeldore_dset,
                    novelty_memory=novelty_memory, oracle_tracker_=oracle_tracker
                )
                # The last return value (the buffer data) is not needed in the main loop, so we use _
                val_metric = generated_loggable_dict.get("best_gen_obj", float("-inf"))

            else:  # Original Supervised-only mode
                generated_loggable_dict, top20_text = train_for_one_epoch_supervised(
                    epoch, config, network, network_weights, optimizer, objective_eval, best_validation_metric,
                    oracle_tracker_=oracle_tracker
                )
                val_metric = generated_loggable_dict["best_gen_obj"]

            print("Num Unique Oracle Calls so far: ", generated_loggable_dict["num_unique_oracle_calls"])

            # # --- [NEW] VALIDATION STEP ---
            # current_val_mean_score = float("-inf")
            # current_val_success_rate = 0.0  # Initialize success rate
            #
            # if config.use_validation_for_ckpt and not config.prodrug_mode and config.use_dr_grpo:
            #     # Unpack the two return values
            #     current_val_mean_score, current_val_success_rate = validate_epoch(config, network, objective_eval)
            #
            #     generated_loggable_dict["validation_mean_score"] = current_val_mean_score
            #     generated_loggable_dict["validation_success_rate"] = current_val_success_rate  # Log to file
            #
            # elif config.use_validation_for_ckpt and not config.prodrug_mode and not config.use_dr_grpo:
            #     # Unpack the two return values
            #     current_val_mean_score, current_val_success_rate, individual_scores = validate_supervised('validation', config, network, objective_eval)
            #
            #     generated_loggable_dict["validation_mean_score"] = current_val_mean_score
            #     generated_loggable_dict["validation_success_rate"] = current_val_success_rate  # Log to file
            #
            # # -----------------------------

            # --- [NEW] VALIDATION STEP ---
            current_val_mean_score = float("-inf")
            current_val_success_rate = 0.0  # Initialize success rate

            # Check if it's a validation epoch (every 50 epochs, or the final epoch)
            is_val_epoch = ((epoch + 1) % 50 == 0) or ((epoch + 1) == config.num_epochs)

            if is_val_epoch and config.use_validation_for_ckpt and not config.prodrug_mode:
                if config.use_dr_grpo:
                    # Unpack the two return values
                    current_val_mean_score, current_val_success_rate = validate_epoch(config, network, objective_eval)
                else:
                    # Unpack the three return values
                    current_val_mean_score, current_val_success_rate, individual_scores = validate_supervised(
                        'validation', config, network, objective_eval)

                generated_loggable_dict["validation_mean_score"] = current_val_mean_score
                generated_loggable_dict["validation_success_rate"] = current_val_success_rate  # Log to file
            # -----------------------------

            # Update all-time top-K SMILES archive
            try:
                if rl_mode_active:  # top20_text is a list of strings
                    update_topk_archive_from_epoch(top20_text, rl_mode=True)
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
                    "epoch": checkpoint["epochs_trained"],
                    "best_all_time_obj": best_validation_metric,
                    "best_current_obj": val_metric,
                    'mean_current_obj': generated_loggable_dict.get('mean_best_gen_obj'),
                    'worst_current_obj': generated_loggable_dict.get('worst_gen_obj'),
                    "mean_top_20_all_time_obj": generated_loggable_dict.get("mean_top_20_obj"),
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                # Fetch count if not already in dict
                if "num_unique_oracle_calls" not in generated_loggable_dict:
                    count = ray.get(oracle_tracker.get_count.remote())
                    generated_loggable_dict["num_unique_oracle_calls"] = count

                wandb_log["num_unique_oracle_calls"] = generated_loggable_dict["num_unique_oracle_calls"]

                if config.use_validation_for_ckpt and not config.prodrug_mode:
                    wandb_log["validation_mean_score"] = current_val_mean_score
                    wandb_log["best_validation_mean_score"] = best_val_mean_score
                    wandb_log["validation_success_rate"] = current_val_success_rate  # Log to WandB

                # Add specific RL metrics if in RL mode
                if rl_mode_active:
                    # Define the specific list of metrics you want to log from the RL update
                    keys_to_log = [
                        'baseline',
                        'mean_reward',
                        'best_reward',
                        # 'best_objective',
                        'mean_advantage',
                        'std_advantage',
                        'policy_loss',
                        'mean_entropy',
                        'mean_traj_length',
                        'num_trajectories',
                        'mean_top_20_obj'
                        # 'mean_novelty_bonus'
                    ]
                    # Add only the specified metrics to the log
                    for key in keys_to_log:
                        if key in generated_loggable_dict:
                            wandb_log[key] = generated_loggable_dict[key]

                    # This captures 'prodrug/mean_logp_delta', 'prodrug/fraction_cleavable', etc.
                    for key, val in generated_loggable_dict.items():
                        if key.startswith("prodrug/"):
                            wandb_log[key] = val

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
            checkpoint = torch.load(best_ckpt_path)
            network.load_state_dict(checkpoint["model_weights"])
        else:
            print("WARNING: best_model.pt not found; using last model.")

    if checkpoint["model_weights"] is None and config.num_epochs == 0:
        print("WARNING! No training performed and no checkpoint loaded. Evaluating random model.")

    torch.cuda.empty_cache()
    with torch.no_grad():
        if config.use_dr_grpo:
            test_loggable_dict = evaluate('test', config, network, objective_eval)
        else:
            test_loggable_dict = evaluate_supervised('test', config, network, objective_eval)
    print(">> TEST")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=0, step_desc="test")

    # WanB finish
    if hasattr(config, 'use_wandb') and config.use_wandb:
        wandb.finish()

    print("Finished. Shutting down ray.")
    ray.shutdown()
