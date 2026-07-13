import copy
import os
import pickle

from model.molecule_transformer import MoleculeTransformer
from molecule_design import MoleculeDesign

os.environ["RAY_DEDUP_LOGS"] = "0"
import sys
import ray
import torch
import time
import numpy as np
from ray.thirdparty_files import psutil
from tqdm import tqdm
from rdkit import RDLogger

from core.abstracts import Config, Instance, BaseTrajectory
import core.stochastic_beam_search as sbs

from typing import List, Callable, Tuple, Any, Type, Optional

from core.incremental_sbs import IncrementalSBS

import random
from config import MoleculeConfig
from molecule_evaluator import MoleculeObjectiveEvaluator

from contextlib import nullcontext
from rl_updates import _make_autocast_ctx


def batched_iid_monte_carlo_sampling(
        root_nodes: List[MoleculeDesign],
        num_samples_per_instance: int,
        log_prob_fn: Callable[[List[MoleculeDesign]], List[np.ndarray]],
        config: MoleculeConfig
) -> List[List[MoleculeDesign]]:
    """
    Performs batched IID Monte Carlo sampling. For each root node, it generates
    `num_samples_per_instance` trajectories in a batched manner to leverage GPU parallelism.
    """
    results_by_root = []
    temperature = config.gumbeldore_config.get("sampling_temperature", 1.0)

    autocast_ctx, _ = _make_autocast_ctx(config) if config.use_amp_inference else (nullcontext(), None)

    for root_node in root_nodes:
        active_trajectories = [root_node._shallow_clone() for _ in range(num_samples_per_instance)]
        finished_trajectories = []

        while active_trajectories:

            with autocast_ctx:
                # Get log probabilities for all active trajectories in a single batch
                log_probs_list = log_prob_fn(active_trajectories)

            next_active_trajectories = []

            for i, trajectory in enumerate(active_trajectories):
                log_probs = log_probs_list[i]

                # Apply temperature to logits (log_probs)
                with np.errstate(divide='ignore'):  # Ignore division by zero if temperature is 0
                    probs = np.exp(log_probs / temperature)

                # Handle cases where all actions are infeasible, leading to sum(probs)=0
                if np.sum(probs) <= 0 or not np.isfinite(probs).all():
                    print("Warning: All actions infeasible or non-finite probabilities encountered. Marking trajectory as infeasible.")
                    trajectory.infeasibility_flag = True
                    trajectory.finalize()
                    finished_trajectories.append(trajectory)
                    continue

                probs /= np.sum(probs)
                action = np.random.choice(len(probs), p=probs)

                # # The transition_fn performs a shallow copy and applies the action
                # new_trajectory, is_done = trajectory.transition_fn(action)

                chosen_log_prob = log_probs_list[i][action]
                new_trajectory, is_done = trajectory.transition_fn(action, chosen_log_prob)

                if is_done:
                    finished_trajectories.append(new_trajectory)
                else:
                    next_active_trajectories.append(new_trajectory)

            active_trajectories = next_active_trajectories

        results_by_root.append(finished_trajectories)

    return results_by_root


@ray.remote
class JobPool:
    def __init__(self, problem_instances: List[Instance]):
        self.jobs = [(i, instance) for i, instance in enumerate(problem_instances)]
        self.job_results = []

    def get_jobs(self, n_items: int):
        if len(self.jobs) > 0:
            items = self.jobs[:n_items]
            self.jobs = self.jobs[n_items:]
            return items
        else:
            return None

    def push_results(self, results: List[Tuple[int, Any]]):
        self.job_results.extend(results)

    def fetch_results(self):
        results = self.job_results
        self.job_results = []
        return results


class GumbeldoreDataset:
    def __init__(self, config: MoleculeConfig,
                 objective_evaluator: MoleculeObjectiveEvaluator,
                 oracle_tracker=None
                 ):
        self.config = config
        self.gumbeldore_config = config.gumbeldore_config
        self.objective_evaluator = objective_evaluator
        self.oracle_tracker = oracle_tracker
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"]

        self.fragment_library = []
        if config.use_fragment_library:
            with open(config.fragment_library_path, 'r') as f:
                self.fragment_library = [line.strip() for line in f if line.strip()]
            if not self.fragment_library:
                raise FileNotFoundError("Fragment file is empty.")
            print(f"[GRPO] Loaded {len(self.fragment_library)} fragments from {config.fragment_library_path}.")

    def generate_dataset(self, network_weights: dict, best_objective: Optional[float] = None,
                         memory_aggressive: bool = False,
                         prompts: Optional[List[str]] = None,
                         return_raw_trajectories: bool = False,
                         mode: str = "eval"
                         ):
        """
        Parameters:
            return_raw_trajectories: During eval we want to always return the raw trajectories
            network_weights: [dict] Network weights to use for generating data.
            memory_aggressive: [bool] If True, IncrementalSBS is performed "memory aggressive" meaning that
                intermediate states in the search tree are not stored after transitioning from them, only their
                policies.
            prompts: [List[str], optional] List of SMILES strings to use as prompts in Prodrug mode.

        Behavior:
            - If config.use_dr_grpo is False: returns metrics dict (original behavior).
            - If config.use_dr_grpo is True: returns a flat List[MoleculeDesign] (raw trajectories) and does NOT call process_results.
            - Also accept `prompts` (List[str]) for Prodrug mode.

        Generates trajectories.
        Hierarchy:
        1. prompts arg (Testing)
        2. Prodrug Mode (Config)
        3. Fragment Library (Training Scaffolds)
        4. C-Chain / Single Atom (De Novo)
        """
        batch_size_gpu, batch_size_cpu = (self.gumbeldore_config["batch_size_per_worker"],
                                          self.gumbeldore_config["batch_size_per_cpu_worker"])

        problem_instances = []

        is_prodrug_mode = getattr(self.config, 'prodrug_mode', False)

        # Explicit Prompts (TESTING / INFERENCE)
        # Used when evaluate() passes the test_scaffolds list
        if prompts is not None and len(prompts) > 0 and not is_prodrug_mode:
            print(f"[GumbeldoreDataset] Using {len(prompts)} explicit prompts for generation.")
            for smi in prompts:
                problem_instances.append(
                    MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                )

        # Prodrug Mode (Training)
        elif is_prodrug_mode and mode=="train":
            include_carbon_prompt = getattr(self.config, 'include_carbon_prompt', True)
            target_smiles = getattr(self.config, 'prodrug_parents_train', [])
            if not target_smiles: raise ValueError("Prodrug mode enabled but no parents found.")
            for smi in target_smiles:
                problem_instances.append(
                    MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                )
            if include_carbon_prompt:
                print(f"[GumbeldoreDataset] Including carbon prompt in Prodrug training mode.")
                problem_instances.append(
                    MoleculeDesign.from_smiles(self.config, 'C', do_finish=False)
                )

        elif is_prodrug_mode and mode=="eval":
            print(f"[GumbeldoreDataset] Using {len(prompts)} explicit prodrug parent test groups for generation.")
            for smi in prompts:
                problem_instances.append(
                    MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                )

        # Scaffold Training Mode
        elif self.config.use_dr_grpo and self.config.use_fragment_library and self.fragment_library:
            # Sample N random scaffolds from the loaded library
            n_prompts = self.config.num_prompts_per_epoch

            # Safety check if library is smaller than requested batch
            if n_prompts > len(self.fragment_library):
                sampled = random.sample(self.fragment_library, len(self.fragment_library))
            else:
                print(f"[GumbeldoreDataset] Including carbon prompt in scaffold sampling. Sampling {n_prompts-1} from library.")
                include_carbon_prompt = getattr(self.config, 'include_carbon_prompt', True)
                if include_carbon_prompt:
                    sampled = random.sample(self.fragment_library, n_prompts-1)
                    sampled.append('C')
                else:
                    print(f"[Gumbeldoredataset] Sampling {n_prompts} from library.")
                    sampled = random.sample(self.fragment_library, n_prompts)

            # print(f"[Generator] Sampling {len(sampled)} scaffolds for training.")
            for smi in sampled:
                try:
                    problem_instances.append(
                        MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                    )
                except Exception as e:
                    print(f"[Warning] Failed to load scaffold {smi}: {e}")

        # De Novo / C-Chain Mode
        # De Novo / C-Chain Mode
        elif self.config.start_from_c_chains:
            if getattr(self.config, 'use_fragment_action_space', True):
                # Fragment mode: use empty graphs
                n_instances = self.config.repeat_start_instances
                problem_instances = MoleculeDesign.get_empty_graph_batch(
                    self.config, n_instances
                )
            else:
                # Atomic mode: use C-chains
                problem_instances = MoleculeDesign.get_c_chains(self.config)

        # Single Atom Fallback
        else:
            problem_instances = MoleculeDesign.get_single_atom_molecules(self.config,
                                                                         repeat=self.config.repeat_start_instances)

        if not problem_instances:
            raise ValueError("No instances created. Check Config.")


        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        future_tasks = [
            async_sbs_worker.remote(
                self.config, job_pool, network_weights, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive,
                self.oracle_tracker
            )
            for i, device in enumerate(self.devices_for_workers)
        ]

        with tqdm(total=len(problem_instances)) as progress_bar:
            while True:
                # Check if all workers are done. If so, break after this iteration
                do_break = len(ray.wait(future_tasks, num_returns=len(future_tasks), timeout=0.5)[1]) == 0

                fetched_results = ray.get(job_pool.fetch_results.remote())
                for (i, result) in fetched_results:
                    results[i] = result
                if len(fetched_results):
                    progress_bar.update(len(fetched_results))

                if do_break:
                    break

        ray.get(future_tasks)
        del job_pool
        del network_weights
        torch.cuda.empty_cache()

        # RL mode: return raw MoleculeDesign list (no metrics dict)
        if self.config.use_dr_grpo:
            grouped_designs: List[List[MoleculeDesign]] = []

            for group_result in results:
                if not group_result:
                    continue

                # Normalize: Ensure we always have a list of MoleculeDesign objects
                # Check if it's BeamLeaf objects (from 'wor') or MoleculeDesign (from 'iid_mc')
                if isinstance(group_result[0], sbs.BeamLeaf):
                    grouped_designs.append([leaf.state for leaf in group_result])
                else:
                    grouped_designs.append(group_result)

            # Caller want raw trajectories during evaluation
            if return_raw_trajectories:
                # During Evaluation, we ALWAYS want the groups preserved to calculate per-scaffold stats.
                # Even if we trained with Global Baseline (no grouping), we Evaluate per scaffold.
                return grouped_designs

            # Check if we should enforce grouping (Local Baseline) or flatten (Global Baseline)
            # Default to True (Standard GRPO)
            use_grouping = getattr(self.config, 'use_grpo_grouping', True)

            if use_grouping:
                # STANDARD GRPO: Return List of Lists [[Mols_Scaffold_A], [Mols_Scaffold_B]]
                # Advantage is calculated locally per list.
                return grouped_designs
            else:
                # GLOBAL BASELINE (Ablation): Return List of one List [[All_Mols]]
                # Advantage is calculated globally across the entire batch.
                print("[GRPO] Grouping DISABLED. Flattening batch.")
                all_mols = [m for group in grouped_designs for m in group]
                return [all_mols]

        # LOCAL BASELINE MODE (Standard GRPO)
        # Returns List of Lists. Advantage is calculated relative to the SCAFFOLD mean.
        return self.process_results(problem_instances, results)


            # flat: List[MoleculeDesign] = []
            # for lst in results:
            #     if not lst:
            #         continue
            #     flat.extend(lst)
            # return flat

        # Supervised / non-RL: original metrics path
        # print(self.process_results(problem_instances, results))
        return self.process_results(problem_instances, results)

    def process_results(self, problem_instances, results):
        """
        Processes the results from Gumbeldore search and save it to a pickle. Each trajectory will be represented as a dict with the
        following keys and values
          "start_atom": [int] the int representing the atom from which to start
          "action_seq": List[List[int]] Actions which need to be taken on each index to create the molecule
          "smiles": [str] Corresponding smiles string
          "obj": [float] Objective function evaluation

        Then:
        1. The results will be cleaned from duplicate SMILES and molecules which do have an objective of -inf.
        2. If the dataset already exists at the path where to save, we load it, merge them and take the best from the
            merged dataset.

        Then returns the following dictionary:
        - "mean_best_gen_obj": Mean best generated obj. -> over the unmerged best molecules generated
        - "best_gen_obj": Best generated obj. -> Best obj. of the unmerged molecules generated
        - "worst_gen_obj": Worst generated obj. -> Worst obj. of the unmerged molecules generated
        - "mean_top_20_obj": Mean top 20 obj. -> over the merged best molecules
        - "top_20_molecules": A list of SMILES strings with obj. of the top 20 obj.
        """
        metrics_return = dict()
        instances_dict = dict()  # Use a dict to directly avoid duplicates
        for i, _ in enumerate(problem_instances):
            if not results[i]:
                continue
            for molecule in results[i]:  # type: MoleculeDesign
                if molecule.objective > float("-inf"):
                    instances_dict[molecule.smiles_string] = dict(
                        start_atom=getattr(molecule, 'initial_atom', None),
                        initial_fragment=getattr(molecule, 'initial_fragment', None),
                        action_seq=molecule.history,
                        smiles=molecule.smiles_string,
                        obj=molecule.objective,
                        sa_score=molecule.sa_score
                    )
        if not instances_dict:
            return {
                "mean_best_gen_obj": float("-inf"),
                "mean_best_gen_sa_score": float("-inf"),
                "best_gen_obj": float("-inf"),
                "best_gen_sa_score": float("-inf"),
                "worst_gen_obj": float("-inf"),
                "worst_gen_sa_score": float("-inf"),
                "mean_top_20_obj": float("-inf"),
                "mean_kept_obj": float("-inf"),
                "mean_top_20_sa_score": float("-inf"),
                "top_20_molecules": []
            }

        generated_mols = list(instances_dict.values())
        generated_mols = sorted(generated_mols, key=lambda x: x["obj"], reverse=True)[
            :self.gumbeldore_config["num_trajectories_to_keep"]]
        generated_objs = np.array([x["obj"] for x in generated_mols])
        generated_sa_scores = np.array([x["sa_score"] for x in generated_mols])
        metrics_return["mean_best_gen_obj"] = generated_objs.mean()
        metrics_return["mean_best_gen_sa_score"] = generated_sa_scores.mean()
        metrics_return["best_gen_obj"] = generated_objs[0]
        metrics_return["best_gen_sa_score"] = generated_sa_scores[0]
        metrics_return["worst_gen_obj"] = generated_objs[-1]
        metrics_return["worst_gen_sa_score"] = generated_sa_scores[-1]

        # Now check if there already is a data file, and if so, load it and merge it.
        destination_path = self.gumbeldore_config["destination_path"]
        merged_mols = generated_mols
        if destination_path is not None:
            if os.path.isfile(destination_path):
                with open(destination_path, "rb") as f:
                    existing_mols = pickle.load(f)  # list of dicts
                temp_d = {x["smiles"]: x for x in existing_mols + merged_mols}
                merged_mols = list(temp_d.values())
                merged_mols = sorted(merged_mols, key=lambda x: x["obj"], reverse=True)[
                    :self.gumbeldore_config["num_trajectories_to_keep"]]

            # Pickle the generated data again
            with open(destination_path, "wb") as f:
                pickle.dump(merged_mols, f)

        # # Get overall best metrics and molecules
        # metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in merged_mols[:20]]).mean()
        # metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_mols]).mean()
        # metrics_return["mean_top_20_sa_score"] = np.array([x["sa_score"] for x in merged_mols[:20]]).mean()
        # metrics_return["top_20_molecules"] = [{x["smiles"]: x["obj"] for x in merged_mols[:20]}]

        # TODO: Add a seperate return for all molecules; don't use top 20
        metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in merged_mols]).mean()
        metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_mols]).mean()
        metrics_return["mean_top_20_sa_score"] = np.array([x["sa_score"] for x in merged_mols]).mean()
        metrics_return["top_20_molecules"] = [{x["smiles"]: x["obj"] for x in merged_mols}]

        return metrics_return


@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False,
                    oracle_tracker=None
                     ):
    # Add this debug print:
    print(
        f"[Worker Debug] Fragment mode: {config.use_fragment_action_space}, K: {len(config.fragment_vocabulary) if config.fragment_vocabulary else 0}")

    network = MoleculeTransformer(config, device)
    network.load_state_dict(network_weights)
    network.to(network.device)
    network.eval()

    autocast_ctx, _ = _make_autocast_ctx(config) if config.use_amp_inference else (nullcontext(), None)

    def child_log_probability_fn(trajectories: List[MoleculeDesign]) -> [np.array]:
        if not all(trajectories):
            print(f"[DEBUG] child_log_probability_fn received a list containing None.", flush=True)
            # Print the list for inspection
            print(f"[DEBUG] Trajectories list: {trajectories}", flush=True)
        # --- END DEBUG PRINT ---

        with autocast_ctx:
            return MoleculeDesign.log_probability_fn(trajectories=trajectories, network=network)

    def batch_leaf_evaluation_fn(trajectories: List[MoleculeDesign]) -> np.array:
        objs = objective_evaluator.predict_objective(trajectories)
        for i, obj in enumerate(objs):
            trajectories[i].objective = obj
        return objs

    def child_transition_fn(trajectory_action_pairs: List[Tuple[MoleculeDesign, int]]):
        # Collect all trajectories that need a forward pass into a single list.
        trajectories = [traj for traj, _ in trajectory_action_pairs]

        with autocast_ctx:
            # Perform a SINGLE batched forward pass to get all log probabilities at once.
            log_probs_list = MoleculeDesign.log_probability_fn(trajectories, network)

        # Now, iterate through the original pairs and apply the actions using the pre-computed log probs.
        results = []
        for i, (traj, action) in enumerate(trajectory_action_pairs):
            # Get the log probability for the chosen action from the batched result.
            chosen_log_prob = log_probs_list[i][action]

            # Call the transition function with the correct log probability.
            new_trajectory, is_done = traj.transition_fn(action, chosen_log_prob)
            results.append((new_trajectory, is_done))

        return results

        # return [traj.transition_fn(action) for traj, action in trajectory_action_pairs]

    def _sample_diverse_from_sorted(
            beam_leaves: List[MoleculeDesign], num_keep: int, randomly=False
    ) -> List[MoleculeDesign]:
        """
        Picks items from a list sorted in descending order based on a stratified strategy.

        The strategy is:
        - 25% from the top (highest objective).
        - 25% from the bottom (lowest objective within the leaves).
        - 50% sampled randomly from the middle portion.

        This ensures diversity in the training batch by including top performers,
        marginal performers, and a random sample of the rest.

        Args:
            beam_leaves: A list of objects, pre-sorted in descending order by their
                         objective value.
            num_keep: The total number of items to select and return.

        Returns:
            A new list containing the selected items, totaling num_keep.
        """
        if randomly:
            # just keep the top N molecules and randomly sample the rest
            N = 10
            if len(beam_leaves) <= num_keep:
                return beam_leaves
            top_samples = beam_leaves[:N]
            remaining_pool = beam_leaves[N:]
            remaining_samples = random.sample(remaining_pool, k=num_keep - N)
            final_selection = top_samples + remaining_samples
            random.shuffle(final_selection)
            return final_selection

        total_leaves = len(beam_leaves)

        # --- Edge Case Handling ---
        # If we don't have enough leaves to sample from, just return them all.
        if total_leaves <= num_keep:
            return beam_leaves

        # --- 1. Calculate the counts for each stratum ---
        num_top = int(0.25 * num_keep)
        num_bottom = int(0.25 * num_keep)
        # The remainder will be sampled from the middle to ensure we get exactly num_keep
        num_middle = num_keep - num_top - num_bottom

        # --- 2. Select the top samples ---
        # These are the first `num_top` elements because the list is sorted descendingly.
        top_samples = beam_leaves[:num_top]

        # --- 3. Select the bottom samples ---
        # These are the last `num_bottom` elements.
        # Handle the case where num_bottom might be 0.
        bottom_samples = beam_leaves[-num_bottom:] if num_bottom > 0 else []

        # --- 4. Define the middle pool and sample from it ---
        # The middle pool consists of elements not in the top or bottom sections.
        middle_pool = beam_leaves[num_top:-num_bottom] if num_bottom > 0 else beam_leaves[num_top:]

        # Ensure we don't try to sample more than available in the pool.
        # This shouldn't happen with the initial `total_leaves <= num_keep` check,
        # but it's good practice for robustness.
        actual_middle_sample_size = min(num_middle, len(middle_pool))

        # Sample randomly without replacement from the middle pool.
        middle_samples = random.sample(middle_pool, k=actual_middle_sample_size)

        # --- 5. Combine and return the results ---
        # The final list will have top, a random assortment from the middle, and bottom.
        final_selection = top_samples + middle_samples + bottom_samples

        # As a final guardrail, shuffle the result so the training process doesn't see
        # samples in a biased (top-middle-bottom) order.
        random.shuffle(final_selection)

        return final_selection

    # Silence RDKit warnings
    RDLogger.DisableLog('rdApp.*')

    # Pin worker to core if wanted
    if cpu_core is not None:
        os.sched_setaffinity(0, {cpu_core})
        psutil.Process().cpu_affinity([cpu_core])

    with torch.no_grad():

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        device = torch.device(device)
        network = MoleculeTransformer(config, device)
        network.load_state_dict(network_weights)
        network.to(network.device)
        network.eval()

        objective_evaluator = MoleculeObjectiveEvaluator(config, torch.device(config.objective_gnn_device),
                                                         oracle_tracker=oracle_tracker)

        while True:
            batch = ray.get(job_pool.get_jobs.remote(batch_size))
            if batch is None:
                break

            idx_list = [i for i, _ in batch]
            root_nodes = [instance for _, instance in batch]

            if config.gumbeldore_config["search_type"] == "iid_mc":
                # New Batched IID Monte Carlo Sampling
                num_samples = config.gumbeldore_config.get("num_samples_per_instance", 512)
                trajectories_by_root = batched_iid_monte_carlo_sampling(root_nodes, num_samples,
                                                                        child_log_probability_fn,
                                                                        config)

                # Evaluate all generated trajectories
                all_trajectories = [traj for traj_list in trajectories_by_root for traj in traj_list]
                if all_trajectories:
                    batch_leaf_evaluation_fn(all_trajectories)

                # Group results for pushing back to the job pool
                results_to_push = [(idx, trajectories) for idx, trajectories in zip(idx_list, trajectories_by_root)]

            elif config.gumbeldore_config["search_type"] == "beam_search":
                # Stochastic/deterministic beam search.
                beam_leaves_batch: List[List[sbs.BeamLeaf]] = sbs.stochastic_beam_search(
                    child_log_probability_fn=child_log_probability_fn,
                    child_transition_fn=child_transition_fn,
                    root_states=root_nodes,
                    beam_width=config.gumbeldore_config["beam_width"],
                    deterministic=config.gumbeldore_config["deterministic"]
                )
                results_to_push = []
                for j, result_idx in enumerate(idx_list):
                    # result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j][
                    #     :config.gumbeldore_config["num_trajectories_to_keep"]]]
                    result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j]]
                    if result and result[0].objective is None:
                        batch_leaf_evaluation_fn(result)
                    results_to_push.append((result_idx, result))

            else:  # tasar, wor
                inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=MoleculeDesign.to_max_evaluation_fn,
                                         batch_leaf_evaluation_fn=batch_leaf_evaluation_fn,
                                         memory_aggressive=False)

                if config.gumbeldore_config["search_type"] == "tasar":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_tasar(
                        beam_width=config.gumbeldore_config["beam_width"],
                        deterministic=config.gumbeldore_config["deterministic"],
                        nucleus_top_p=config.gumbeldore_config["nucleus_top_p"],
                        replan_steps=config.gumbeldore_config["replan_steps"],
                        sbs_keep_intermediate=config.gumbeldore_config["keep_intermediate_trajectories"]
                    )
                elif config.gumbeldore_config["search_type"] == "wor":
                    beam_leaves_batch: List[List[sbs.BeamLeaf]] = inc_sbs.perform_incremental_sbs(
                        beam_width=config.gumbeldore_config["beam_width"],
                        num_rounds=config.gumbeldore_config["num_rounds"],
                        nucleus_top_p=config.gumbeldore_config["nucleus_top_p"],
                        sbs_keep_intermediate=config.gumbeldore_config["keep_intermediate_trajectories"],
                        best_objective=best_objective
                    )
                else:
                    raise ValueError(f"Unknown search type: {config.gumbeldore_config['search_type']}")

                results_to_push = []
                for j, result_idx in enumerate(idx_list):
                    # result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j][
                    #     :config.gumbeldore_config["num_trajectories_to_keep"]]]
                    if config.gumbeldore_config["search_type"] == "wor":
                        result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j]]
                    elif config.gumbeldore_config["search_type"] == "tasar":
                        result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j][
                            :config.gumbeldore_config["num_trajectories_to_keep"]]]
                    # we need to sample a diverse set from the leaves beam_leaves_batch[j] which is already sorted in
                    # descending order based on objective as expected by _sample_diverse_from_sorted
                    # result: List[MoleculeDesign] = _sample_diverse_from_sorted(result,
                    #     config.gumbeldore_config["num_trajectories_to_keep"], randomly=True)
                    # Check if they need objective evaluation (this will only be true for deterministic beam search
                    if result and result[0].objective is None:
                        batch_leaf_evaluation_fn(result)
                    results_to_push.append((result_idx, result))

            ray.get(job_pool.push_results.remote(results_to_push))

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()