import copy
import os
import pickle

from model.molecule_transformer import MoleculeTransformer
from molecule_design import MoleculeDesign

os.environ["RAY_DEDUP_LOGS"] = "0"
import sys
import ray
import torch
# import time
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
from molecule_evaluator import RemoteEvaluatorProxy

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
    temperature = config.gumbeldore_config.get("sampling_temperature")

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

# def batched_iid_monte_carlo_sampling(
#         root_nodes: List[MoleculeDesign],
#         num_samples_per_instance: int,
#         log_prob_fn: Callable[[List[MoleculeDesign]], List[np.ndarray]],
#         config: MoleculeConfig
# ) -> List[List[MoleculeDesign]]:
#     """
#     Performs batched IID Monte Carlo sampling with a Random Pivot.
#     Generates a single shared prefix up to a random depth (max 33% of max length),
#     then branches into `num_samples_per_instance` unique suffix trajectories.
#     """
#     import random
#     results_by_root = []
#     temperature = config.gumbeldore_config.get("sampling_temperature", 1.0)
#     autocast_ctx, _ = _make_autocast_ctx(config) if config.use_amp_inference else (nullcontext(), None)
#
#     # DYNAMIC CAPPING MATH:
#     # 3 micro-actions per atom (Atom, Node, Bond).
#     # We want the pivot to happen in the first half of the molecule's construction
#     # to leave plenty of room for branching.
#     max_micro_actions = config.max_num_atoms * 3
#     max_pivot_depth = int(max(1, max_micro_actions // 2))
#
#     for root_node in root_nodes:
#         # --- 1. GENERATE THE SHARED PREFIX (The Pivot) ---
#         prefix_traj = root_node._shallow_clone()
#
#         # Pick a random pivot depth between 0 and the dynamic cap
#         pivot_depth = random.randint(0, max_pivot_depth)
#
#         for _ in range(pivot_depth):
#             if prefix_traj.synthesis_done or prefix_traj.infeasibility_flag:
#                 break
#             with autocast_ctx:
#                 log_probs = log_prob_fn([prefix_traj])[0]
#
#             with np.errstate(divide='ignore'):
#                 probs = np.exp(log_probs / temperature)
#
#             if np.sum(probs) <= 0 or not np.isfinite(probs).all():
#                 prefix_traj.infeasibility_flag = True
#                 break
#
#             probs /= np.sum(probs)
#             action = np.random.choice(len(probs), p=probs)
#
#             # The transition_fn expects the chosen log prob
#             chosen_log_prob = log_probs[action]
#             prefix_traj, _ = prefix_traj.transition_fn(action, chosen_log_prob)
#
#         # --- 2. CLONE THE PREFIX N TIMES ---
#         # If the pivot accidentally finished or broke the molecule early, just return N copies of it.
#         if prefix_traj.synthesis_done or prefix_traj.infeasibility_flag:
#             if not prefix_traj.synthesis_done:
#                 prefix_traj.finalize()
#             results_by_root.append([prefix_traj._shallow_clone() for _ in range(num_samples_per_instance)])
#             continue
#
#         # Create N active clones starting from the exact same pivot state
#         active_trajectories = [prefix_traj._shallow_clone() for _ in range(num_samples_per_instance)]
#         finished_trajectories = []
#
#         # --- 3. ROLLOUT THE SIBLINGS (Standard Batched IID) ---
#         while active_trajectories:
#             with autocast_ctx:
#                 log_probs_list = log_prob_fn(active_trajectories)
#
#             next_active_trajectories = []
#
#             for i, trajectory in enumerate(active_trajectories):
#                 log_probs = log_probs_list[i]
#                 with np.errstate(divide='ignore'):
#                     probs = np.exp(log_probs / temperature)
#
#                 if np.sum(probs) <= 0 or not np.isfinite(probs).all():
#                     trajectory.infeasibility_flag = True
#                     trajectory.finalize()
#                     finished_trajectories.append(trajectory)
#                     continue
#
#                 probs /= np.sum(probs)
#                 action = np.random.choice(len(probs), p=probs)
#                 chosen_log_prob = log_probs_list[i][action]
#                 new_trajectory, is_done = trajectory.transition_fn(action, chosen_log_prob)
#
#                 if is_done:
#                     finished_trajectories.append(new_trajectory)
#                 else:
#                     next_active_trajectories.append(new_trajectory)
#
#             active_trajectories = next_active_trajectories
#
#         results_by_root.append(finished_trajectories)
#
#     return results_by_root


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
                 objective_evaluator: Any  # Typed as Any to accept Proxy or Evaluator
                 ):
        self.config = config
        self.gumbeldore_config = config.gumbeldore_config
        self.objective_evaluator = objective_evaluator
        self.devices_for_workers: List[str] = self.gumbeldore_config["devices_for_workers"]

        self.fragment_library = []
        if config.use_fragment_library:
            with open(config.fragment_library_path, 'r') as f:
                self.fragment_library = [line.strip() for line in f if line.strip()]
            if not self.fragment_library:
                raise FileNotFoundError("Fragment file is empty.")
            print(f"[GRPO] Loaded {len(self.fragment_library)} fragments from {config.fragment_library_path}.")

    def generate_dataset(self, network_weights: dict, central_oracle_handle, best_objective: Optional[float] = None,
                         memory_aggressive: bool = False,
                         custom_prompts: Optional[List[str]] = None,
                         surrogate_model: Optional[Any] = None,
                         surrogate_active: bool = False): # --- NEW ARGUMENT ---
        """
        Parameters:
            surrogate_model: An instance of SurrogateModel (from chem_utils) that implements
                             predict_ranking_scores(smiles_list).
        """

        # # --- NEW CODE: Sanity Check for Tree-GRPO ---
        # if self.config.use_tree_grpo:
        #     search_type = self.gumbeldore_config.get("search_type", "")
        #     if search_type != "wor":
        #         print(f"[Tree-GRPO] WARNING: Tree-GRPO is enabled but search_type is '{search_type}'. "
        #               f"It is highly recommended to use 'wor' (Sampling Without Replacement) to organically build the Trie.")
        # # --------------------------------------------

        batch_size_gpu, batch_size_cpu = (self.gumbeldore_config["batch_size_per_worker"],
                                          self.gumbeldore_config["batch_size_per_cpu_worker"])

        if custom_prompts is not None and len(custom_prompts) > 0:
            # [PMO Strategy] Use specific prompts (e.g. "C" or "Elite Scaffold")
            problem_instances = []
            for smi in custom_prompts:
                try:
                    # do_finish=False treats it as a partial graph (prompt)
                    problem_instances.append(
                        MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                    )
                except Exception as e:
                    print(f"[GRPO] Warning: Failed to load prompt '{smi}'. Fallback to C.")
                    problem_instances.append(MoleculeDesign.get_c_chains(self.config)[0])

            print(f"[GRPO] Using {len(problem_instances)} custom prompts.")

        elif self.config.use_dr_grpo and self.config.use_fragment_library and self.fragment_library:
            # We want one prompt to always be "C", so we sample N-1 from the library
            num_prompts_from_lib = self.config.num_prompts_per_epoch - 1
            if num_prompts_from_lib > 0:
                sampled_smiles_list = random.sample(self.fragment_library, num_prompts_from_lib)
            else:
                sampled_smiles_list = []  # In case num_prompts_per_epoch is 1

            # Manually add the single Carbon atom as a prompt
            sampled_smiles_list.append("C")
            print(f"[GRPO] Sampling {num_prompts_from_lib} fragments + 1 'C' atom prompt.")

            # Create MoleculeDesign objects from these SMILES
            problem_instances = []
            for smi in sampled_smiles_list:
                try:
                    # Our from_smiles function will correctly handle "C"
                    # and turn it into a clean prompt, just like get_c_chains did.
                    problem_instances.append(
                        MoleculeDesign.from_smiles(self.config, smi, do_finish=False)
                    )
                except Exception as e:
                    print(f"[GRPO] Warning: Failed to load fragment '{smi}'. Skipping. Error: {e}")

            if not problem_instances:
                raise ValueError("[GRPO] Error: No valid fragments could be loaded from the fragment library.")

        elif self.config.start_from_c_chains:
            problem_instances = MoleculeDesign.get_c_chains(self.config)
        elif self.config.start_from_smiles is not None:
            problem_instances = [MoleculeDesign.from_smiles(self.config, self.config.start_from_smiles)]
        else:
            problem_instances = MoleculeDesign.get_single_atom_molecules(self.config,
                                                                         repeat=self.config.repeat_start_instances)

        job_pool = JobPool.remote(copy.deepcopy(problem_instances))
        results = [None] * len(problem_instances)

        # Check if we should pin the workers to core
        cpu_cores = [None] * len(self.devices_for_workers)
        if self.gumbeldore_config["pin_workers_to_core"] and sys.platform == "linux":
            # Get available core IDs
            affinity = list(os.sched_getaffinity(0))
            cpu_cores = [affinity[i % len(cpu_cores)] for i in range(len(self.devices_for_workers))]

        # Kick off workers
        # --- PASSING SURROGATE MODEL TO WORKER ---
        future_tasks = [
            async_sbs_worker.remote(
                self.config, job_pool, network_weights, central_oracle_handle, device,
                batch_size_gpu if device != "cpu" else batch_size_cpu,
                cpu_cores[i], best_objective, memory_aggressive,
                surrogate_model=surrogate_model, surrogate_active=surrogate_active
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
            # Check if we should enforce grouping (Local Baseline) or flatten (Global Baseline)
            use_grouping = getattr(self.config, 'use_grpo_grouping', True)

            # --- NEW CODE: Force grouping if Tree-GRPO is active ---
            if self.config.use_tree_grpo:
                if not use_grouping:
                    print(
                        "[GRPO] Warning: use_grpo_grouping was False, but Tree-GRPO is enabled. Forcing grouping to build Tries.")
                use_grouping = True
            # -------------------------------------------------------

            grouped_designs: List[List[MoleculeDesign]] = []
            for group_result in results:  # `group_result` is one List[BeamLeaf] or List[MoleculeDesign]
                if not group_result:
                    continue

                # The worker already unpacks BeamLeaf, but we keep this check just in case
                if isinstance(group_result[0], sbs.BeamLeaf):
                    grouped_designs.append([leaf.state for leaf in group_result])
                else:
                    grouped_designs.append(group_result)  # It's already List[MoleculeDesign]

            if use_grouping:
                if self.config.use_tree_grpo:
                    print("[Tree-GRPO] Grouping ENABLED. Returning groups to build scaffold Tries.")
                return grouped_designs
            else:
                print("[GRPO] Grouping DISABLED. Flattening batch.")
                all_mols = [m for group in grouped_designs for m in group]
                return [all_mols]

        # Supervised / non-RL: original metrics path
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
                        start_atom=molecule.initial_atom,
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

        # Get overall best metrics and molecules
        metrics_return["mean_top_20_obj"] = np.array([x["obj"] for x in merged_mols[:20]]).mean()
        metrics_return["mean_kept_obj"] = np.array([x["obj"] for x in merged_mols]).mean()
        metrics_return["mean_top_20_sa_score"] = np.array([x["sa_score"] for x in merged_mols[:20]]).mean()
        metrics_return["top_20_molecules"] = [{x["smiles"]: x["obj"] for x in merged_mols[:20]}]

        return metrics_return


@ray.remote(max_calls=1)
def async_sbs_worker(config: Config, job_pool: JobPool, network_weights: dict, central_oracle_handle,
                     device: str, batch_size: int,
                     cpu_core: Optional[int] = None,
                     best_objective: Optional[float] = None,
                     memory_aggressive: bool = False,
                     surrogate_model: Optional[Any] = None,
                     surrogate_active: bool = False
                     ):
    network = MoleculeTransformer(config, device)
    network.load_state_dict(network_weights)
    network.to(network.device)
    network.eval()

    autocast_ctx, _ = _make_autocast_ctx(config) if config.use_amp_inference else (nullcontext(), None)

    # Initialize the Proxy to talk to the Central Oracle Actor
    objective_evaluator = RemoteEvaluatorProxy(central_oracle_handle)

    def child_log_probability_fn(trajectories: List[MoleculeDesign]) -> [np.array]:
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

    # --- Helper for Surrogate Filtering ---
    def apply_surrogate_filtering_and_evaluate(candidates: List[MoleculeDesign], num_keep: int):
        """
        1. If surrogate exists & is fitted, rank candidates by predicted probability.
        2. Keep top `num_keep` UNIQUE candidates.
        3. Run Real Oracle (batch_leaf_evaluation_fn) only on the kept ones.
        """
        # If no candidates, return empty
        if not candidates:
            return []

        # If no surrogate or not fitted, behave like standard random/truncate strategy
        if (surrogate_model is None or not surrogate_model.is_fitted) and surrogate_active:
            # Fallback: Just take first N (or shuffle)
            selected = candidates[:num_keep]
            batch_leaf_evaluation_fn(selected)
            return selected
        if not surrogate_active:
            # Surrogate inactive: behave normally (evaluate full beam)
            batch_leaf_evaluation_fn(candidates)
            return candidates

        # 1. Get SMILES
        smiles_list = [c.smiles_string for c in candidates]

        # 2. Predict Ranking Scores (Prob of being good)
        pred_scores = surrogate_model.predict_ranking_scores(smiles_list)

        # 3. Sort candidates by predicted score (descending)
        paired = zip(candidates, pred_scores)
        sorted_paired = sorted(paired, key=lambda x: x[1], reverse=True)

        # 4. Keep Top N UNIQUE Molecules
        selected_candidates = []
        seen_smiles = set()

        for cand, score in sorted_paired:
            if len(selected_candidates) >= num_keep:
                break

            # Check for uniqueness
            smi = cand.smiles_string
            if smi and smi not in seen_smiles:
                seen_smiles.add(smi)
                selected_candidates.append(cand)

        # 5. Evaluate REAL Oracle on selected
        # Note: This consumes PMO budget only for the filtered ones!
        batch_leaf_evaluation_fn(selected_candidates)

        return selected_candidates

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

                # --- MODIFIED: FILTERING ---
                # trajectories_by_root is List[List[MoleculeDesign]]
                # We need to filter each sub-list
                filtered_results_by_root = []
                for traj_list in trajectories_by_root:
                    # Filter down to 'num_trajectories_to_keep'
                    keep_k = config.gumbeldore_config["num_trajectories_to_keep"]
                    filtered_list = apply_surrogate_filtering_and_evaluate(traj_list, keep_k)
                    filtered_results_by_root.append(filtered_list)

                # Group results for pushing back to the job pool
                results_to_push = [(idx, trajectories) for idx, trajectories in zip(idx_list, filtered_results_by_root)]

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
                    # result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j]]
                    raw_result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j]]

                    # --- MODIFIED: FILTERING ---
                    keep_k = config.gumbeldore_config["num_trajectories_to_keep"]

                    # We have `beam_width` candidates (e.g. 100). We only want to eval top 10.
                    filtered_result = apply_surrogate_filtering_and_evaluate(raw_result, keep_k)

                    results_to_push.append((result_idx, filtered_result))

            else:  # tasar, wor

                # --- NEW LOGIC FOR WOR: Defer Oracle Calls ---
                if config.gumbeldore_config["search_type"] == "wor":
                    # WOR calls `batch_leaf_evaluation_fn` internally inside IncrementalSBS.
                    # To support "Fake Oracle", we replace this with a dummy function that
                    # returns Zeros and does not set .objective.
                    # We also must provide a safe sorting key because .objective will be None.
                    eval_fn_for_sbs = lambda trajs: np.zeros(len(trajs))
                    leaf_eval_fn_for_sbs = lambda state: (state.objective if state.objective is not None else -float('inf'))
                else:
                    # TASAR requires real feedback
                    eval_fn_for_sbs = batch_leaf_evaluation_fn
                    leaf_eval_fn_for_sbs = MoleculeDesign.to_max_evaluation_fn

                inc_sbs = IncrementalSBS(root_nodes, child_log_probability_fn, child_transition_fn,
                                         leaf_evaluation_fn=leaf_eval_fn_for_sbs,
                                         batch_leaf_evaluation_fn=eval_fn_for_sbs,
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
                    if config.gumbeldore_config["search_type"] == "wor":
                        # WOR typically returns the whole beam (e.g. 100)
                        raw_result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j]]
                    elif config.gumbeldore_config["search_type"] == "tasar":
                        # TASAR slicing kept as is
                        raw_result: List[MoleculeDesign] = [x.state for x in beam_leaves_batch[j][
                            :config.gumbeldore_config["num_trajectories_to_keep"]]]

                    # --- MODIFIED: FILTERING (Corrected for WOR) ---
                    # Only evaluate if objective is None (which happens if we used the dummy evaluator)
                    if raw_result and raw_result[0].objective is None:
                        keep_k = config.gumbeldore_config["num_trajectories_to_keep"]
                        # Filter down to keep_k and evaluate
                        result = apply_surrogate_filtering_and_evaluate(raw_result, keep_k)
                    else:
                        result = raw_result

                    results_to_push.append((result_idx, result))

            ray.get(job_pool.push_results.remote(results_to_push))

            if device != "cpu":
                torch.cuda.empty_cache()

    del network
    del network_weights
    torch.cuda.empty_cache()