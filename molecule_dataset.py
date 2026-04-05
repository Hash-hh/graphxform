import math
from typing import Optional, Tuple, List

import torch
import pickle
import random
import numpy as np
from torch.utils.data import Dataset

from config import MoleculeConfig
from molecule_design import MoleculeDesign


class RandomMoleculeDataset(Dataset):
    """
    Dataset for supervised training of the molecule design given as a list pseudo-expert molecules.
    Each molecule is given as a dictionary with the following keys and values:
          "task_type": [str] "additive", "removal", or "replacement"
          "start_atom": [int] or None
          "prompt_smiles": [str] or None
          "action_seq": List[int] Actions taken on each index to create/edit the molecule
          "smiles": [str] Corresponding target smiles string
          "obj": [float] Objective function evaluation
    """

    def __init__(self, config: MoleculeConfig, path_to_pickle: str, batch_size: int, custom_num_batches: Optional[int],
                 no_random: bool = False):
        self.config = config
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.path_to_pickle = path_to_pickle
        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f)  # list of dictionaries

        # We want to uniformly sample from partial molecules.
        self.targets_to_sample: List[Tuple[int, int]] = []

        for i, instance in enumerate(self.instances):
            sequence_of_actions_idx = list(range(len(instance["action_seq"])))
            self.targets_to_sample.extend([(i, j) for j in sequence_of_actions_idx])

        print(f"Loaded dataset: {path_to_pickle}")
        print(f" -> {len(self.instances)} molecules with a total of {len(self.targets_to_sample)} datapoints.")

        if custom_num_batches is None:
            self.length = len(self.targets_to_sample) // self.batch_size  # one item is a batch of datapoints.
        else:
            self.length = custom_num_batches

        self.no_random = no_random

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        :param idx: is not used, as we directly randomly sample a full batch from the datapoints here.
        Returns: Dictionary with keys: input, target_zero, target_one, target_two
        """
        partial_molecules: List[MoleculeDesign] = []  # partial molecules which will become the batch
        instance_targets: List[int] = []  # corresponding targets taken from the instances

        if self.no_random:
            batch_to_pick = self.targets_to_sample[idx * self.batch_size: (idx + 1) * self.batch_size]
        else:
            batch_to_pick = random.choices(self.targets_to_sample, k=self.batch_size)  # with replacement

        for instance_idx, target_idx in batch_to_pick:
            instance = self.instances[instance_idx]
            task_type = instance.get("task_type", "additive")

            # --- MULTI-TASK FIX: Load the correct starting state ---
            if task_type == "additive":
                molecule = MoleculeDesign(self.config, initial_atom=instance["start_atom"])
            else:
                molecule = MoleculeDesign.from_smiles(self.config, instance["prompt_smiles"])

            # Prevent premature termination for large molecules during playback
            molecule.max_actions = 1000
            # --------------------------------------------------------

            # create molecule up to (excluding) target actions
            for action in instance["action_seq"][:target_idx]:
                molecule.take_action(action)

            partial_molecules.append(molecule)
            instance_targets.append(instance["action_seq"][target_idx])

        # --- SIGNATURE FIX: Format as list of dictionaries ---
        list_of_samples = [{'molecule': mol} for mol in partial_molecules]

        batch_input = MoleculeDesign.list_to_batch(list_of_samples=list_of_samples,
                                                   device=torch.device("cpu"))
        # -----------------------------------------------------

        # We now create the targets. We separate it into targets for level 0, 1 and 2.
        batch_targets = [
            torch.LongTensor([target if partial_molecules[i].current_action_level == level else -1 for i, target in
                              enumerate(instance_targets)])  # (B,)
            for level in [0, 1, 2]
        ]

        return dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1],
            target_two=batch_targets[2]
        )