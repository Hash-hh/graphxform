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
    Each molecule is given as a dictionary with the following keys and values
          "start_atom": [int] the int representing the atom from which to start
          "action_seq": List[List[int]] Actions which need to be taken on each index to create the molecule
          "smiles": [str] Corresponding smiles string
          "obj": [float] Objective function evaluation

    Each datapoint in this dataset is a partial molecule: We sample an instance, randomly choose an index up to which
    all actions will be performed. Then, ending up at action index 0, we take the next item in the action seq
    (which corresponds to a list all actions that need to be taken from index to index) as training target.
    As the number of atoms will be different for molecules in a batch, we pad the atoms, and set all labels corresponding
    to the padded atoms to -1 (in the CE-loss, this will be specified as `ignore_index=-1`.
    """
    def __init__(self, config: MoleculeConfig, path_to_pickle: str, batch_size: int, custom_num_batches: Optional[int],
                 no_random: bool = False):
        self.config = config
        self.batch_size = batch_size
        self.custom_num_batches = custom_num_batches
        self.path_to_pickle = path_to_pickle
        with open(path_to_pickle, "rb") as f:
            self.instances = pickle.load(f)  # list of dictionaries

        # We want to uniformly sample from partial molecules. So for each instance, check how many partial molecules
        # there are, and create a list of them where each entry is a tuple (int, int), where first entry is index of
        # the instance, and second entry is the index in the action sequence which is the training target.
        self.targets_to_sample: List[Tuple[int, int]] = []

        for i, instance in enumerate(self.instances):
            sequence_of_actions_idx = list(range(len(instance["action_seq"])))
            self.targets_to_sample.extend([(i, j) for j in sequence_of_actions_idx])

        print(f"Loaded dataset. {len(self.instances)} molecules with a total of {len(self.targets_to_sample)} datapoints.")

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

        Returns: Dictionary with keys:

        """
        partial_molecules: List[MoleculeDesign] = []   # partial molecules which will become the batch
        instance_targets: List[List[int]] = []  # corresponding targets taken from the instances

        if self.no_random:
            batch_to_pick = self.targets_to_sample[idx * self.batch_size: (idx+1) * self.batch_size]
        else:
            batch_to_pick = random.choices(self.targets_to_sample, k=self.batch_size)  # with replacement
        for instance_idx, target_idx in batch_to_pick:
            instance = self.instances[instance_idx]
            # Build up the molecule
            molecule = MoleculeDesign(self.config, initial_atom=instance["start_atom"])
            # create molecule up to (excluding) target actions
            for action in instance["action_seq"][:target_idx]:
                molecule.take_action(action)
            partial_molecules.append(molecule)
            instance_targets.append(instance["action_seq"][target_idx])

        # Create the input batch from the partial molecules.
        batch_input = MoleculeDesign.list_to_batch(molecules=partial_molecules,
                                                   device=torch.device("cpu"),
                                                   include_feasibility_masks=True)

        # We now create the targets. We separate it into targets for level 0, 1 and 2.
        # We only set the target action as target for the current level the molecule is in.
        # For all other levels, we set it to -1 for a molecule. (ignore)
        batch_targets = [
            torch.LongTensor([target if partial_molecules[i].current_action_level == level else -1 for i, target in
                              enumerate(instance_targets)])  # (B,)
            for level in [0,1,2]
        ]

        return dict(
            input=batch_input,
            target_zero=batch_targets[0],
            target_one=batch_targets[1],
            target_two=batch_targets[2]
        )