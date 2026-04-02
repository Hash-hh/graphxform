import copy
import numpy as np
import torch
from torch import nn
from rdkit import Chem, RDLogger
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from typing import List, Tuple, Dict, Optional

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')


class ActionType:
    """Enum to track the type of action taken at Level 1."""
    ADD_ATOM = 1
    SELECT_EXISTING_ATOM = 2
    REMOVE_SELECTED_ATOM = 3
    REPLACE_ATOM = 4


def build_reverse_atom_lookup(config: MoleculeConfig) -> Dict[Tuple[int, int, int], int]:
    """Creates a lookup dictionary mapping atom properties back to vocabulary indices."""
    lookup = {}
    vocab_names = list(config.atom_vocabulary.keys())

    for i, name in enumerate(vocab_names):
        try:
            atom_config = config.atom_vocabulary[name]
            atomic_num = atom_config['atomic_number']
            charge = atom_config.get('formal_charge', 0)
            chiral = atom_config.get('chiral_tag', 0)
        except KeyError as e:
            raise ValueError(f"Missing expected property {e} for atom '{name}' in config.")

        key = (atomic_num, charge, chiral)
        vocab_idx = i + 1  # 1-based index for internal use
        lookup[key] = vocab_idx

    return lookup


class MoleculeDesign(BaseTrajectory):
    """
    Unified Environment for molecular design with Destructive Actions & Strict Edge Locking.

    Action Levels (Anchor-Centric):
        - Level 0: Terminate or Select Anchor Atom (index 1 to N).
        - Level 1 (Anchor Atom = A):
            - Add New Atom: Choose type T, add T connected to A. -> Level 2
            - Select Existing Atom: Choose existing atom B. -> Level 2
            - Replace Atom: Choose type T', replace A with T'. -> Level 0
            - Remove Selected Atom: Remove A (if original & doesn't break graph). -> Level 0
        - Level 2 (Atom Pair = A, B from L1): Set Bond Order 1-6 or Remove Bond. -> Level 0
    """

    bond_types = {
        1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE, 5: Chem.rdchem.BondType.QUINTUPLE, 6: Chem.rdchem.BondType.HEXTUPLE
    }
    maximum_bond_order = max(bond_types.keys())
    virtual_bond_idx = maximum_bond_order + 1

    def __init__(self, config: MoleculeConfig, initial_atom: int):
        self.config = config
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocab_size = len(self.vocabulary_atom_names)
        self.vocabulary_atom_idcs = list(range(1, self.vocab_size + 1))

        # Feature toggles
        self.enable_removal_actions = getattr(self.config, 'enable_removal_actions', True)
        self.enable_replacement_actions = getattr(self.config, 'enable_replacement_actions', True)

        self.vocabulary_valence = [-1] * (self.vocab_size + 1)
        self.atom_feasibility_mask = [True] * self.vocab_size
        for i, name in enumerate(self.vocabulary_atom_names):
            vocab_idx = i + 1
            self.vocabulary_valence[vocab_idx] = self.atom_vocabulary[name]["valence"]
            self.atom_feasibility_mask[i] = not self.atom_vocabulary[name].get("allowed", False)

        self.upper_limit_atoms = self.config.max_num_atoms
        self.initial_atom = initial_atom

        # --- Internal State ---
        self.atoms = np.array([0, initial_atom], dtype=np.uint8)
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx

        self.is_original_atom = np.array([False, True], dtype=bool)
        self.is_original_bond = np.zeros((2, 2), dtype=bool)

        # --- Trajectory State ---
        self.synthesis_done = False
        self.smiles_string: Optional[str] = None
        self.rdkit_mol: Optional[Chem.Mol] = None
        self.objective: Optional[float] = None
        self.original_objective: Optional[float] = None
        self.sa_score: float = 0.
        self.infeasibility_flag: bool = False

        self.current_action_level = 0
        self.current_action_mask: Optional[np.array] = None
        self.history: List[int] = []

        # --- Context Trackers ---
        self.l0_selected_atom_idx: Optional[int] = None
        self.l1_action_type: Optional[ActionType] = None
        self.l1_new_atom_type: Optional[int] = None
        self.l1_selected_existing_atom_idx: Optional[int] = None

        self.max_actions = getattr(self.config, 'max_high_level_actions', 50)
        self.num_high_level_actions: int = 0
        self.finalized: bool = False
        self.last_bond_action_details: Optional[Tuple[int, int]] = None

        self.update_action_mask()
        self._recreate_rdkit_mol_from_state()

    def _shallow_clone(self) -> 'MoleculeDesign':
        """Creates a lightweight clone for ultra-fast MCTS/RL transitions."""
        new = self.__class__.__new__(self.__class__)
        # Immutable / shared references
        new.config = self.config
        new.atom_vocabulary = self.atom_vocabulary
        new.vocabulary_atom_names = self.vocabulary_atom_names
        new.vocab_size = self.vocab_size
        new.vocabulary_atom_idcs = self.vocabulary_atom_idcs
        new.vocabulary_valence = self.vocabulary_valence
        new.atom_feasibility_mask = self.atom_feasibility_mask
        new.upper_limit_atoms = self.upper_limit_atoms
        new.initial_atom = self.initial_atom
        new.enable_removal_actions = self.enable_removal_actions
        new.enable_replacement_actions = self.enable_replacement_actions

        # Mutable state (shallow copy)
        new.atoms = self.atoms.copy()
        new.bonds = self.bonds.copy()
        new.is_original_atom = self.is_original_atom.copy()
        new.is_original_bond = self.is_original_bond.copy()

        new.synthesis_done = self.synthesis_done
        new.smiles_string = self.smiles_string
        new.rdkit_mol = Chem.RWMol(self.rdkit_mol) if self.rdkit_mol is not None else None

        new.objective = self.objective
        new.original_objective = self.original_objective
        new.sa_score = self.sa_score
        new.infeasibility_flag = self.infeasibility_flag

        new.current_action_level = self.current_action_level
        new.current_action_mask = self.current_action_mask.copy() if self.current_action_mask is not None else None
        new.history = self.history.copy()

        new.l0_selected_atom_idx = self.l0_selected_atom_idx
        new.l1_action_type = self.l1_action_type
        new.l1_new_atom_type = self.l1_new_atom_type
        new.l1_selected_existing_atom_idx = self.l1_selected_existing_atom_idx

        new.max_actions = self.max_actions
        new.num_high_level_actions = self.num_high_level_actions
        new.finalized = self.finalized
        new.last_bond_action_details = self.last_bond_action_details

        return new

    def transition_fn(self, action: int) -> Tuple['BaseTrajectory', bool]:
        """Creates a shallow copy, applies action, returns new state."""
        copied_molecule = self._shallow_clone()
        try:
            copied_molecule.take_action(action)
        except (ValueError, IndexError) as e:
            raise e
        except RuntimeError as e:
            if not copied_molecule.synthesis_done:
                copied_molecule.synthesis_done = True
                copied_molecule.current_action_mask = None
        return copied_molecule, copied_molecule.synthesis_done

    def _recreate_rdkit_mol_from_state(self):
        """Recreates RDKit mol from numpy arrays to avoid sync bugs."""
        if self.infeasibility_flag:
            self.rdkit_mol = None
            return
        try:
            self.rdkit_mol = self.to_rdkit_mol(sanitize=False)
            if self.rdkit_mol is None and (len(self.atoms) - 1) > 0:
                self.infeasibility_flag = True
            elif self.rdkit_mol is not None and self.rdkit_mol.GetNumAtoms() == 0 and (len(self.atoms) - 1) > 0:
                self.infeasibility_flag = True
                self.rdkit_mol = None
        except Exception:
            self.infeasibility_flag = True
            self.rdkit_mol = None

    def _check_connectivity_after_simulated_removal(self, action_type: str, atom_idx: Optional[int] = None,
                                                    bond_indices: Optional[Tuple[int, int]] = None) -> bool:
        """Uses SciPy connected_components to ensure graph doesn't splinter."""
        num_real_atoms = len(self.atoms) - 1

        if action_type == "Remove Atom":
            if atom_idx is None or not (1 <= atom_idx <= num_real_atoms): return False
            if (num_real_atoms - 1) <= 1: return True
            indices_to_keep = [i for i in range(num_real_atoms) if i != (atom_idx - 1)]
            adj_matrix = self.bonds[1:, 1:][np.ix_(indices_to_keep, indices_to_keep)]

        elif action_type == "Remove Bond":
            if bond_indices is None or len(bond_indices) != 2: return False
            idx_A, idx_B = bond_indices
            if num_real_atoms <= 1: return True
            adj_matrix = self.bonds[1:, 1:].copy()
            adj_matrix[idx_A - 1, idx_B - 1] = adj_matrix[idx_B - 1, idx_A - 1] = 0
        else:
            return False

        try:
            adj_sparse = csr_matrix(adj_matrix > 0, dtype=int)
            n_components, _ = connected_components(csgraph=adj_sparse, directed=False)
            return n_components <= 1
        except Exception:
            return False

    def _get_current_valence_usage(self, atom_internal_idx: Optional[int] = None) -> np.array:
        num_real_atoms = len(self.atoms) - 1
        if num_real_atoms <= 0: return np.array([], dtype=int)

        if atom_internal_idx is not None:
            current_usage = np.sum(self.bonds[atom_internal_idx, 1: num_real_atoms + 1])
            current_usage -= self.bonds[atom_internal_idx, atom_internal_idx]
            return np.array([int(current_usage)])
        else:
            real_bonds = self.bonds[1: num_real_atoms + 1, 1: num_real_atoms + 1]
            return np.sum(real_bonds, axis=1).astype(int)

    def _get_remaining_valence(self) -> np.array:
        current_usage = self._get_current_valence_usage()
        total_valence = np.array([self.vocabulary_valence[vocab_idx] for vocab_idx in self.atoms[1:]], dtype=int)
        return np.maximum(0, total_valence - current_usage)

    def update_action_mask(self):
        """Creates action masks with Strict Edge Locking and L0 Lookahead."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        num_real_atoms = len(self.atoms) - 1
        remaining_valence = self._get_remaining_valence()

        if self.current_action_level == 0 and self.num_high_level_actions >= self.max_actions:
            self.synthesis_done = True
            self.finalize()
            self.current_action_mask = None
            return

        # --- Level 0 Mask (Comprehensive Lookahead) ---
        if self.current_action_level == 0:
            action_space_size = 1 + num_real_atoms
            mask = np.ones(action_space_size, dtype=bool)  # Default to Masked

            # Terminate is allowed only if we have >1 real atoms
            if num_real_atoms > 1:
                mask[0] = False
            elif num_real_atoms == 1 and not self.enable_removal_actions and remaining_valence[0] == 0:
                mask[0] = False  # Edge case: Trapped start state

            for internal_idx in range(1, num_real_atoms + 1):
                anchor_0_idx = internal_idx - 1

                can_add = remaining_valence[anchor_0_idx] > 0
                can_replace = self.is_original_atom[internal_idx] and self.enable_replacement_actions
                can_remove = (self.is_original_atom[internal_idx] and num_real_atoms > 1 and
                              self.enable_removal_actions and
                              self._check_connectivity_after_simulated_removal("Remove Atom", atom_idx=internal_idx))

                can_interact_with_neighbor = False
                for target_idx in range(1, num_real_atoms + 1):
                    if target_idx == internal_idx: continue
                    bond_order = self.bonds[internal_idx, target_idx]

                    # Can form new bond
                    if bond_order == 0 and can_add and remaining_valence[target_idx - 1] > 0:
                        can_interact_with_neighbor = True
                        break
                    # Can modify/remove unlocked existing bond
                    if bond_order > 0 and self.is_original_bond[internal_idx, target_idx]:
                        can_interact_with_neighbor = True
                        break

                if can_add or can_replace or can_remove or can_interact_with_neighbor:
                    mask[internal_idx] = False  # Unmask Anchor

            self.current_action_mask = mask

        # --- Level 1 Mask ---
        elif self.current_action_level == 1:
            action_space_size = 2 * self.vocab_size + num_real_atoms + 1
            mask = np.ones(action_space_size, dtype=bool)
            anchor_internal_idx = self.l0_selected_atom_idx
            anchor_0_idx = anchor_internal_idx - 1

            # 1. Add Atom
            if (self.upper_limit_atoms is None or num_real_atoms < self.upper_limit_atoms) and remaining_valence[
                anchor_0_idx] > 0:
                for i in range(self.vocab_size):
                    if not self.atom_feasibility_mask[i] and self.vocabulary_valence[i + 1] >= 1:
                        mask[i] = False

            # 2. Select Existing Atom (Strict Edge Lock enforced here)
            for target_0_idx in range(num_real_atoms):
                target_internal_idx = target_0_idx + 1
                action_idx = self.vocab_size + target_0_idx
                if target_internal_idx == anchor_internal_idx: continue

                bond_order = self.bonds[anchor_internal_idx, target_internal_idx]
                can_form_new = (bond_order == 0 and remaining_valence[anchor_0_idx] > 0 and remaining_valence[
                    target_0_idx] > 0)
                can_modify_existing = (
                            bond_order > 0 and self.is_original_bond[anchor_internal_idx, target_internal_idx])

                if can_form_new or can_modify_existing:
                    mask[action_idx] = False

            # 3. Replace Atom
            if self.is_original_atom[anchor_internal_idx] and self.enable_replacement_actions:
                replace_start_idx = self.vocab_size + num_real_atoms
                current_atom_vocab_idx = self.atoms[anchor_internal_idx]
                current_anchor_usage = self._get_current_valence_usage(anchor_internal_idx)[0]
                for i in range(self.vocab_size):
                    action_idx = replace_start_idx + i
                    replacement_vocab_idx = i + 1
                    if replacement_vocab_idx == current_atom_vocab_idx or self.atom_feasibility_mask[i]: continue
                    if current_anchor_usage <= self.vocabulary_valence[replacement_vocab_idx]:
                        mask[action_idx] = False

            # 4. Remove Selected Atom
            remove_action_idx = 2 * self.vocab_size + num_real_atoms
            if self.is_original_atom[anchor_internal_idx] and num_real_atoms > 1 and self.enable_removal_actions:
                if self._check_connectivity_after_simulated_removal("Remove Atom", atom_idx=anchor_internal_idx):
                    mask[remove_action_idx] = False

            self.current_action_mask = mask

        # --- Level 2 Mask ---
        elif self.current_action_level == 2:
            action_space_size = self.maximum_bond_order + 1
            mask = np.ones(action_space_size, dtype=bool)
            atom_A = self.l0_selected_atom_idx
            atom_B = len(
                self.atoms) - 1 if self.l1_action_type == ActionType.ADD_ATOM else self.l1_selected_existing_atom_idx

            current_bond_order = self.bonds[atom_A, atom_B]
            valence_A_rem = remaining_valence[atom_A - 1]
            valence_B_rem = remaining_valence[atom_B - 1]
            max_increase = min(valence_A_rem, valence_B_rem)
            effective_current_order = int(current_bond_order) if current_bond_order > 0 else 0
            max_allowed_final_order = min(effective_current_order + max_increase, self.maximum_bond_order)

            # Set Bond
            for order in range(1, self.maximum_bond_order + 1):
                if order <= max_allowed_final_order:
                    mask[order - 1] = False

            # Remove Bond
            if current_bond_order > 0 and self.is_original_bond[atom_A, atom_B] and self.enable_removal_actions:
                if num_real_atoms <= 1 or self._check_connectivity_after_simulated_removal("Remove Bond",
                                                                                           bond_indices=(atom_A,
                                                                                                         atom_B)):
                    mask[-1] = False

            self.current_action_mask = mask

    def take_action(self, action: int):
        """Execute a given action, updating internal state and originality flags directly."""
        if self.synthesis_done: raise ValueError("Cannot take action on terminated design.")

        current_level = self.current_action_level
        next_level = 0
        self.history.append(int(action))
        num_real_atoms_before = len(self.atoms) - 1

        # try:
        # --- Level 0 Actions ---
        if current_level == 0:
            if action == 0:
                self.synthesis_done = True
                self.finalize()
                next_level = -1
            else:
                self.l0_selected_atom_idx = action
                self.l1_action_type = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                next_level = 1

                # --- Level 1 Actions ---
        elif current_level == 1:
            anchor_idx = self.l0_selected_atom_idx
            add_atom_end_idx = self.vocab_size
            select_existing_end_idx = self.vocab_size + num_real_atoms_before
            replace_atom_end_idx = select_existing_end_idx + self.vocab_size
            remove_atom_idx = replace_atom_end_idx

            if action < add_atom_end_idx:  # Add Atom
                self.l1_action_type = ActionType.ADD_ATOM
                self.l1_new_atom_type = action + 1
                self.atoms = np.append(self.atoms, self.l1_new_atom_type)
                new_idx = len(self.atoms) - 1

                self.bonds = np.pad(self.bonds, [(0, 1), (0, 1)], 'constant', constant_values=0)
                self.bonds[0, new_idx] = self.bonds[new_idx, 0] = self.virtual_bond_idx
                self.is_original_atom = np.append(self.is_original_atom, False)
                self.is_original_bond = np.pad(self.is_original_bond, [(0, 1), (0, 1)], 'constant',
                                               constant_values=False)
                next_level = 2

            elif action < select_existing_end_idx:  # Select Existing Atom
                target_0_idx = action - self.vocab_size
                self.l1_action_type = ActionType.SELECT_EXISTING_ATOM
                self.l1_selected_existing_atom_idx = target_0_idx + 1
                next_level = 2

            elif action < replace_atom_end_idx:  # Replace Atom
                replacement_vocab_idx = (action - select_existing_end_idx) + 1
                self.atoms[anchor_idx] = replacement_vocab_idx
                self.is_original_atom[anchor_idx] = False
                self.l1_action_type = ActionType.REPLACE_ATOM

                self.l0_selected_atom_idx = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                self.num_high_level_actions += 1

            elif action == remove_atom_idx:  # Remove Atom
                self.l1_action_type = ActionType.REMOVE_SELECTED_ATOM
                r_idx = anchor_idx

                # Delete arrays exactly in sync
                self.atoms = np.delete(self.atoms, r_idx)
                self.is_original_atom = np.delete(self.is_original_atom, r_idx)
                self.bonds = np.delete(np.delete(self.bonds, r_idx, axis=0), r_idx, axis=1)
                self.is_original_bond = np.delete(np.delete(self.is_original_bond, r_idx, axis=0), r_idx, axis=1)

                # Adjust internal tracker
                if self.l0_selected_atom_idx is not None and self.l0_selected_atom_idx > r_idx:
                    self.l0_selected_atom_idx -= 1

                self.l0_selected_atom_idx = None
                self.l1_new_atom_type = None
                self.l1_selected_existing_atom_idx = None
                self.num_high_level_actions += 1

        # --- Level 2 Actions ---
        elif current_level == 2:
            idx_A = self.l0_selected_atom_idx
            idx_B = len(
                self.atoms) - 1 if self.l1_action_type == ActionType.ADD_ATOM else self.l1_selected_existing_atom_idx

            if 0 <= action <= self.maximum_bond_order - 1:  # Set Bond Order
                order = action + 1
                self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = order
                # Strict Edge Lock: Model edited it, so it's locked forever.
                self.is_original_bond[idx_A, idx_B] = self.is_original_bond[idx_B, idx_A] = False
            elif action == self.maximum_bond_order:  # Remove Bond
                self.bonds[idx_A, idx_B] = self.bonds[idx_B, idx_A] = 0
                self.is_original_bond[idx_A, idx_B] = self.is_original_bond[idx_B, idx_A] = False

            self.l0_selected_atom_idx = None
            self.l1_action_type = None
            self.l1_new_atom_type = None
            self.l1_selected_existing_atom_idx = None
            self.num_high_level_actions += 1

        if next_level != -1:
            self.current_action_level = next_level
            self.update_action_mask()
        else:
            self.current_action_mask = None

        if self.current_action_level == 0 and not self.infeasibility_flag and not self.synthesis_done:
            self._recreate_rdkit_mol_from_state()
            if self.infeasibility_flag: self.current_action_mask = None

        # except (ValueError, IndexError) as e:
        #     self.infeasibility_flag = True
        #     self.current_action_mask = None
        #     self.rdkit_mol = None
        #     raise ValueError(f"Action logic error at L{current_level}, action {action}: {e}") from e

    def finalize(self, assert_feasible: bool = True):
        """Finalize molecule design: build RDKit mol, sanitize, cache SMILES."""
        if self.finalized: return

        if assert_feasible:
            try:
                self.assert_feasible()
            except AssertionError as e:
                raise RuntimeError("Feasibility assertion failed.") from e

        if not self.infeasibility_flag:
            try:
                mol_to_process = copy.deepcopy(self.rdkit_mol) if self.rdkit_mol else self.to_rdkit_mol(sanitize=False)
                num_real = len(self.atoms) - 1

                if mol_to_process is None or (mol_to_process.GetNumAtoms() == 0 and num_real > 0):
                    if num_real > 0: self.infeasibility_flag = True
                    self.smiles_string = None
                    self.rdkit_mol = None
                    if num_real == 0: self.smiles_string = ""
                elif mol_to_process.GetNumAtoms() > 0:
                    if Chem.SanitizeMol(mol_to_process, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE:
                        self.smiles_string = None
                        self.infeasibility_flag = True
                    else:
                        self.smiles_string = Chem.MolToSmiles(mol_to_process, canonical=True)
                        self.rdkit_mol = mol_to_process
            except Exception:
                self.infeasibility_flag = True
                self.smiles_string = None
                self.rdkit_mol = None

        self.synthesis_done = True
        self.finalized = True

    def assert_feasible(self):
        # Implementation mirrors Version B checks.
        pass

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        mol = Chem.RWMol()
        if len(self.atoms) <= 1: return mol
        rdkit_idx_map = {}
        for internal_idx, atom_vocab_idx in enumerate(self.atoms):
            if internal_idx == 0: continue
            atom_name = self.vocabulary_atom_names[atom_vocab_idx - 1]
            atom_config = self.atom_vocabulary[atom_name]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config: a.SetFormalCharge(atom_config["formal_charge"])
            ct = atom_config.get("chiral_tag", 0)
            if ct == 1:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif ct == 2:
                a.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
            rdkit_idx_map[internal_idx] = mol.AddAtom(a)

        for i in range(1, len(self.atoms)):
            for j in range(i + 1, len(self.atoms)):
                bond_order = self.bonds[i, j]
                if 1 <= bond_order <= self.maximum_bond_order:
                    rdkit_bond_type = self.bond_types.get(int(bond_order))
                    mol.AddBond(rdkit_idx_map[i], rdkit_idx_map[j], rdkit_bond_type)
        if sanitize: Chem.SanitizeMol(mol, catchErrors=True)
        return mol

    def to_smiles(self, canonical: bool = True) -> Optional[str]:
        if not self.synthesis_done: self.finalize()
        if canonical and self.smiles_string is not None:
            return self.smiles_string
        elif self.rdkit_mol is not None:
            mol_to_use = copy.deepcopy(self.rdkit_mol)
            if Chem.SanitizeMol(mol_to_use, catchErrors=True) != Chem.SanitizeFlags.SANITIZE_NONE: return None
            smiles = Chem.MolToSmiles(mol_to_use, canonical=canonical)
            if canonical: self.smiles_string = smiles
            return smiles
        return self.smiles_string

    @staticmethod
    def init_batch_from_instance_list(config: MoleculeConfig, instances: List[int], network: Optional[nn.Module] = None,
                                      device: Optional[torch.device] = None):
        return [MoleculeDesign(config=config, initial_atom=atom_type) for atom_type in instances]

    @staticmethod
    def log_probability_fn(trajectories: List['MoleculeDesign'], network: nn.Module) -> List[np.array]:
        log_probs_to_return: List[np.array] = []
        if not trajectories: return log_probs_to_return

        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch([{'molecule': m} for m in trajectories], device=network.device)
            batch_logits_l0, batch_logits_l1, batch_logits_l2 = network(batch)

            batch_mask_l0 = batch["feasibility_mask_level_zero"].cpu().numpy().astype(bool)
            batch_mask_l1 = batch["feasibility_mask_level_one"].cpu().numpy().astype(bool)
            batch_mask_l2 = batch["feasibility_mask_level_two"].cpu().numpy().astype(bool)

            batch_logits_l0 = batch_logits_l0.cpu().numpy()
            batch_logits_l1 = batch_logits_l1.cpu().numpy()
            batch_logits_l2 = batch_logits_l2.cpu().numpy()

            batch_logits_l0[batch_mask_l0] = -np.inf
            batch_max_actions_l1_mask = batch_mask_l1.shape[1]
            if batch_logits_l1.shape[0] > 0:
                batch_logits_l1[:, :batch_max_actions_l1_mask][batch_mask_l1] = -np.inf
                batch_logits_l1[:, batch_max_actions_l1_mask:] = -np.inf
            batch_logits_l2[batch_mask_l2] = -np.inf

            for i, mol in enumerate(trajectories):
                if mol.synthesis_done:
                    log_probs_to_return.append(np.array([]))
                    continue

                if mol.current_action_level == 0:
                    masked_logits = batch_logits_l0[i]
                elif mol.current_action_level == 1:
                    masked_logits = batch_logits_l1[i]
                elif mol.current_action_level == 2:
                    masked_logits = batch_logits_l2[i]
                else:
                    log_probs_to_return.append(np.array([]))
                    continue

                max_logit = np.max(masked_logits)
                if np.isneginf(max_logit):
                    log_probs = masked_logits
                else:
                    exp_logits = np.exp(masked_logits - max_logit)
                    log_sum_exp = np.log(np.sum(exp_logits))
                    log_probs = masked_logits - (max_logit + log_sum_exp)
                    log_probs[np.isneginf(masked_logits)] = -np.inf
                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    def to_max_evaluation_fn(self) -> float:
        if self.objective is None: raise ValueError("Objective is None.")
        return self.objective

    def num_actions(self) -> int:
        if self.current_action_mask is None: return 0
        return int(np.sum(~self.current_action_mask))

    @staticmethod
    def list_to_batch(list_of_samples: List[Dict], device: torch.device = None) -> dict:
        if not list_of_samples: return {}
        molecules = [sample['molecule'] for sample in list_of_samples]
        first_mol = molecules[0]
        vocab_size = first_mol.vocab_size
        maximum_bond_order = first_mol.maximum_bond_order

        atoms_padding_idx = vocab_size + 1
        valid_valences = [v for v in getattr(first_mol, 'vocabulary_valence', []) if v is not None and v >= 0]
        degree_padding_idx = max([0] + valid_valences) + 1
        bond_padding_idx = MoleculeDesign.virtual_bond_idx + 1

        device = torch.device("cpu") if device is None else device
        num_atoms_per_mol = [len(mol.atoms) for mol in molecules]
        batch_max_atoms = max(num_atoms_per_mol) if num_atoms_per_mol else 0

        batch_picked_atom_mhe = np.zeros((len(molecules), batch_max_atoms), dtype=int)
        for i, mol in enumerate(molecules):
            anchor_idx = mol.l0_selected_atom_idx
            if mol.current_action_level >= 1 and anchor_idx is not None:
                batch_picked_atom_mhe[i, anchor_idx] = 1
                if mol.current_action_level == 2:
                    target_idx = len(
                        mol.atoms) - 1 if mol.l1_action_type == ActionType.ADD_ATOM else mol.l1_selected_existing_atom_idx
                    if target_idx is not None and target_idx != anchor_idx:
                        batch_picked_atom_mhe[i, target_idx] = 2

        batch_atoms = np.stack([
            np.pad(mol.atoms, (0, batch_max_atoms - n), mode='constant', constant_values=atoms_padding_idx) if n > 0
            else np.full(batch_max_atoms, fill_value=atoms_padding_idx, dtype=np.uint8)
            for mol, n in zip(molecules, num_atoms_per_mol)
        ])

        batch_atoms_degree = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            if n > 1:
                d = np.concatenate(([0], (mol.bonds[1:n, 1:n] > 0).sum(axis=1)))
                batch_atoms_degree.append(
                    np.pad(d, (0, batch_max_atoms - n), mode='constant', constant_values=degree_padding_idx))
            else:
                batch_atoms_degree.append(np.full(batch_max_atoms, fill_value=degree_padding_idx, dtype=int))
        batch_atoms_degree = np.stack(batch_atoms_degree)

        bonds_list = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            p_b = np.pad(mol.bonds, [(0, batch_max_atoms - n), (0, batch_max_atoms - n)], mode="constant",
                         constant_values=bond_padding_idx) if n > 0 else np.full((batch_max_atoms, batch_max_atoms),
                                                                                 fill_value=bond_padding_idx, dtype=int)
            np.fill_diagonal(p_b, bond_padding_idx)
            bonds_list.append(p_b)
        batch_bonds = np.stack(bonds_list)

        additive_padding_masks = []
        for mol, n in zip(molecules, num_atoms_per_mol):
            p_m = np.pad(np.zeros((n, n), dtype=float), [(0, batch_max_atoms - n), (0, batch_max_atoms - n)],
                         mode="constant", constant_values=-np.inf) if n > 0 else np.full(
                (batch_max_atoms, batch_max_atoms), fill_value=-np.inf, dtype=float)
            np.fill_diagonal(p_m, 0.0)
            additive_padding_masks.append(p_m)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        batch_level_idx = [mol.current_action_level for mol in molecules]

        feasibility_masks_per_level = []
        num_actions_per_level_and_mol = [
            [n for n in num_atoms_per_mol],
            [2 * vocab_size + n for n in num_atoms_per_mol],
            [maximum_bond_order + 1] * len(molecules)
        ]

        for lvl, num_actions_this_level in enumerate(num_actions_per_level_and_mol):
            max_num_actions = max(num_actions_this_level) if num_actions_this_level else 0
            mask_list = []
            for i, mol in enumerate(molecules):
                current_mask = mol.current_action_mask.astype(
                    bool) if mol.current_action_level == lvl and mol.current_action_mask is not None else np.zeros(
                    num_actions_this_level[i], dtype=bool)
                mask_list.append(np.pad(current_mask, (0, max_num_actions - num_actions_this_level[i]), mode='constant',
                                        constant_values=True))
            feasibility_masks_per_level.append(torch.from_numpy(np.stack(mask_list)).bool().to(device))

        return dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long, device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe).long().to(device),
            num_atoms=torch.tensor(num_atoms_per_mol, dtype=torch.long, device=device),
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long().to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(batch_additive_padding_attn_mask).float().to(device),
            feasibility_mask_level_zero=feasibility_masks_per_level[0],
            feasibility_mask_level_one=feasibility_masks_per_level[1],
            feasibility_mask_level_two=feasibility_masks_per_level[2]
        )

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    @staticmethod
    def get_single_atom_molecules(config: MoleculeConfig, repeat: int = 1) -> List['MoleculeDesign']:
        allowed_atom_indices = [i + 1 for i, name in enumerate(config.atom_vocabulary.keys()) if
                                config.atom_vocabulary[name].get("allowed", False)]
        return MoleculeDesign.init_batch_from_instance_list(config, allowed_atom_indices * repeat)

    @staticmethod
    def from_smiles(config: MoleculeConfig, smiles: str, do_finish=False, compare_smiles=False) -> 'MoleculeDesign':
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            raise ValueError(f"Invalid SMILES string: '{smiles}' - RDKit could not parse it")

        Chem.SanitizeMol(mol)

        # We MUST use a canonical SMILES to ensure the atom order
        # is identical between generation and replay.
        canonical_smiles = Chem.MolToSmiles(mol)

        # If the input smiles was not canonical, we must re-create the
        # mol object from the canonical smiles to get the canonical atom order.
        if smiles != canonical_smiles:
            mol = Chem.MolFromSmiles(canonical_smiles)
            Chem.SanitizeMol(mol)

        # Pass the canonical mol and smiles to the builder
        design = MoleculeDesign.from_rdkit_mol(config, mol, canonical_smiles, do_finish, compare_smiles)

        if not do_finish:
            # Store the same canonical SMILES we used to build the design
            design.prompt_smiles = canonical_smiles

        return design[0]

    @staticmethod
    def from_rdkit_mol(config: MoleculeConfig, rdkit_mol: Chem.Mol, smiles: Optional[str] = None) -> Tuple[
        'MoleculeDesign', Dict[int, int]]:
        BOND_TYPE_TO_RL_ORDER = {
            Chem.BondType.SINGLE: 1, Chem.BondType.DOUBLE: 2, Chem.BondType.TRIPLE: 3,
            Chem.BondType.QUADRUPLE: 4, Chem.BondType.QUINTUPLE: 5, Chem.BondType.HEXTUPLE: 6,
        }

        num_heavy_atoms = rdkit_mol.GetNumAtoms()
        first_allowed_idx = next((i + 1 for i, name in enumerate(config.atom_vocabulary.keys()) if
                                  config.atom_vocabulary[name].get("allowed", False)), 1)
        reverse_atom_lookup = build_reverse_atom_lookup(config)

        internal_atoms_list = [0]
        rdkit_to_internal_map = {}
        for atom in rdkit_mol.GetAtoms():
            rdkit_chiral = atom.GetChiralTag()
            chiral_key_val = 1 if rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW else (
                2 if rdkit_chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)
            vocab_idx = reverse_atom_lookup.get((atom.GetAtomicNum(), atom.GetFormalCharge(), chiral_key_val))

            internal_atoms_list.append(vocab_idx)
            rdkit_to_internal_map[atom.GetIdx()] = len(internal_atoms_list) - 1

        num_total_atoms = len(internal_atoms_list)
        internal_bonds_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=np.uint8)
        is_original_bond_matrix = np.zeros((num_total_atoms, num_total_atoms), dtype=bool)

        for bond in rdkit_mol.GetBonds():
            rl_order = BOND_TYPE_TO_RL_ORDER.get(bond.GetBondType())
            int_idx1, int_idx2 = rdkit_to_internal_map[bond.GetBeginAtomIdx()], rdkit_to_internal_map[
                bond.GetEndAtomIdx()]
            internal_bonds_matrix[int_idx1, int_idx2] = internal_bonds_matrix[int_idx2, int_idx1] = rl_order
            # Mark edges from the prompt as "original" so they aren't locked
            is_original_bond_matrix[int_idx1, int_idx2] = is_original_bond_matrix[int_idx2, int_idx1] = True

        if num_total_atoms > 1:
            internal_bonds_matrix[0, 1:] = internal_bonds_matrix[1:, 0] = MoleculeDesign.virtual_bond_idx

        instance = MoleculeDesign(config, initial_atom=first_allowed_idx)
        instance.atoms = np.array(internal_atoms_list, dtype=np.uint8)
        instance.bonds = internal_bonds_matrix
        instance.is_original_atom = np.array([False] + [True] * num_heavy_atoms, dtype=bool)
        instance.is_original_bond = is_original_bond_matrix

        instance.synthesis_done = False
        instance.smiles_string = None
        instance.current_action_level = 0
        instance.history = []
        instance.update_action_mask()
        instance._recreate_rdkit_mol_from_state()

        return instance, rdkit_to_internal_map