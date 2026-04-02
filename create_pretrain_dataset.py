"""
Pretraining Dataset Generator for GRXForm (Unified Destructive/Additive MDP).
Generates three types of trajectories for each SMILES:
1. Additive (Building from scratch)
2. Removal (Dismantling leaf-by-leaf / ring-breaking)
3. Replacement (Corrupting and correcting)
"""
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
from rdkit import Chem, RDLogger
from typing import Optional, Tuple, Dict, List

from config import MoleculeConfig
from molecule_design import MoleculeDesign, build_reverse_atom_lookup

# Suppress RDKit warnings for cleaner console output
RDLogger.DisableLog('rdApp.*')


class PretrainingTrajectoryGenerator:
    """Generates expert histories for the 3 distinct pretraining tasks."""

    def __init__(self, config: MoleculeConfig):
        self.config = config
        self.reverse_vocab = build_reverse_atom_lookup(config)
        self.vocab_size = len(config.atom_vocabulary)

    def _get_vocab_idx(self, atom: Chem.Atom) -> int:
        """Translates an RDKit atom to the exact 1-based internal vocabulary index."""
        chiral = atom.GetChiralTag()
        chiral_val = 1 if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CW else (
            2 if chiral == Chem.ChiralType.CHI_TETRAHEDRAL_CCW else 0)

        lookup_key = (atom.GetAtomicNum(), atom.GetFormalCharge(), chiral_val)
        idx = self.reverse_vocab.get(lookup_key)
        if idx is None:
            raise ValueError(f"Atom {lookup_key} not found in strict vocabulary.")
        return idx

    def generate_additive(self, mol: Chem.RWMol) -> Tuple[int, list]:
        """
        Simulates building the molecule from scratch using the Anchor-centric MDP.
        Strictly follows RDKit's canonical DFS index ordering to match legacy behavior.
        """
        if mol.GetNumAtoms() == 0: return 1, []

        Chem.Kekulize(mol, clearAromaticFlags=True)
        adjacency_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol, useBO=True)

        start_atom_vocab_idx = self._get_vocab_idx(mol.GetAtomWithIdx(0))
        env = MoleculeDesign(self.config, initial_atom=start_atom_vocab_idx)

        for i in range(1, mol.GetNumAtoms()):
            atom_to_add_vocab_idx = self._get_vocab_idx(mol.GetAtomWithIdx(i))
            action_add = atom_to_add_vocab_idx - 1

            atom_is_placed = False
            for j in range(0, i):
                bond_order = adjacency_matrix[i, j]
                if bond_order > 0:
                    if not atom_is_placed:
                        # Step A: Place the new atom (First connection)
                        anchor_internal_idx = j + 1
                        env.take_action(anchor_internal_idx)  # L0: Pick Anchor j
                        env.take_action(action_add)  # L1: Add Atom i
                        env.take_action(int(bond_order) - 1)  # L2: Set Bond Order
                        atom_is_placed = True
                    else:
                        # Step B: Additional connections (Ring closures)
                        anchor_internal_idx = len(env.atoms) - 1
                        target_0_idx = j
                        action_select = self.vocab_size + target_0_idx

                        env.take_action(anchor_internal_idx)  # L0: Pick Anchor i
                        env.take_action(action_select)  # L1: Select Existing j
                        env.take_action(int(bond_order) - 1)  # L2: Set Bond Order

        env.take_action(0)  # Terminate
        return start_atom_vocab_idx, env.history

    def generate_removal(self, mol: Chem.RWMol) -> list:
        """Simulates safely dismantling the molecule without breaking graph connectivity."""
        env, _ = MoleculeDesign.from_rdkit_mol(self.config, Chem.Mol(mol))

        while len(env.atoms) - 1 > 1:  # While > 1 real atoms remain
            env.update_action_mask()
            num_real_atoms = len(env.atoms) - 1

            # Find a leaf atom (degree == 1)
            degrees = env._get_current_valence_usage()
            leaf_candidates = np.where(degrees == 1)[0] + 1

            if len(leaf_candidates) > 0:
                # Remove Leaf
                target_leaf = int(leaf_candidates[0])
                action_remove = 2 * self.vocab_size + num_real_atoms
                env.take_action(target_leaf)  # L0: Pick Leaf
                env.take_action(action_remove)  # L1: Remove Atom
            else:
                # No leaves found -> We are in a ring. Find a safe bond to remove.
                bond_removed = False
                for i in range(1, num_real_atoms + 1):
                    for j in range(i + 1, num_real_atoms + 1):
                        if env.bonds[i, j] > 0 and env.is_original_bond[i, j]:
                            # Check if removing this bond splinters the graph
                            if env._check_connectivity_after_simulated_removal("Remove Bond", bond_indices=(i, j)):
                                env.take_action(i)  # L0: Anchor i
                                env.take_action(self.vocab_size + j - 1)  # L1: Target j
                                env.take_action(env.maximum_bond_order)  # L2: Remove Bond
                                bond_removed = True
                                break
                    if bond_removed: break

                # If we get stuck (unlikely with valid chemistry), break gracefully
                if not bond_removed: break

        env.take_action(0)  # Terminate
        return env.history

    def generate_replacement(self, mol: Chem.RWMol, max_mutations: int = 2) -> Tuple[list, str]:
        """Corrupts random atoms and generates the trajectory to fix them."""
        vocab_list = list(self.config.atom_vocabulary.items())
        mutated_mol = Chem.RWMol(mol)
        mutations_made = {}

        num_atoms = mol.GetNumAtoms()
        indices_to_mutate = random.sample(range(num_atoms), min(max_mutations, num_atoms))

        for idx in indices_to_mutate:
            atom = mutated_mol.GetAtomWithIdx(idx)
            current_degree = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())
            try:
                original_vocab_idx = self._get_vocab_idx(atom)
            except ValueError:
                continue  # Skip if origin atom is somehow weird

            # Find a valid substitute capable of handling the current bonds
            valid_subs = [
                (name, data) for name, data in vocab_list
                if data.get('valence', -1) >= current_degree and data.get('atomic_number') != atom.GetAtomicNum()
            ]

            if valid_subs:
                sub_name, sub_data = random.choice(valid_subs)
                atom.SetAtomicNum(sub_data['atomic_number'])
                if 'formal_charge' in sub_data: atom.SetFormalCharge(sub_data['formal_charge'])
                mutations_made[idx] = original_vocab_idx

        if not mutations_made:
            raise ValueError("Could not find valid chemical mutations.")

        # Load corrupted state
        corrupted_smiles = Chem.MolToSmiles(mutated_mol)
        env, rdkit_to_internal_map = MoleculeDesign.from_rdkit_mol(self.config, mutated_mol)

        # Simulate Corrections
        for rdkit_idx, correct_vocab_idx in mutations_made.items():
            internal_idx = rdkit_to_internal_map[rdkit_idx]
            num_real_atoms = len(env.atoms) - 1
            replace_start_idx = self.vocab_size + num_real_atoms
            action_replace = replace_start_idx + (correct_vocab_idx - 1)

            env.take_action(internal_idx)  # L0: Pick Corrupted Atom
            env.take_action(action_replace)  # L1: Replace Atom

        env.take_action(0)  # Terminate
        return env.history, corrupted_smiles


def verify_trajectory(config: MoleculeConfig, data_dict: dict) -> bool:
    """
    Plays back a generated trajectory blindly to guarantee the end result matches the target.
    """
    task = data_dict["task_type"]
    expected_smiles = data_dict["smiles"]

    # 1. Initialize from the exact starting state
    if task == "additive":
        env = MoleculeDesign(config, initial_atom=data_dict["start_atom"])
    else:
        # Removal and Replacement start from a prompt string
        env = MoleculeDesign.from_smiles(config, data_dict["prompt_smiles"])

    # 2. Blindly play back the action sequence
    try:
        for action in data_dict["action_seq"]:
            env.take_action(action)
    except Exception as e:
        print(f"Playback failed mid-sequence for {task}: {e}")
        return False

    # 3. Finalize and evaluate the result
    env.finalize(assert_feasible=False)

    if env.infeasibility_flag:
        print(f"Playback resulted in chemically invalid state for {task}.")
        return False

    # 4. Assert the final SMILES matches the expectation
    if task == "removal":
        # Removal target should be a single atom (no bonds)
        num_heavy_atoms = env.rdkit_mol.GetNumAtoms() if env.rdkit_mol else 0
        if num_heavy_atoms > 1:
            print(f"Removal playback failed: Left {num_heavy_atoms} atoms.")
            return False
    else:
        # Additive and Replacement targets must perfectly match the target SMILES
        final_smiles = env.to_smiles(canonical=True)
        target_canonical = Chem.CanonSmiles(expected_smiles)

        if final_smiles != target_canonical:
            print(f"Playback mismatch ({task})!\nExpected: {target_canonical}\nGot:      {final_smiles}")
            return False

    return True

if __name__ == "__main__":
    datatypes = ["valid", "train"]
    limit_num_atoms = 100
    limit_num_smiles_to = None  # Set to an integer for testing (e.g., 500)

    for datatype in datatypes:
        start_time = time.perf_counter()
        molecules: List[Tuple[Chem.RWMol, str]] = []
        molecule_designs: List[dict] = []

        path_to_smiles = f"./data/chembl/chembl_{datatype}_filtered.smiles"
        destination_path = f"./data/chembl/pretrain_sequences/chembl_{datatype}.pickle"

        print(f"--- Processing Datatype: {datatype} ---")
        print("Loading and parsing SMILES...")

        num_differing_smiles = 0
        with open(path_to_smiles) as f:
            for line in tqdm(f):
                smiles = line.rstrip()
                if len(smiles) > 0:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is None: continue
                    Chem.SanitizeMol(mol)
                    canonical_s = Chem.CanonSmiles(Chem.MolToSmiles(mol))

                    if canonical_s != smiles:
                        num_differing_smiles += 1
                    if mol.GetNumAtoms() <= limit_num_atoms:
                        molecules.append((mol, canonical_s))

                if limit_num_smiles_to and len(molecules) == limit_num_smiles_to:
                    break

        print(f"Created {len(molecules)} RDkit molecules. {num_differing_smiles} differed from canonical.")
        if not molecules: continue
        max_num_atoms = max([x.GetNumAtoms() for x, _ in molecules])

        config = MoleculeConfig()
        config.max_num_atoms = max_num_atoms
        generator = PretrainingTrajectoryGenerator(config)

        stats = {"additive": 0, "removal": 0, "replacement": 0, "errors": 0}

        print("Simulating Pretraining Trajectories (Additive, Removal, Replacement)...")
        for mol, smiles in tqdm(molecules):
            # --- 1. Pure Additive Task ---
            try:
                start_vocab_idx, add_history = generator.generate_additive(mol)
                data_dict_add = {
                    "task_type": "additive",
                    "start_atom": start_vocab_idx,  # Int: Starting Vocab Index
                    "prompt_smiles": None,  # No prompt
                    "action_seq": add_history,
                    "smiles": smiles,  # Target to build
                    "obj": 0.0, "sa_score": 0.0
                }

                # VERIFICATION
                if verify_trajectory(config, data_dict_add):
                    molecule_designs.append(data_dict_add)
                    stats["additive"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                stats["errors"] += 1

            # --- 2. Pure Removal Task ---
            try:
                rem_history = generator.generate_removal(mol)
                data_dict_rem = {
                    "task_type": "removal",
                    "start_atom": None,  # No start atom
                    "prompt_smiles": smiles,  # String: The starting prompt to dismantle
                    "action_seq": rem_history,
                    "smiles": "",  # Target is single atom/empty
                    "obj": 0.0, "sa_score": 0.0
                }

                # VERIFICATION
                if verify_trajectory(config, data_dict_rem):
                    molecule_designs.append(data_dict_rem)
                    stats["removal"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                stats["errors"] += 1

            # --- 3. Replacement (Corrupt & Fix) Task ---
            try:
                rep_history, corrupted_smiles = generator.generate_replacement(mol, max_mutations=2)
                data_dict_rep = {
                    "task_type": "replacement",
                    "start_atom": None,  # No start atom
                    "prompt_smiles": corrupted_smiles,  # String: The corrupted starting state
                    "action_seq": rep_history,
                    "smiles": smiles,  # Target is the fixed molecule
                    "obj": 0.0, "sa_score": 0.0
                }

                # VERIFICATION
                if verify_trajectory(config, data_dict_rep):
                    molecule_designs.append(data_dict_rep)
                    stats["replacement"] += 1
                else:
                    stats["errors"] += 1

            except Exception as e:
                stats["errors"] += 1

        print(f"\nGeneration Complete in {time.perf_counter() - start_time:.2f}s.")
        print(f"Successfully generated AND verified trajectories:")
        print(f"  - Additive:    {stats['additive']}")
        print(f"  - Removal:     {stats['removal']}")
        print(f"  - Replacement: {stats['replacement']}")
        print(f"  - Failed/Errors: {stats['errors']} (Skipped)")

        with open(destination_path, "wb") as f:
            pickle.dump(molecule_designs, f)

        print(f"Saved verified dataset to {destination_path}\n")