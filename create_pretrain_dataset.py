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
import os
import concurrent.futures
from functools import partial
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

        # Don't need removal or replacement actions for purely additive trajectories
        env.enable_removal_actions = False
        env.enable_replacement_actions = False

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
        """
        Dismantles the molecule by randomly selecting valid, unmasked removal actions.
        Relies entirely on the environment's native connectivity checks.
        """
        env, _ = MoleculeDesign.from_rdkit_mol(self.config, Chem.Mol(mol))
        env.max_actions = 1000

        while len(env.atoms) - 1 > 1:  # While > 1 real atoms remain
            num_real_atoms = len(env.atoms) - 1
            action_taken = False

            # --- 1. Try to remove a RANDOM valid atom ---
            # We shuffle the indices to ensure diverse destructive trajectories
            atom_indices = list(range(1, num_real_atoms + 1))
            random.shuffle(atom_indices)

            for target_idx in atom_indices:
                # The environment already has a function to check if removal splinters the graph!
                if env._check_connectivity_after_simulated_removal("Remove Atom", atom_idx=target_idx):
                    action_remove = 2 * self.vocab_size + num_real_atoms
                    env.take_action(target_idx)  # L0: Anchor
                    env.take_action(action_remove)  # L1: Remove Atom
                    action_taken = True
                    break  # Break the for-loop, move to next while-loop step

            if action_taken:
                continue

            # --- 2. If no atom can be removed (e.g. all locked in rings), remove a RANDOM valid bond ---
            bond_indices = []
            for i in range(1, num_real_atoms + 1):
                for j in range(i + 1, num_real_atoms + 1):
                    if env.bonds[i, j] > 0 and env.is_original_bond[i, j]:
                        bond_indices.append((i, j))
            random.shuffle(bond_indices)

            for i, j in bond_indices:
                if env._check_connectivity_after_simulated_removal("Remove Bond", bond_indices=(i, j)):
                    env.take_action(i)  # L0: Anchor i
                    env.take_action(self.vocab_size + j - 1)  # L1: Target j
                    env.take_action(env.maximum_bond_order)  # L2: Remove Bond
                    action_taken = True
                    break

            # --- 3. Safety Catch ---
            # If we literally cannot remove any atom or bond, the graph is deadlocked.
            # (Mathematically very rare, but good for skipping weird edge cases)
            if not action_taken:
                raise RuntimeError(f"Deadlock during removal. Left {num_real_atoms} atoms.")

        env.take_action(0)  # Terminate
        return env.history

    def generate_replacement(self, mol: Chem.RWMol, max_mutations: int = 2, max_attempts: int = 20) -> Tuple[list, str]:
        """
        Corrupts random atoms and uses Substructure Matching to perfectly
        align the internal indices with the final Canonical SMILES order.
        """
        vocab_list = list(self.config.atom_vocabulary.items())
        num_atoms = mol.GetNumAtoms()

        # --- 1. RETRY LOOP (Mutation & Canonical Alignment) ---
        for attempt in range(max_attempts):
            temp_mol = Chem.RWMol(mol)
            mutated_indices = {}  # Tracks: {old_rdkit_idx: original_vocab_idx}

            indices_to_mutate = random.sample(range(num_atoms), min(max_mutations, num_atoms))

            for idx in indices_to_mutate:
                atom = temp_mol.GetAtomWithIdx(idx)
                current_degree = sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds())

                try:
                    original_vocab_idx = self._get_vocab_idx(atom)
                except ValueError:
                    continue

                    # Find valid substitute
                valid_subs = [
                    (name, data) for name, data in vocab_list
                    if data.get('valence', -1) >= current_degree and data.get('atomic_number') != atom.GetAtomicNum()
                ]

                if valid_subs:
                    sub_name, sub_data = random.choice(valid_subs)

                    # Apply substitution
                    atom.SetAtomicNum(sub_data['atomic_number'])
                    atom.SetFormalCharge(sub_data.get('formal_charge', 0))

                    new_chiral = sub_data.get('chiral_tag', 0)
                    if new_chiral == 0:
                        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                    elif new_chiral == 1:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CW)
                    elif new_chiral == 2:
                        atom.SetChiralTag(Chem.ChiralType.CHI_TETRAHEDRAL_CCW)

                    mutated_indices[idx] = original_vocab_idx

            if not mutated_indices:
                continue

            # --- RDKit Graph Isomorphism Check ---
            try:
                Chem.SanitizeMol(temp_mol)

                # 1. Generate the pristine SMILES (What the Verifier will see)
                corrupted_smiles = Chem.MolToSmiles(temp_mol, canonical=True)

                # 2. Parse it back (This scrambles the indices into Canonical Order)
                canon_mol = Chem.MolFromSmiles(corrupted_smiles)
                Chem.SanitizeMol(canon_mol)

                # 3. Find the index mapping: canon_mol (Haystack) vs temp_mol (Needle)
                # match[i] gives the index in canon_mol that corresponds to atom i in temp_mol
                match = canon_mol.GetSubstructMatch(temp_mol, useChirality=True)

                # If exact aromatic Kekulization shifted, try a looser match
                if not match or len(match) != num_atoms:
                    match = canon_mol.GetSubstructMatch(temp_mol, useChirality=False)

                # If RDKit fails to align them, the mutation altered the graph unpredictably. Retry.
                if not match or len(match) != num_atoms:
                    continue

                # SUCCESS: We have the perfect index translation map!
                aligned_mol = canon_mol
                alignment_map = match
                final_mutations = mutated_indices
                final_smiles = corrupted_smiles
                break
            except Exception:
                continue
        else:
            raise ValueError(f"Could not find an alignable mutation after {max_attempts} attempts.")

        # --- 2. SIMULATE THE CORRECTIONS ---
        env, rdkit_to_internal_map = MoleculeDesign.from_rdkit_mol(self.config, aligned_mol)
        env.max_actions = 1000

        for original_idx, correct_vocab_idx in final_mutations.items():
            # Translate: Old Index -> Canonical RDKit Index -> Internal Env Index
            canonical_rdkit_idx = alignment_map[original_idx]
            internal_idx = rdkit_to_internal_map[canonical_rdkit_idx]

            num_real_atoms = len(env.atoms) - 1
            replace_start_idx = self.vocab_size + num_real_atoms
            action_replace = replace_start_idx + (correct_vocab_idx - 1)

            env.take_action(internal_idx)  # L0: Pick Corrupted Atom
            env.take_action(action_replace)  # L1: Replace Atom

        env.take_action(0)  # Terminate
        return env.history, final_smiles


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


def process_single_molecule(mol_data, config):
    """Generates and verifies all 3 tasks for a single molecule."""
    mol, smiles = mol_data
    generator = PretrainingTrajectoryGenerator(config)
    results = []
    additive_errors = 0
    removal_errors = 0
    replacement_errors = 0
    total_errors = 0

    # --- 1. Pure Additive Task ---
    # try:
    start_vocab_idx, add_history = generator.generate_additive(mol)
    data_dict_add = {
        "task_type": "additive",
        "start_atom": start_vocab_idx,
        "prompt_smiles": None,
        "action_seq": add_history,
        "smiles": smiles,
        "obj": 0.0, "sa_score": 0.0
    }
    if verify_trajectory(config, data_dict_add):
        results.append(data_dict_add)
    else:
        additive_errors += 1
        total_errors += 1
    # except Exception:
    #     additive_errors += 1

    # --- 2. Pure Removal Task ---
    # try:
    rem_history = generator.generate_removal(mol)
    data_dict_rem = {
        "task_type": "removal",
        "start_atom": None,
        "prompt_smiles": smiles,
        "action_seq": rem_history,
        "smiles": "",
        "obj": 0.0, "sa_score": 0.0
    }
    if verify_trajectory(config, data_dict_rem):
        results.append(data_dict_rem)
    else:
        removal_errors += 1
        total_errors += 1
    # except Exception:
    #     errors += 1

    # --- 3. Replacement (Corrupt & Fix) Task ---
    # try:
    rep_history, corrupted_smiles = generator.generate_replacement(mol, max_mutations=2)
    data_dict_rep = {
        "task_type": "replacement",
        "start_atom": None,
        "prompt_smiles": corrupted_smiles,
        "action_seq": rep_history,
        "smiles": smiles,
        "obj": 0.0, "sa_score": 0.0
    }
    if verify_trajectory(config, data_dict_rep):
        results.append(data_dict_rep)
    else:
        replacement_errors += 1
        total_errors += 1
    # except Exception:
    #     errors += 1

    return results, (additive_errors, removal_errors, replacement_errors, total_errors)


# --- MAIN EXECUTION BLOCK ---
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

        stats = {"additive": 0, "removal": 0, "replacement": 0, "additive_errors": 0, "removal_errors": 0, "replacement_errors": 0, "total_errors": 0}

        print("Simulating Pretraining Trajectories in Parallel (Additive, Removal, Replacement)...")

        # Determine optimal number of cores (leave 1 free so your OS doesn't freeze)
        num_cores = max(1, os.cpu_count() - 1)

        # 'partial' binds the config object to the function so we can map it over 'molecules'
        process_func = partial(process_single_molecule, config=config)

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Map the function across all molecules and wrap in tqdm for a progress bar
            results_iterator = list(tqdm(executor.map(process_func, molecules), total=len(molecules)))

            # Unpack the results returned by the worker processes
            for valid_results, errors in results_iterator:
                stats["additive_errors"] += errors[0]
                stats["removal_errors"] += errors[1]
                stats["replacement_errors"] += errors[2]
                stats["total_errors"] += errors[3]
                for res in valid_results:
                    molecule_designs.append(res)
                    stats[res["task_type"]] += 1

        print(f"\nGeneration Complete in {time.perf_counter() - start_time:.2f}s.")
        print(f"Successfully generated AND verified trajectories:")
        print(f"  - Additive:    {stats['additive']}")
        print(f"  - Removal:     {stats['removal']}")
        print(f"  - Replacement: {stats['replacement']}")
        print(f"  - Additive Errors: {stats['additive_errors']}")
        print(f"  - Removal Errors: {stats['removal_errors']}")
        print(f"  - Replacement Errors: {stats['replacement_errors']}")
        print(f"  - Total Errors: {stats['total_errors']} (Skipped)")

        with open(destination_path, "wb") as f:
            pickle.dump(molecule_designs, f)

        print(f"Saved verified dataset to {destination_path}\n")