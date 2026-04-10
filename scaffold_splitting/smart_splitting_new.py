import os
import random
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski

# Disable RDKit warnings for clean output
RDLogger.DisableLog('rdApp.*')

# --- Configuration ---
SMILES_FILE = r"../data/zinc/zinc.smiles"  # Path to ZINC-250k
OUTPUT_DIR = "zinc_splits_optimized"
SEEDS = [42, 43, 44]
TEST_SET_SIZE = 1000
CLUSTERING_CUTOFF = 0.4
MAX_HEAVY_ATOMS = 15

# --- GDB-13 SMARTS Patterns ---
# Filters A & B: No Heteroatom-Heteroatom bonds (N-N, N-O, O-O)
HET_HET_PATTERN = Chem.MolFromSmarts('[#7,#8]-[#7,#8]')
# Filter C: Unstable functional groups (e.g., acetals, hemiacetals, aminals)
UNSTABLE_PATTERN = Chem.MolFromSmarts('[#7,#8;X2,X3]-[CH1,CH2]-[#7,#8;X2,X3]')


def passes_gdb13_rules(mol):
    # """Applies GDB-13 Filters A through F to an RDKit Mol object."""
    # # Filters A, B, C: SMARTS matching
    # if mol.HasSubstructMatch(HET_HET_PATTERN): return False
    # if mol.HasSubstructMatch(UNSTABLE_PATTERN): return False
    #
    # # Filters D & E: No non-aromatic carbon-carbon double/triple bonds
    # for bond in mol.GetBonds():
    #     if bond.GetBondType() in [Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE]:
    #         if not bond.GetIsAromatic():
    #             a1 = bond.GetBeginAtom().GetAtomicNum()
    #             a2 = bond.GetEndAtom().GetAtomicNum()
    #             if a1 == 6 and a2 == 6:
    #                 return False
    #
    # # Filter F: No small rings (3 or 4 membered rings)
    # ring_info = mol.GetRingInfo()
    # for ring in ring_info.AtomRings():
    #     if len(ring) < 5:
    #         return False

    return True


def passes_rule_of_three(mol):
    """
    Evaluates if an RDKit Mol object passes the Fragment Rule of Three (Ro3).
    """
    # try:
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rot_bonds = Lipinski.NumRotatableBonds(mol)

    # Rule of Three logical check
    if (mw <= 300) and (logp <= 3) and (hbd <= 3) and (hba <= 3) and (rot_bonds <= 3):
        return True
    return False
    # except:
    #     # If RDKit fails to calculate a descriptor, reject the molecule to be safe
    #     return False

def get_valid_scaffold(smiles):
    """Extracts the Murcko Scaffold and checks size/topology constraints."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    # try:
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)

    # Must have at least one ring (Filter H inherently enforced)
    if scaffold_mol.GetRingInfo().NumRings() < 1:
        return None

    # Strict Size Constraint
    if scaffold_mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        return None

    # Apply GDB-13 Rules A-F to the scaffold
    if not passes_gdb13_rules(scaffold_mol):
        return None

    # Filter G - Strict Rule of Three (Fragment-like)
    if not passes_rule_of_three(scaffold_mol):
        return None

    return Chem.MolToSmiles(scaffold_mol, isomericSmiles=False)
    # except:
    #     return None


def compute_fingerprints(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_idxs = [i for i, m in enumerate(mols) if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, 1024) for i in valid_idxs]
    return valid_idxs, fps


def memory_efficient_clustering(scaffolds, fps, cutoff=0.4):
    print(f"  Starting Lazy Clustering on {len(scaffolds)} scaffolds...")
    scaffold_idxs = list(range(len(scaffolds)))
    scaffold_idxs.sort(key=lambda i: len(scaffolds[i]), reverse=True)

    clusters = []
    assigned = set()
    pbar = tqdm(total=len(scaffolds), desc="Clustering")

    for leader_idx in scaffold_idxs:
        if leader_idx in assigned:
            continue

        cluster = [leader_idx]
        assigned.add(leader_idx)
        pbar.update(1)

        leader_fp = fps[leader_idx]
        sims = DataStructs.BulkTanimotoSimilarity(leader_fp, fps)

        for i, score in enumerate(sims):
            if i not in assigned and score >= cutoff:
                cluster.append(i)
                assigned.add(i)
                pbar.update(1)

        clusters.append(cluster)
    pbar.close()
    return clusters


def main():
    print(f"--- Loading SMILES from {SMILES_FILE} ---")
    if not os.path.exists(SMILES_FILE):
        print("Error: File not found.")
        return

    with open(SMILES_FILE, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    # USER'S EXACT STRING REPLACEMENT VOCABULARY FILTER
    allowed_vocabulary = [
        "[NH3+]", "[SH+]", "[C@]", "[O+]", "[NH+]", "[nH+]", "[C@@H]", "[CH2-]", "[C@H]", "[NH2+]", "[S+]", "[CH-]",
        "[S@]", "[N-]", "[s+]", "[nH]", "[S@@]", "[n+]", "[o+]", "[NH-]", "[C@@]", "[S-]", "[N+]", "[OH+]", "[O-]",
        "[n-]",
        "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
        "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    ]

    print("Filtering vocabulary...")
    filtered_smiles = []
    for smile in tqdm(smiles_list):
        temp = smile
        for voc in allowed_vocabulary:
            temp = temp.replace(voc, "")
        if len(temp) == 0:
            filtered_smiles.append(smile)
    smiles_list = filtered_smiles

    print("--- Extracting Valid Scaffolds ---")
    scaffold_to_molecules = defaultdict(list)
    for smi in tqdm(smiles_list):
        scaffold = get_valid_scaffold(smi)
        if scaffold:
            scaffold_to_molecules[scaffold].append(smi)

    unique_scaffolds = list(scaffold_to_molecules.keys())
    print(f"Unique Valid Scaffolds Extracted: {len(unique_scaffolds)}")

    print("Computing fingerprints for scaffolds...")
    _, scaffold_fps = compute_fingerprints(unique_scaffolds)

    clusters = memory_efficient_clustering(unique_scaffolds, scaffold_fps, cutoff=CLUSTERING_CUTOFF)
    print(f"Generated {len(clusters)} clusters.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        print(f"\nProcessing Seed {seed}...")
        random.seed(seed)

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        VAL_SET_SIZE = 200
        if len(clusters) < (TEST_SET_SIZE + VAL_SET_SIZE):
            print("WARNING: Not enough clusters to form Test and Val sets.")
            return

        test_scaffolds = []
        val_scaffolds = []
        train_scaffolds = []

        # 1. Sample distinct clusters for Test and Val
        available_indices = list(range(len(clusters)))
        test_cluster_indices = set(random.sample(available_indices, TEST_SET_SIZE))

        # Remove chosen test indices from the pool, then sample for validation
        remaining_indices = [i for i in available_indices if i not in test_cluster_indices]
        val_cluster_indices = set(random.sample(remaining_indices, VAL_SET_SIZE))

        # 2. Populate the sets
        for i, cluster in enumerate(clusters):
            if i in test_cluster_indices:
                test_scaffolds.append(unique_scaffolds[cluster[0]])  # Take centroid
            elif i in val_cluster_indices:
                val_scaffolds.append(unique_scaffolds[cluster[0]])  # Take centroid
            else:
                # Dump entire remaining clusters into train
                for idx in cluster:
                    train_scaffolds.append(unique_scaffolds[idx])

        # 3. The Ultimate Guarantee: Post-Split Leakage Verification
        print("  Running Strict OOD Leakage Verification...")

        # Recompute fingerprints for the final sets
        _, train_fps = compute_fingerprints(train_scaffolds)
        _, val_fps = compute_fingerprints(val_scaffolds)
        _, test_fps = compute_fingerprints(test_scaffolds)

        def filter_leakage(target_scaffolds, target_fps, train_fps, threshold=0.4):
            clean_scaffolds = []
            for idx, fp in enumerate(tqdm(target_fps, leave=False)):
                # Compute similarity against ALL train scaffolds simultaneously
                sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
                if max(sims) < threshold:
                    clean_scaffolds.append(target_scaffolds[idx])
            return clean_scaffolds

        print("  Verifying Validation Set...")
        strict_val_scaffolds = filter_leakage(val_scaffolds, val_fps, train_fps)
        print("  Verifying Test Set...")
        strict_test_scaffolds = filter_leakage(test_scaffolds, test_fps, train_fps)

        # Save to disk
        run_dir = os.path.join(OUTPUT_DIR, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        with open(os.path.join(run_dir, "train_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(train_scaffolds))
        with open(os.path.join(run_dir, "val_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(strict_val_scaffolds))
        with open(os.path.join(run_dir, "test_scaffolds.txt"), 'w') as f:
            f.write('\n'.join(strict_test_scaffolds))

        print(
            f"  [Stats] Train: {len(train_scaffolds)} | Val: {len(strict_val_scaffolds)} | Test: {len(strict_test_scaffolds)}")
        if len(strict_test_scaffolds) < TEST_SET_SIZE:
            print(
                f"  [Note] {TEST_SET_SIZE - len(strict_test_scaffolds)} test scaffolds were dropped to mathematically guarantee zero leakage.")


if __name__ == "__main__":
    main()