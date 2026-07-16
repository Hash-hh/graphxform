import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm
import time

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# For Visualization
from sklearn.manifold import TSNE

# --- Configuration ---
SMILES_FILE = r"../data/zinc/zinc.smiles"
OUTPUT_DIR = "zinc_splits_optimized"
SEEDS = [43]  # Changed to seed 43

# --- SPLIT CONFIGURATION ---
MIN_VAL_SCAFFOLDS = 100
MIN_TEST_SCAFFOLDS = 500
CLUSTERING_CUTOFF = 0.4


def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except:
        return None


def compute_fingerprints(smiles_list):
    """Generates FPs and keeps valid indices."""
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid_idxs = [i for i, m in enumerate(mols) if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, 1024) for i in valid_idxs]
    return valid_idxs, fps


def memory_efficient_clustering(scaffolds, fps, cutoff=0.4):
    """
    Performs Butina-like clustering without the O(N^2) memory overhead.
    """
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


def run_leakage_analysis(train_scaffolds, test_scaffolds, output_path):
    print("  Computing leakage metrics...")
    _, train_fps = compute_fingerprints(train_scaffolds)
    test_indices, test_fps = compute_fingerprints(test_scaffolds)

    max_sims = []
    for test_fp in tqdm(test_fps, desc="Verifying Leakage", leave=False):
        sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
        max_sims.append(max(sims) if sims else 0.0)

    # Save Data
    with open(output_path.replace(".pdf", ".csv"), "w") as f:
        f.write("Test_Scaffold_Index,Max_Sim_To_Train\n")
        for idx, score in zip(test_indices, max_sims):
            f.write(f"{idx},{score:.4f}\n")

    # Plot with colorblind-safe colors
    plt.figure(figsize=(10, 6))
    sns.histplot(max_sims, bins=30, kde=True, color='tab:purple', alpha=0.6)
    plt.axvline(np.mean(max_sims), color='k', linestyle='--', label=f'Mean: {np.mean(max_sims):.2f}')
    plt.title("Leakage Check: Nearest Neighbor Similarity")
    plt.xlabel("Max Similarity to Train")
    plt.legend()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()


def run_visual_analysis(train_mols, val_mols, test_mols, output_path):
    print("  Generating t-SNE visualization...")
    LIMIT = 2000

    def sample_and_fp(mols, label):
        sampled = random.sample(mols, min(len(mols), LIMIT))
        _, fps = compute_fingerprints(sampled)
        arrs = []
        for fp in fps:
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            arrs.append(arr)
        return np.array(arrs), [label] * len(arrs)

    X_tr, y_tr = sample_and_fp(train_mols, "Train")
    X_va, y_va = sample_and_fp(val_mols, "Val")
    X_te, y_te = sample_and_fp(test_mols, "Test")

    if len(X_tr) == 0: return

    X = np.vstack([X_tr, X_va, X_te])
    y = y_tr + y_va + y_te

    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(12, 8))

    # Colorblind-safe palette with proper hue_order
    sns.scatterplot(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        hue=y,
        style=y,
        hue_order=["Train", "Val", "Test"],
        palette={
            "Train": "tab:blue",
            "Val": "tab:orange",
            "Test": "tab:purple"
        },
        alpha=0.7
    )

    plt.title("Chemical Space Visualization")
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.close()


def main():
    print(f"--- Loading SMILES from {SMILES_FILE} ---")
    if not os.path.exists(SMILES_FILE):
        print("Error: File not found.")
        return

    with open(SMILES_FILE, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]



    # allowed_vocabulary = [
    #     "[NH3+]", "[SH+]", "[C@]", "[O+]", "[NH+]", "[nH+]", "[C@@H]", "[CH2-]", "[C@H]", "[NH2+]", "[S+]", "[CH-]",
    #     "[S@]", "[N-]", "[s+]", "[nH]", "[S@@]", "[n+]", "[o+]", "[NH-]", "[C@@]", "[S-]", "[N+]", "[OH+]", "[O-]",
    #     "[n-]",
    #     "o", "8", "N", "1", "4", "6", "-", ")", "5", "c", "(", "#", "n", "3", "=", "2", "7",
    #     "C", "O", "S", "s", "F", "P", "p", "Cl", "Br", "I"
    # ]
    #
    # print("Filtering vocabulary...")
    # filtered_smiles = []
    # for smile in tqdm(smiles_list):
    #     temp = smile
    #     for voc in allowed_vocabulary:
    #         temp = temp.replace(voc, "")
    #     if len(temp) == 0:
    #         filtered_smiles.append(smile)
    # smiles_list = filtered_smiles



    print("--- Computing Scaffolds ---")
    scaffold_to_molecules = defaultdict(list)
    for smi in tqdm(smiles_list):
        scaffold = get_scaffold(smi)
        if scaffold:
            scaffold_to_molecules[scaffold].append(smi)

    unique_scaffolds = list(scaffold_to_molecules.keys())
    print(f"Unique Scaffolds: {len(unique_scaffolds)}")

    print("Computing fingerprints for scaffolds...")
    _, scaffold_fps = compute_fingerprints(unique_scaffolds)

    clusters = memory_efficient_clustering(unique_scaffolds, scaffold_fps, cutoff=CLUSTERING_CUTOFF)
    print(f"Generated {len(clusters)} clusters.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for seed in SEEDS:
        print(f"\nProcessing Seed {seed}...")
        random.seed(seed)

        random.shuffle(clusters)

        train_scaffolds, val_scaffolds, test_scaffolds = [], [], []
        counts = {'test': 0, 'val': 0}

        for cluster in clusters:
            scaffs = [unique_scaffolds[i] for i in cluster]

            if counts['test'] < MIN_TEST_SCAFFOLDS:
                test_scaffolds.extend(scaffs)
                counts['test'] += len(scaffs)
            elif counts['val'] < MIN_VAL_SCAFFOLDS:
                val_scaffolds.extend(scaffs)
                counts['val'] += len(scaffs)
            else:
                train_scaffolds.extend(scaffs)

        def get_mols(scaff_list):
            return [m for s in scaff_list for m in scaffold_to_molecules[s]]

        train_mols = get_mols(train_scaffolds)
        val_mols = get_mols(val_scaffolds)
        test_mols = get_mols(test_scaffolds)

        run_dir = os.path.join(OUTPUT_DIR, f"run_seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        for name, data in [("train_molecules.txt", train_mols), ("val_molecules.txt", val_mols),
                           ("test_molecules.txt", test_mols)]:
            with open(os.path.join(run_dir, name), 'w') as f: f.write('\n'.join(data))

        for name, data in [("train_scaffolds.txt", train_scaffolds), ("val_scaffolds.txt", val_scaffolds),
                           ("test_scaffolds.txt", test_scaffolds)]:
            with open(os.path.join(run_dir, name), 'w') as f: f.write('\n'.join(data))

        print(f"  [Stats] Train: {len(train_scaffolds)} | Val: {len(val_scaffolds)} | Test: {len(test_scaffolds)}")

        # Changed to PDF output
        run_leakage_analysis(train_scaffolds, test_scaffolds, os.path.join(run_dir, "leakage_analysis.pdf"))
        run_visual_analysis(train_mols, val_mols, test_mols, os.path.join(run_dir, "chemical_space.pdf"))


if __name__ == "__main__":
    main()
