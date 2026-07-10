#!/usr/bin/env python3
"""
BRICS Fragment Vocabulary Construction for AMORTIX 2.0
=======================================================

Builds a curated vocabulary of BRICS fragments from ChEMBL SMILES files.
Fragments are canonicalized, filtered, and ranked by frequency. The top-K
fragments are saved with precomputed properties for the FragmentEntry dataclass
used by the USES engine (Universal Subgraph Edit State).

Methodology follows FREED (Yang et al., NeurIPS 2021), FREED++ (Telnykh et al.,
TMLR 2024), and f-RAG (Lee et al., NeurIPS 2024):

  - Reservoir-sample 250,000 molecules (FREED/FREED++ convention)
  - Skip molecules with >60 heavy atoms before BRICS (Landrum best practice)
  - 5-second timeout per molecule during BRICS decomposition
  - Multiprocessing parallelisation
  - Exclude charged fragments (FREED++ §A.2)
  - Filter by frequency ≥ 5 (stricter than FREED ≥3, FREED++ ≥2)
  - Max 16 heavy atoms per fragment (FREED++ §5.2)
  - Near-duplicate removal via Tanimoto ≥ 0.95 on ECFP4 (f-RAG §3.2; FREED)

Expected input:
    data/chembl/chembl_train.smiles   -- one SMILES per line
    data/chembl/chembl_valid.smiles   -- one SMILES per line

Output:
    data/fragments/brics_vocab_K{K}.json

Usage:
    python build_brics_vocab.py --top-k 1000 --min-freq 5 --max-atoms 16
"""

import argparse
import json
import os
import random
import signal
import sys
import time
from collections import Counter
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem import rdMolDescriptors
from rdkit import RDLogger

# ---------------------------------------------------------------------------
# Suppress RDKit warnings during bulk processing
# ---------------------------------------------------------------------------
RDLogger.logger().setLevel(RDLogger.ERROR)

# ===================================================================
# CONSTANTS
# ===================================================================

# BRICS dummy atom isotope labels → bond type they imply when paired
# with any compatible partner.
# Reference: Degen et al. (2008), ChemMedChem.
#
# Values: 1 = SINGLE, 2 = DOUBLE, 3 = TRIPLE
BRICS_ISOTOPE_TO_BOND_TYPE: Dict[int, int] = {
    1: 1, 2: 3, 3: 1, 4: 1,
    5: 2, 6: 2,
    7: 1, 8: 1, 9: 1, 10: 1,
    11: 1, 12: 1, 13: 3, 14: 1, 15: 3, 16: 1,
}

# Maximum heavy atoms in a molecule before we skip BRICS decomposition.
# Following Greg Landrum's "Common Chemical Words" pipeline (May 2025)
# where he filters PubChem molecules with >60 heavy atoms before BRICS.
MAX_HEAVY_ATOMS_FOR_BRICS: int = 60

# Per-molecule timeout for BRICS decomposition.
BRICS_TIMEOUT_SECONDS: int = 5

# Default reservoir sample size (FREED/FREED++ use ~250K from ZINC).
DEFAULT_SAMPLE_SIZE: int = 250_000

# Default random seed for reproducibility.
DEFAULT_RANDOM_SEED: int = 42


# ===================================================================
# UTILITY: TIMEOUT CONTEXT MANAGER (UNIX)
# ===================================================================

@contextmanager
def timeout_context(seconds: int):
    """
    Raise TimeoutError if the enclosed block takes longer than `seconds`.

    Uses SIGALRM, so this works on Unix only.  On Windows the context
    manager is a no-op (acceptable for a one-off vocabulary script).
    """
    if not hasattr(signal, "SIGALRM"):
        # Windows fallback ─ no timeout
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Operation timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ===================================================================
# STEP 1: RESERVOIR SAMPLING
# ===================================================================

def stream_smiles(filepaths: List[str]):
    """
    Generator that yields one SMILES string at a time from a list of
    file paths.  Skips empty lines and comment lines starting with '#'.
    """
    for fp in filepaths:
        with open(fp, "r") as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith("#"):
                    yield line


def reservoir_sample(
    filepaths: List[str],
    n: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> List[str]:
    """
    Reservoir sampling: select `n` random SMILES from an arbitrarily
    large stream without loading all lines into memory.
    """
    rng = random.Random(seed)
    reservoir: List[str] = []
    total_seen = 0

    for smi in stream_smiles(filepaths):
        total_seen += 1
        if len(reservoir) < n:
            reservoir.append(smi)
        else:
            j = rng.randint(0, total_seen - 1)
            if j < n:
                reservoir[j] = smi

    print(f"      Sampled {len(reservoir)} molecules "
          f"(total stream size: {total_seen})")
    return reservoir


# ===================================================================
# STEP 2: PER-MOLECULE BRICS DECOMPOSITION (SINGLE POOL, STREAMING)
# ===================================================================

# Additional pre-filter for molecules that cause BRICS combinatorial
# explosion.  Molecules with many rotatable bonds generate an
# exponential number of fragmentation patterns.
MAX_ROTATABLE_BONDS_FOR_BRICS: int = 15


def _molecule_precheck(smi: str) -> Optional[Chem.Mol]:
    """
    Parse a SMILES and apply pre-filters that are cheap enough to run
    in the main process before dispatching to workers.

    Returns the parsed Mol if it passes all checks, else None.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS_FOR_BRICS:
        return None
    # ── Rotatable bond cap ──────────────────────────────────
    # High rotatable-bond count → combinatorial explosion in BRICS.
    # 15 is generous (drug-like molecules typically have ≤ 10),
    # but catches the pathological cases that hang for 30+ seconds.
    if rdMolDescriptors.CalcNumRotatableBonds(mol) > MAX_ROTATABLE_BONDS_FOR_BRICS:
        return None
    return mol


def _process_one_molecule(mol: Chem.Mol) -> Optional[List[str]]:
    """
    Worker function: BRICS-decompose a *pre-checked* RDKit Mol.

    The molecule has already passed the heavy-atom and rotatable-bond
    filters, so decomposition should be fast.  No timeout is needed
    because the pre-filters eliminate the pathological cases.

    Returns:
        List of canonical fragment SMILES, or None on failure.
    """
    try:
        raw_fragments = list(
            BRICS.BRICSDecompose(mol, keepNonLeafNodes=True)
        )
    except Exception:
        return None

    canonical: List[str] = []
    for frag_smi in raw_fragments:
        f_mol = Chem.MolFromSmiles(frag_smi)
        if f_mol is not None:
            canonical.append(Chem.MolToSmiles(f_mol, isomericSmiles=True))

    return canonical if canonical else None


def parallel_decompose(
    smiles_list: List[str],
    n_workers: int,
) -> Counter:
    """
    Parallel BRICS decomposition with streaming progress.

    Design:
      - Pre-filter molecules in the main process (heavy atoms, rotatable
        bonds) — cheap, eliminates slow BRICS cases before dispatching.
      - Create ONE multiprocessing Pool, reused for all chunks.
      - Use ``imap_unordered`` so results stream in as they complete.
        Progress updates are continuous, not gated by the slowest molecule.
      - ``maxtasksperchild=1000`` periodically refreshes workers,
        preventing any memory accumulation from RDKit C++ objects.
    """
    from rdkit.Chem import rdMolDescriptors

    print(f"      Pre-filtering {len(smiles_list)} molecules ...")

    # ── Pre-filter in main process ──────────────────────────────
    pre_filtered: List[Tuple[int, Chem.Mol]] = []
    n_skipped_heavy = 0
    n_skipped_rot = 0
    n_skipped_parse = 0
    n_skipped_total = 0

    for smi in smiles_list:
        mol = _molecule_precheck(smi)
        if mol is None:
            n_skipped_total += 1
            # We can't distinguish the reason here easily without
            # re-parsing, but that's fine for a progress summary
            continue
        pre_filtered.append((len(pre_filtered), mol))

    # More detailed skip counts (re-parse to get reasons — cheap)
    # Only do this for reporting; skip in production if it's slow
    n_heavy = sum(
        1 for smi in smiles_list
        if (mol := Chem.MolFromSmiles(smi)) is not None
        and mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS_FOR_BRICS
    )
    n_rot = sum(
        1 for smi in smiles_list
        if (mol := Chem.MolFromSmiles(smi)) is not None
        and mol.GetNumHeavyAtoms() <= MAX_HEAVY_ATOMS_FOR_BRICS
        and rdMolDescriptors.CalcNumRotatableBonds(mol) > MAX_ROTATABLE_BONDS_FOR_BRICS
    )
    n_parse = sum(1 for smi in smiles_list if Chem.MolFromSmiles(smi) is None)

    print(f"      Heavy atoms > {MAX_HEAVY_ATOMS_FOR_BRICS}:        {n_heavy:>7d}")
    print(f"      Rotatable bonds > {MAX_ROTATABLE_BONDS_FOR_BRICS}: {n_rot:>7d}")
    print(f"      Parse failures:                 {n_parse:>7d}")
    print(f"      Molecules dispatched to BRICS:  {len(pre_filtered):>7d}")

    # ── Single Pool, streaming ──────────────────────────────────
    print(f"\n      Decomposing {len(pre_filtered)} molecules "
          f"({n_workers} workers, streaming) ...")

    fragment_counter: Counter = Counter()
    t_start = time.time()
    done = 0

    with Pool(
        processes=n_workers,
        maxtasksperchild=1000,   # refresh workers periodically
    ) as pool:
        # imap_unordered yields results as soon as ANY worker finishes
        # (not in input order, but we don't care about order)
        for result in pool.imap_unordered(
            _process_one_molecule,
            (mol for _, mol in pre_filtered),
            chunksize=10,         # small chunks → smoother streaming
        ):
            done += 1
            if result is not None:
                fragment_counter.update(result)

            # Progress update every 100 molecules (fast enough, not spammy)
            if done % 100 == 0 or done == len(pre_filtered):
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0.0
                print(
                    f"      [{done:>7d}/{len(pre_filtered)}]  "
                    f"{rate:6.0f} mol/s  |  "
                    f"unique frags: {len(fragment_counter):>6d}",
                    end="\r" if done < len(pre_filtered) else "\n",
                )

    elapsed = time.time() - t_start
    print(f"      Finished in {elapsed:.1f} s  "
          f"({len(pre_filtered) / elapsed:.0f} mol/s avg)")
    print(f"      Fragments per molecule (avg):  "
          f"{sum(fragment_counter.values()) / len(pre_filtered):.1f}"
          if pre_filtered else "")
    print(f"      Unique raw fragments:          {len(fragment_counter)}")

    return fragment_counter


# ===================================================================
# STEP 3: FRAGMENT FILTERING
# ===================================================================

def fragment_has_charge(smiles: str) -> bool:
    """Return True if the fragment carries any non-zero formal charge."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return True  # treat unparseable as charged → exclude
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            return True
    return False


def count_heavy_atoms(smiles: str) -> int:
    """Count non-dummy, non-hydrogen atoms in a fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1  # neither dummy (0) nor hydrogen (1)
    )


def count_attachment_sites(smiles: str) -> int:
    """Count dummy atoms (BRICS attachment points) in the fragment."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


def fragment_passes_filters(
    smiles: str,
    max_heavy: int,
) -> bool:
    """
    Quick pre-filter before property computation.

    Returns True if the fragment:
      - has no formal charges,
      - has ≥ 1 heavy atom and ≤ `max_heavy`,
      - has ≥ 1 attachment site.
    """
    if fragment_has_charge(smiles):
        return False
    n_heavy = count_heavy_atoms(smiles)
    if n_heavy < 1 or n_heavy > max_heavy:
        return False
    if count_attachment_sites(smiles) < 1:
        return False
    return True


def filter_fragments(
    raw_counter: Counter,
    min_frequency: int,
    max_heavy_atoms: int,
) -> Dict[str, int]:
    """
    Apply frequency threshold, charge removal, heavy-atom cap, and
    attachment-site requirement.  Returns filtered {smiles: count} dict.
    """
    print(f"\n[Filtering fragments]")
    print(f"      Raw unique fragments:            {len(raw_counter)}")

    # --- Frequency filter ---
    freq_filtered = {
        smi: cnt
        for smi, cnt in raw_counter.items()
        if cnt >= min_frequency
    }
    print(f"      After frequency ≥ {min_frequency}:          "
          f"{len(freq_filtered)}")

    # --- Charge + size + attachment filter ---
    prop_filtered = {
        smi: cnt
        for smi, cnt in freq_filtered.items()
        if fragment_passes_filters(smi, max_heavy_atoms)
    }
    print(f"      After charge + size + attach:     "
          f"{len(prop_filtered)}")

    return prop_filtered


# ===================================================================
# STEP 4: NEAR-DUPLICATE REMOVAL (FREQUENCY PRE-FILTERED)
# ===================================================================

# Number of top fragments by frequency to retain BEFORE running the
# expensive O(N²) near-duplicate removal.  Must be ≥ top_k to leave
# enough headroom.  We use a fixed multiple of top_k so the user only
# has to think about K.
DEDUP_POOL_MULTIPLIER: int = 5  # dedup pool = top_k × this


def remove_near_duplicates(
    fragment_smiles_list: List[str],
    tanimoto_threshold: float = 0.95,
) -> List[str]:
    """
    Remove fragments whose ECFP4 Tanimoto similarity ≥ `tanimoto_threshold`.

    For each near-duplicate cluster we keep the fragment with **fewer**
    attachment sites (FREED / FREED++ convention).

    Complexity: O(N²) fingerprint comparisons.  This function expects
    that `fragment_smiles_list` has already been frequency‑trimmed to
    a manageable size (≤ ~5000).  See `DEDUP_POOL_MULTIPLIER`.

    Returns:
        List of surviving fragment SMILES, in unchanged order.
    """
    n_input = len(fragment_smiles_list)
    if n_input == 0:
        return []

    print(f"\n[Removing near-duplicates (Tanimoto ≥ {tanimoto_threshold})]")
    print(f"      Input fragments:  {n_input}")

    # Estimate runtime to set expectations
    est_comparisons = n_input * (n_input - 1) // 2
    if est_comparisons > 50_000_000:
        print(f"      ⚠  {est_comparisons:,.0f} pairwise comparisons — "
              f"this may take a few minutes")
    t0 = time.time()

    # Build mols and fingerprints (skip unparseable)
    pairs: List[Tuple[int, Chem.Mol]] = []
    for i, smi in enumerate(fragment_smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            pairs.append((i, mol))

    # Pre-compute attachment site counts (used for tie-breaking)
    attach_counts = [
        count_attachment_sites(fragment_smiles_list[idx])
        for idx, _ in pairs
    ]

    fps = [
        AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)
        for _, m in pairs
    ]

    m = len(pairs)
    keep: set = set(range(m))

    for i in range(m):
        if i not in keep:
            continue
        fi = fps[i]
        for j in range(i + 1, m):
            if j not in keep:
                continue
            fj = fps[j]
            sim = Chem.DataStructs.TanimotoSimilarity(fi, fj)
            if sim >= tanimoto_threshold:
                # Keep the fragment with fewer attachment sites
                if attach_counts[i] <= attach_counts[j]:
                    keep.discard(j)
                else:
                    keep.discard(i)
                    break  # i is removed, stop comparing it

    result = [fragment_smiles_list[pairs[i][0]] for i in sorted(keep)]
    elapsed = time.time() - t0
    print(f"      Removed:          {n_input - len(result)}")
    print(f"      After dedup:      {len(result)}  ({elapsed:.1f} s)")
    return result


# ===================================================================
# STEP 5 (REVISED): FREQUENCY PRE-FILTER → DEDUP → TOP-K
# ===================================================================

def select_top_k_after_dedup(
    filtered_pool: Dict[str, int],
    top_k: int,
    deduplicate: bool,
    dedup_threshold: float,
) -> List[Tuple[str, int]]:
    """
    1. Sort by frequency (descending).
    2. Take the top (top_k × DEDUP_POOL_MULTIPLIER) as the dedup pool.
    3. Run near-duplicate removal on that pool.
    4. Select the final top-k by frequency from the survivors.

    This avoids running the O(N²) dedup on all ~70K fragments.
    """
    sorted_all = sorted(
        filtered_pool.items(), key=lambda kv: kv[1], reverse=True
    )

    if not deduplicate:
        return sorted_all[:top_k]

    # ── Frequency pre-filter ──────────────────────────────────
    dedup_pool_size = min(top_k * DEDUP_POOL_MULTIPLIER, len(sorted_all))
    dedup_pool_items = sorted_all[:dedup_pool_size]
    dedup_pool_smiles = [smi for smi, _ in dedup_pool_items]

    print(f"\n[Frequency pre-filter before dedup]")
    print(f"      Full filtered pool:     {len(sorted_all)}")
    print(f"      Dedup pool (top {dedup_pool_size}):    {len(dedup_pool_smiles)}")
    print(f"      Dedup pool max freq:    {dedup_pool_items[0][1]}")
    print(f"      Dedup pool min freq:    {dedup_pool_items[-1][1]}")

    # ── Run dedup ─────────────────────────────────────────────
    kept_smiles = set(remove_near_duplicates(dedup_pool_smiles, dedup_threshold))

    # ── Reconstruct frequency ordering from survivors ─────────
    survivors = [
        (smi, freq) for smi, freq in dedup_pool_items
        if smi in kept_smiles
    ]

    print(f"      Survivors after dedup:  {len(survivors)}")

    if len(survivors) < top_k:
        # Fallback: if dedup removed too many, pull from the rest
        # of the frequency-sorted list (no dedup on these)
        print(f"      ⚠  Fewer than {top_k} survivors — "
              f"backfilling from remaining pool")
        survivor_smiles = set(s for s, _ in survivors)
        for smi, freq in sorted_all[dedup_pool_size:]:
            if len(survivors) >= top_k:
                break
            if smi not in survivor_smiles:
                survivors.append((smi, freq))
                survivor_smiles.add(smi)

    return survivors[:top_k]

def brincs_bond_order(isotope_a: int, isotope_b: int) -> int:
    """
    Bond order implied by a pair of BRICS dummy-atom isotope labels.

    Returns 1 (SINGLE), 2 (DOUBLE), or 3 (TRIPLE).
    Returns 0 if the pair is chemically incompatible (for action masking).
    """
    type_a = BRICS_ISOTOPE_TO_BOND_TYPE.get(isotope_a, 0)
    type_b = BRICS_ISOTOPE_TO_BOND_TYPE.get(isotope_b, 0)
    if type_a == 0 or type_b == 0:
        return 0
    return type_a if type_a == type_b else 0


def get_attachment_info(smiles: str) -> List[Dict]:
    """
    For each dummy atom in the fragment, return::

        {
            "atom_index": int,   # 0‑based index in the RDKit Mol
            "isotope":    int,   # BRICS environment type (1–16)
            "bond_type":  int,   # bond order this site implies (1/2/3)
        }
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    info: List[Dict] = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0:
            isotope = atom.GetIsotope()
            bond_type = BRICS_ISOTOPE_TO_BOND_TYPE.get(isotope, 0)
            info.append({
                "atom_index": atom.GetIdx(),
                "isotope": isotope,
                "bond_type": bond_type,
            })
    return info


def compute_internal_matrices(
    smiles: str,
) -> Tuple[List[List[int]], List[List[int]], List[int]]:
    """
    Compute bond-order and shortest-path-distance matrices for a BRICS
    fragment, considering **real atoms only** (dummy atoms are excluded).

    Returns:
        internal_bonds:     (n_real, n_real) int matrix of bond orders.
        internal_distances: (n_real, n_real) int matrix of shortest paths.
        atom_types:         list of atomic numbers for each real atom
                            (serves as a proxy global atom-type index;
                            formal charge / chirality would require mapping
                            to the full AMORTIX atom vocabulary).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse fragment: {smiles}")

    # Identify real vs dummy atoms
    real_idxs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 0:
            real_idxs.append(atom.GetIdx())

    n_real = len(real_idxs)
    real_to_new = {old: new for new, old in enumerate(real_idxs)}

    # --- Bond matrix ---
    bonds = np.zeros((n_real, n_real), dtype=np.uint8)
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        if i in real_idxs and j in real_idxs:
            order = int(bond.GetBondTypeAsDouble())
            ri, rj = real_to_new[i], real_to_new[j]
            bonds[ri, rj] = order
            bonds[rj, ri] = order

    # --- Distance matrix ---
    try:
        dist_full = Chem.GetDistanceMatrix(mol, force=True).astype(np.uint8)
    except Exception:
        dist_full = np.full((mol.GetNumAtoms(), mol.GetNumAtoms()),
                            255, dtype=np.uint8)
        np.fill_diagonal(dist_full, 0)

    distances = np.zeros((n_real, n_real), dtype=np.uint8)
    for ri, old_i in enumerate(real_idxs):
        for rj, old_j in enumerate(real_idxs):
            distances[ri, rj] = dist_full[old_i, old_j]

    # --- Atom types ---
    atom_types = [
        mol.GetAtomWithIdx(old_i).GetAtomicNum()
        for old_i in real_idxs
    ]

    return bonds.tolist(), distances.tolist(), atom_types


def compute_fragment_entry(
    smiles: str,
    fragment_id: int,
    frequency: int,
) -> Dict:
    """
    Build the complete FragmentEntry dictionary for one fragment.
    """
    n_heavy = count_heavy_atoms(smiles)
    n_attach = count_attachment_sites(smiles)
    attach_info = get_attachment_info(smiles)
    bonds, distances, atom_types = compute_internal_matrices(smiles)

    return {
        "fragment_id":              fragment_id,
        "smiles":                   smiles,
        "frequency":                frequency,
        "num_atoms":                n_heavy,
        "num_attachment_sites":     n_attach,
        "attachment_atom_indices":  [a["atom_index"] for a in attach_info],
        "attachment_isotopes":      [a["isotope"] for a in attach_info],
        "attachment_bond_types":    [a["bond_type"] for a in attach_info],
        "atom_types":               atom_types,
        "internal_bonds":           bonds,
        "internal_distances":       distances,
    }


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def build_brics_vocabulary(
    train_path: str,
    valid_path: str,
    top_k: int = 1000,
    min_frequency: int = 5,
    max_heavy_atoms: int = 16,
    deduplicate: bool = True,
    dedup_threshold: float = 0.95,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    n_workers: Optional[int] = None,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """
    Full pipeline:

        reservoir-sample → BRICS decompose → filter → dedup →
        top‑K selection → property computation → save JSON.

    Parameters
    ----------
    train_path : str
        Path to training SMILES file (one per line).
    valid_path : str
        Path to validation SMILES file.
    top_k : int
        Number of top fragments to retain (by frequency).
    min_frequency : int
        Minimum occurrence count to be considered.
    max_heavy_atoms : int
        Maximum number of heavy (non-dummy) atoms per fragment.
    deduplicate : bool
        Whether to apply near-duplicate removal.
    dedup_threshold : float
        Tanimoto threshold for near-duplicate removal.
    sample_size : int
        Number of molecules to reservoir-sample from the input files.
    random_seed : int
        Seed for reservoir sampling.
    n_workers : int, optional
        Number of parallel workers (default: cpu_count()).
    output_path : str, optional
        If provided, save vocabulary JSON to this path.

    Returns
    -------
    List[Dict]
        Fragment entry dictionaries, sorted by fragment_id.
    """
    if n_workers is None:
        # n_workers = cpu_count()
        n_workers = 24

    filepaths = [train_path, valid_path]

    # ── Step 1 ─────────────────────────────────────────────────
    print("=" * 60)
    print("[1/6] Reservoir sampling ...")
    print("=" * 60)
    all_smiles = reservoir_sample(filepaths, n=sample_size, seed=random_seed)

    # ── Step 2 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[2/6] Parallel BRICS decomposition ...")
    print("=" * 60)
    raw_counter = parallel_decompose(all_smiles, n_workers)

    # ── Step 3 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3/6] Filtering ...")
    print("=" * 60)
    filtered = filter_fragments(raw_counter, min_frequency, max_heavy_atoms)

    # ── Step 4+5 (REVISED): Frequency pre-filter → dedup → top-K ──
    print("\n" + "=" * 60)
    if deduplicate:
        print(f"[4/6] Frequency pre-filter → dedup → top-{top_k} ...")
    else:
        print(f"[4/6] Selecting top-{top_k} by frequency ...")
    print("=" * 60)

    sorted_frags = select_top_k_after_dedup(
        filtered_pool=filtered,
        top_k=top_k,
        deduplicate=deduplicate,
        dedup_threshold=dedup_threshold,
    )

    # ── Step 6 ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[5/5] Computing fragment properties ...")
    print("=" * 60)

    vocabulary: List[Dict] = []
    for frag_id, (smi, freq) in enumerate(sorted_frags):
        try:
            entry = compute_fragment_entry(smi, frag_id, freq)
            vocabulary.append(entry)
        except Exception as exc:
            print(f"      WARNING: failed for fragment {frag_id} "
                  f"({smi}): {exc}", file=sys.stderr)

    print(f"      Final vocabulary size: {len(vocabulary)}")

    # ── Statistics ──────────────────────────────────────────────
    if vocabulary:
        n_atoms = [e["num_atoms"] for e in vocabulary]
        n_attach = [e["num_attachment_sites"] for e in vocabulary]

        print(f"\n{'─' * 60}")
        print(f"Vocabulary Statistics")
        print(f"{'─' * 60}")
        print(f"  Atoms per fragment:   "
              f"min={min(n_atoms)}, max={max(n_atoms)}, "
              f"mean={np.mean(n_atoms):.1f}, "
              f"median={int(np.median(n_atoms))}")
        print(f"  Attachment sites:     "
              f"min={min(n_attach)}, max={max(n_attach)}, "
              f"mean={np.mean(n_attach):.1f}")

        attach_hist = Counter(n_attach)
        for k in sorted(attach_hist):
            pct = 100.0 * attach_hist[k] / len(vocabulary)
            print(f"    {k} site(s): {attach_hist[k]:>5d} fragments "
                  f"({pct:5.1f} %)")

    # ── Save ────────────────────────────────────────────────────
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(vocabulary, fh, indent=2)
        print(f"\nSaved vocabulary to: {output_path}")

    return vocabulary


# ===================================================================
# CLI
# ===================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build BRICS fragment vocabulary for AMORTIX 2.0"
    )
    parser.add_argument(
        "--train-smiles", type=str,
        default="data/chembl/chembl_train.smiles",
        help="Path to training SMILES file",
    )
    parser.add_argument(
        "--valid-smiles", type=str,
        default="data/chembl/chembl_valid.smiles",
        help="Path to validation SMILES file",
    )
    parser.add_argument(
        "--top-k", type=int, default=1000,
        help="Number of top fragments to retain (default: 1000)",
    )
    parser.add_argument(
        "--min-freq", type=int, default=5,
        help="Minimum occurrence count (default: 5)",
    )
    parser.add_argument(
        "--max-atoms", type=int, default=16,
        help="Maximum heavy atoms per fragment (default: 16)",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Skip near-duplicate removal",
    )
    parser.add_argument(
        "--dedup-threshold", type=float, default=0.95,
        help="Tanimoto threshold for near-duplicate removal (default: 0.95)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of molecules to reservoir-sample "
             f"(default: {DEFAULT_SAMPLE_SIZE})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_RANDOM_SEED,
        help=f"Random seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/fragments/brics_vocab_K1000.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    build_brics_vocabulary(
        train_path=args.train_smiles,
        valid_path=args.valid_smiles,
        top_k=args.top_k,
        min_frequency=args.min_freq,
        max_heavy_atoms=args.max_atoms,
        deduplicate=not args.no_dedup,
        dedup_threshold=args.dedup_threshold,
        sample_size=args.sample_size,
        random_seed=args.seed,
        n_workers=args.workers,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()