"""
Fragment data structures and utilities for AMORTIX 2.0 (USES engine).

Provides:
  - BRICS compatibility checking via isotope-label pairs
  - FragmentEntry: pre-computed properties of a single BRICS fragment
  - load_fragment_vocabulary: JSON → List[FragmentEntry]

Reference: Degen et al. (2008), "On the Art of Compiling and Using
'Drug-Like' Chemical Fragment Spaces", ChemMedChem, for the BRICS
isotope-to-bond-type mapping.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rdkit import Chem

# ══════════════════════════════════════════════════════════════════════
# BRICS isotope → bond-type mapping
# ══════════════════════════════════════════════════════════════════════
#
# BRICS dummy atoms carry isotope labels (1–16) that encode the chemical
# environment of the broken bond.  Two fragments can fuse only if their
# dummy-atom isotopes imply the *same* bond order.
#
# Reference: Degen et al. (2008), ChemMedChem.
# Values: 1 = SINGLE, 2 = DOUBLE, 3 = TRIPLE.

_BRICS_ISOTOPE_TO_BOND_TYPE: Dict[int, int] = {
    1: 1,  2: 3,  3: 1,  4: 1,
    5: 2,  6: 2,
    7: 1,  8: 1,  9: 1,  10: 1,
    11: 1, 12: 1, 13: 3, 14: 1, 15: 3, 16: 1,
}


def brincs_bond_order(isotope_a: int, isotope_b: int) -> int:
    """
    Return the bond order implied by a pair of BRICS dummy-atom isotope labels.

    Parameters
    ----------
    isotope_a, isotope_b : int
        BRICS environment types (1–16).

    Returns
    -------
    int
        Bond order: 1 (SINGLE), 2 (DOUBLE), or 3 (TRIPLE).
        Returns **0** if the pair is chemically incompatible (different
        implied bond orders, or one label is not a recognized BRICS isotope).
    """
    type_a = _BRICS_ISOTOPE_TO_BOND_TYPE.get(isotope_a, 0)
    type_b = _BRICS_ISOTOPE_TO_BOND_TYPE.get(isotope_b, 0)
    if type_a == 0 or type_b == 0:
        return 0
    return type_a if type_a == type_b else 0


# ══════════════════════════════════════════════════════════════════════
# FragmentEntry
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FragmentEntry:
    """
    Pre-computed properties of a single BRICS fragment.

    Each fragment is a connected subgraph obtained from BRICS decomposition
    of drug-like molecules.  Dummy atoms (``[n*]``) mark attachment points
    where the fragment was cleaved from its parent.

    Attributes
    ----------
    fragment_id : int
        Zero-based index; 0 = most frequent fragment in the vocabulary.
    smiles : str
        Canonical SMILES with BRICS dummy atoms preserved (e.g. ``[3*]C(=O)O``).
    frequency : int
        Occurrence count in the source dataset (ChEMBL).
    num_atoms : int
        Number of **heavy** (non-dummy, non-hydrogen) atoms.
    num_attachment_sites : int
        Number of BRICS dummy atoms (= number of attachment points).
    attachment_atom_indices : Tuple[int, ...]
        Index of each dummy atom within ``rdkit_mol``.  Length equals
        ``num_attachment_sites``.
    attachment_isotopes : Tuple[int, ...]
        BRICS environment type (1–16) for each site.  Index-aligned with
        ``attachment_atom_indices``.
    attachment_bond_types : Tuple[int, ...]
        Bond order (1/2/3) each site implies when fused.  Index-aligned.
    atom_types : Tuple[int, ...]
        Atomic numbers of the **real** atoms, ordered by their index in
        ``rdkit_mol``.  Length equals ``num_atoms``.
    internal_bonds : np.ndarray
        ``(num_atoms, num_atoms)`` bond-order matrix (uint8).  Excludes
        dummy atoms.
    internal_distances : np.ndarray
        ``(num_atoms, num_atoms)`` shortest-path distance matrix (uint8).
        Excludes dummy atoms.
    rdkit_mol : Chem.RWMol
        Pre-built editable molecule **with dummy atoms preserved**.
        This is a **template** — never mutate it directly; copy via
        ``Chem.RWMol(frag.rdkit_mol)`` before modification.
    """
    fragment_id: int
    smiles: str
    frequency: int
    num_atoms: int
    num_attachment_sites: int
    attachment_atom_indices: Tuple[int, ...]
    attachment_isotopes: Tuple[int, ...]
    attachment_bond_types: Tuple[int, ...]
    atom_types: Tuple[int, ...]
    internal_bonds: np.ndarray
    internal_distances: np.ndarray
    rdkit_mol: Chem.RWMol

    def __repr__(self) -> str:
        return (
            f"FragmentEntry("
            f"id={self.fragment_id}, "
            f"atoms={self.num_atoms}, "
            f"sites={self.num_attachment_sites}, "
            f"freq={self.frequency}, "
            f"smiles='{self.smiles}')"
        )


# ══════════════════════════════════════════════════════════════════════
# Vocabulary loader
# ══════════════════════════════════════════════════════════════════════

def load_fragment_vocabulary(
    json_path: str,
    top_k: Optional[int] = None,
) -> List[FragmentEntry]:
    """
    Load a BRICS fragment vocabulary from a JSON file.

    The JSON is produced by ``build_brics_vocab.py``.  Entries are assumed
    to be **frequency-sorted** (most frequent first, ``fragment_id``
    equal to list index).

    Parameters
    ----------
    json_path : str
        Path to the JSON file.
    top_k : int, optional
        If provided, retain only the first ``top_k`` entries.  Since the
        file is frequency-sorted, this selects the top-K most common
        fragments.  ``fragment_id`` values are preserved from the file
        (so they remain 0 … top_k−1).

    Returns
    -------
    List[FragmentEntry]
        Fragment vocabulary, ordered by descending frequency.
    """
    with open(json_path, "r") as fh:
        raw_list: List[dict] = json.load(fh)

    if top_k is not None:
        raw_list = raw_list[:top_k]

    entries: List[FragmentEntry] = []
    for raw in raw_list:
        # ── Build RWMol from SMILES (dummy atoms preserved) ──────
        mol = Chem.MolFromSmiles(raw["smiles"])
        if mol is None:
            raise ValueError(
                f"Fragment {raw['fragment_id']}: cannot parse "
                f"SMILES '{raw['smiles']}'"
            )
        rw_mol = Chem.RWMol(mol)

        # ── Stamp atom properties on the RWMol template ──────────
        # These properties survive RWMol copies (Chem.RWMol(other))
        # and are read by MoleculeDesign._rebuild_numpy_state_from_rdkit()
        # after structural mutations (InsertMol, AddBond, RemoveAtom).
        #
        # Real atoms get:   _frag_id
        # Dummy atoms get:  _frag_id, _brics_isotope, _brics_bond_type
        frag_id_prop = raw["fragment_id"]

        # Build lookup: dummy_atom_rdkit_index → (isotope, bond_type)
        dummy_lookup: Dict[int, Tuple[int, int]] = {}
        for i, idx in enumerate(raw["attachment_atom_indices"]):
            dummy_lookup[idx] = (
                raw["attachment_isotopes"][i],
                raw["attachment_bond_types"][i],
            )

        for atom in rw_mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetAtomicNum() == 0:
                # Dummy atom — store BRICS metadata for fusion
                # compatibility checks and site reconstruction
                isotope, bond_type = dummy_lookup.get(idx, (0, 0))
                atom.SetIntProp("_brics_isotope", isotope)
                atom.SetIntProp("_brics_bond_type", bond_type)
                atom.SetIntProp("_frag_id", frag_id_prop)
            else:
                # Real atom — store fragment provenance
                atom.SetIntProp("_frag_id", frag_id_prop)

        # ── Construct entry ──────────────────────────────────────
        entry = FragmentEntry(
            fragment_id=raw["fragment_id"],
            smiles=raw["smiles"],
            frequency=raw["frequency"],
            num_atoms=raw["num_atoms"],
            num_attachment_sites=raw["num_attachment_sites"],
            attachment_atom_indices=tuple(raw["attachment_atom_indices"]),
            attachment_isotopes=tuple(raw["attachment_isotopes"]),
            attachment_bond_types=tuple(raw["attachment_bond_types"]),
            atom_types=tuple(raw["atom_types"]),
            internal_bonds=np.array(raw["internal_bonds"], dtype=np.uint8),
            internal_distances=np.array(raw["internal_distances"], dtype=np.uint8),
            rdkit_mol=rw_mol,
        )

        # ── Defensive: verify fragment_id matches list position ──
        if entry.fragment_id != len(entries):
            print(
                f"WARNING: fragment_id={entry.fragment_id} != "
                f"list index={len(entries)} — "
                f"fragments may not be in expected order."
            )

        entries.append(entry)

    # ── Summary ──────────────────────────────────────────────────
    n = len(entries)
    if n > 0:
        print(
            f"Loaded {n} fragments from {json_path}"
            + (f"  (top_k={top_k})" if top_k is not None else "")
        )
        print(
            f"  Atoms:     "
            f"min={min(e.num_atoms for e in entries)}, "
            f"max={max(e.num_atoms for e in entries)}, "
            f"mean={np.mean([e.num_atoms for e in entries]):.1f}"
        )
        print(
            f"  Sites:     "
            f"min={min(e.num_attachment_sites for e in entries)}, "
            f"max={max(e.num_attachment_sites for e in entries)}, "
            f"mean={np.mean([e.num_attachment_sites for e in entries]):.1f}"
        )
        print(
            f"  Frequency: "
            f"max={entries[0].frequency}, "
            f"min={entries[-1].frequency}"
        )
    else:
        print(f"WARNING: Loaded 0 fragments from {json_path}")

    return entries