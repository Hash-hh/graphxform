import random
import numpy as np
import torch
from torch import nn
from rdkit import Chem

from config import MoleculeConfig
from core.abstracts import BaseTrajectory
from core.utils import softmax

from typing import Optional, List, Tuple


class MoleculeDesign(BaseTrajectory):
    """
    Environment for molecular design.

    Supports two action-space modes (controlled by ``config.use_fragment_action_space``):

    **Atomic mode (legacy):**
        Actions are chosen hierarchically in three levels:
            - Level 0: Terminate, create a new atom, or pick an existing atom.
            - Level 1: Pick a second atom for a bond decision.
            - Level 2: Pick the bond order (1–6).

    **Fragment mode (AMORTIX 2.0 / USES engine):**
        Actions are chosen hierarchically in three levels:
            - Level 0: Terminate, select a BRICS fragment from the vocabulary,
              or select an existing open attachment site (for cyclisation / site–site bonding).
            - Level 1: Select an attachment site on the incoming fragment,
              or a second open site on the scaffold.
            - Level 2: Select the scaffold attachment site for fusion,
              or select the bond type for site-to-site bonding (if both sites are flexible).

    The virtual atom (index 0) is present in both modes and connects to every
    real atom with a special bond index.

    Bond type encoding:
        0       — No bond (padding in adjacency matrix diagonal)
        1–6     — Single through hextuple
        7       — Virtual bond (virtual atom ↔ real atom)
        8       — Aromatic bond
        9       — Padding index (for batched tensors)
    """

    maximum_bond_order = 6
    virtual_bond_idx = 7
    aromatic_bond_idx = 8
    maximum_num_atoms_overall = 100
    bond_types = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.QUADRUPLE,
        5: Chem.rdchem.BondType.QUINTUPLE,
        6: Chem.rdchem.BondType.HEXTUPLE
    }

    # ════════════════════════════════════════════════════════════════
    # CONSTRUCTOR
    # ════════════════════════════════════════════════════════════════
    def __init__(
        self,
        config: MoleculeConfig,
        initial_fragment: Optional[int] = None,
        initial_atom: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        config : MoleculeConfig
            Configuration.  When ``config.use_fragment_action_space`` is True,
            ``config.fragment_vocabulary`` must be populated.
        initial_fragment : int, optional
            Fragment mode.  If ``None`` (default), the molecule starts
            as an **empty graph** (de-novo generation).  If an integer in
            ``[0, K-1]``, the molecule is initialised with that fragment.
        initial_atom : int, optional
            Atomic mode (legacy).  Vocabulary index of the seed atom.
            Raises ``RuntimeError`` if ``config.use_fragment_action_space``
            is True.
        """
        self.config = config

        # ── Atom vocabulary ───────────────────────────────────────
        self.atom_vocabulary = self.config.atom_vocabulary
        self.vocabulary_atom_idcs = list(range(1, len(self.atom_vocabulary) + 1))
        self.vocabulary_atom_names = list(self.atom_vocabulary.keys())
        self.vocabulary_valence = [-1] + [
            self.atom_vocabulary[x]["valence"] for x in self.vocabulary_atom_names
        ]
        self.atom_feasibility_mask = [
            not self.atom_vocabulary[x]["allowed"]
            for x in self.vocabulary_atom_names
        ]
        self.upper_limit_atoms = self.config.max_num_atoms

        # ── Build atom lookup: (atomic_num, charge, chiral) → vocab_idx ──
        self._build_atom_lookup()

        # ── Fragment vocabulary ────────────────────────────────────
        self.fragment_vocabulary = self.config.fragment_vocabulary
        self.K = len(self.fragment_vocabulary) if self.fragment_vocabulary else 0

        # ── Mode flag ──────────────────────────────────────────────
        self._is_fragment_mode = getattr(config, 'use_fragment_action_space', True)

        # ── Atomic mode (legacy) ──────────────────────────────────
        if initial_atom is not None:
            if self._is_fragment_mode:
                raise RuntimeError(
                    "initial_atom is not supported in fragment mode. "
                    "Use initial_fragment instead."
                )
            self._init_atomic_mode(initial_atom)
            return

        # ── Fragment mode ─────────────────────────────────────────
        if initial_fragment is None:
            # PATH A — De-Novo: empty molecular graph
            self.atoms = np.array([0], dtype=np.uint8)
            self.bonds = np.zeros((1, 1), dtype=np.uint8)
            self.rdkit_mol = Chem.RWMol()
            self.open_attachment_sites: List[Tuple[int, int, int, int]] = []
            self.atom_to_fragment: List[int] = [-1]
            self._numpy_to_rdkit = np.array([-1], dtype=np.int32)
            self._atom_has_open_site = np.array([0], dtype=np.uint8)
            self.initial_fragment = None
        else:
            # PATH B — Single-fragment seed
            assert 0 <= initial_fragment < self.K, (
                f"initial_fragment={initial_fragment} out of range [0, {self.K - 1}]"
            )
            self.initial_fragment = initial_fragment
            frag = self.fragment_vocabulary[initial_fragment]
            self.rdkit_mol = Chem.RWMol(frag.rdkit_mol)
            self._rebuild_numpy_state_from_rdkit()

        # ── Action-space indexing ──────────────────────────────────
        self.pick_existing_atoms_start_action_idx_lvl_0 = self.K + 1

        # ── Remaining state ────────────────────────────────────────
        self.synthesis_done = False
        self.smiles_string: Optional[str] = None
        self.current_objective = float("-inf")
        self.current_action_level = 0
        self.current_action_mask: Optional[np.ndarray] = None
        self.history: List[int] = []
        self.log_probs_history: List[float] = []
        self.objective: Optional[float] = None
        self.sa_score: float = 0.0
        self.infeasibility_flag: bool = False
        self._s_before_fragment_insertion: Optional[int] = None

        # ── Precomputed padding sizes ─────────────────────────────
        self._D_max = (
            max(f.num_attachment_sites for f in self.fragment_vocabulary)
            if self.fragment_vocabulary else 2
        )
        self._lvl0_pad_size = 1 + self.K + config.max_open_attachment_sites
        self._lvl1_pad_size = max(self._D_max, config.max_open_attachment_sites)
        self._lvl2_pad_size = max(config.max_open_attachment_sites, 3)

        self.prompt_smiles: Optional[str] = None
        self.update_action_mask()

    # ════════════════════════════════════════════════════════════════
    # ATOM LOOKUP — (atomic_num, charge, chiral) → vocabulary index
    # ════════════════════════════════════════════════════════════════

    def _build_atom_lookup(self):
        """Precompute mappings from atom properties to vocabulary indices."""
        self._atom_key_to_vocab_idx: dict = {}
        self._atomic_num_to_vocab_idx: dict = {}
        for i, atom_name in enumerate(self.vocabulary_atom_names):
            atom_info = self.atom_vocabulary[atom_name]
            vocab_idx = i + 1
            atomic_num = atom_info["atomic_number"]

            key = str(atomic_num)
            if "formal_charge" in atom_info:
                key += f"_{atom_info['formal_charge']}"
            if "chiral_tag" in atom_info:
                key += f"@{atom_info['chiral_tag']}"
            self._atom_key_to_vocab_idx[key] = vocab_idx

            if atomic_num not in self._atomic_num_to_vocab_idx:
                self._atomic_num_to_vocab_idx[atomic_num] = vocab_idx

    def _rdkit_atom_to_vocab_idx(self, atom: Chem.Atom) -> int:
        """Map an RDKit Atom to its vocabulary index."""
        atomic_num = atom.GetAtomicNum()
        formal_charge = int(atom.GetFormalCharge())
        chiral_tag = int(atom.GetChiralTag())

        key = str(atomic_num)
        if formal_charge != 0:
            key += f"_{formal_charge}"
        if chiral_tag != 0:
            key += f"@{chiral_tag}"

        if key in self._atom_key_to_vocab_idx:
            return self._atom_key_to_vocab_idx[key]
        if atomic_num in self._atomic_num_to_vocab_idx:
            return self._atomic_num_to_vocab_idx[atomic_num]
        raise ValueError(
            f"Unknown atom: atomic_num={atomic_num}, charge={formal_charge}, "
            f"chiral={chiral_tag}"
        )

    # ════════════════════════════════════════════════════════════════
    # ATOMIC MODE INIT (legacy)
    # ════════════════════════════════════════════════════════════════

    def _init_atomic_mode(self, initial_atom: int):
        """Legacy atomic-mode initialisation."""
        assert not self.atom_feasibility_mask[initial_atom - 1]
        assert initial_atom in self.vocabulary_atom_idcs

        self.initial_fragment = None
        self.atoms = np.array([0, initial_atom], dtype=np.uint8)
        self.bonds = np.zeros((2, 2), dtype=np.uint8)
        self.bonds[0, 1] = self.bonds[1, 0] = self.virtual_bond_idx
        self.rdkit_mol = Chem.RWMol()
        self.open_attachment_sites = []
        self.atom_to_fragment = [-1, -1]
        self._numpy_to_rdkit = np.array([-1, 0], dtype=np.int32)
        self._atom_has_open_site = np.array([0, 0], dtype=np.uint8)
        self.pick_existing_atoms_start_action_idx_lvl_0 = (
            len(self.vocabulary_atom_idcs) + 1
        )
        self._D_max = 2
        self._lvl0_pad_size = 1 + len(self.vocabulary_atom_idcs) + 100
        self._lvl1_pad_size = 100
        self._lvl2_pad_size = self.maximum_bond_order
        self.synthesis_done = False
        self.smiles_string = None
        self.current_objective = float("-inf")
        self.current_action_level = 0
        self.current_action_mask = None
        self.history = []
        self.log_probs_history = []
        self.objective = None
        self.sa_score = 0.0
        self.infeasibility_flag = False
        self._s_before_fragment_insertion = None
        self.prompt_smiles = None
        self.update_rdkit_mol(new_atom=initial_atom)
        self.update_action_mask()

    # ════════════════════════════════════════════════════════════════
    # ACTION MASKING — Fragment-based (AMORTIX 2.0)
    # ════════════════════════════════════════════════════════════════

    def update_action_mask(self):
        """Build the feasibility mask for the current action level."""
        if self.synthesis_done:
            self.current_action_mask = None
            return

        # ── Atomic mode (legacy) ──────────────────────────────────
        if not self._is_fragment_mode:
            self._update_action_mask_atomic()
            return

        # ── Fragment mode ──────────────────────────────────────────
        S = len(self.open_attachment_sites)
        atom_budget = self.upper_limit_atoms - (len(self.atoms) - 1)

        # ==============================================================
        # LEVEL 0
        # ==============================================================
        if self.current_action_level == 0:
            mask = np.zeros(1 + self.K + S, dtype=bool)

            # --- Terminate ---
            if len(self.atoms) <= 1:
                mask[0] = 1

            # --- Add fragment (actions 1 … K) ---
            if S == 0 and len(self.atoms) > 1:
                mask[1:1 + self.K] = 1
            else:
                for k in range(self.K):
                    frag = self.fragment_vocabulary[k]
                    if frag.num_atoms > atom_budget:
                        mask[1 + k] = 1
                        continue
                    if S > 0 and not self._has_any_compatible_site_pair(frag):
                        mask[1 + k] = 1

            # --- Pick existing open site (actions K+1 … K+S) ---
            if S < 2:
                mask[1 + self.K:] = 1
            else:
                for s in range(S):
                    has_partner = any(
                        self._sites_compatible(s, t)
                        for t in range(S) if t != s
                    )
                    if not has_partner:
                        mask[1 + self.K + s] = 1

            self.current_action_mask = mask

        # ==============================================================
        # LEVEL 1
        # ==============================================================
        elif self.current_action_level == 1:
            l0_action = self.history[-1]

            # --- Case 1A: Fragment chosen at Level 0 ---
            if 1 <= l0_action <= self.K:
                frag = self.fragment_vocabulary[l0_action - 1]
                D = frag.num_attachment_sites
                mask = np.zeros(D, dtype=bool)

                S_before = self._s_before_fragment_insertion

                if S_before is None or S_before == 0:
                    pass
                else:
                    for d in range(D):
                        compatible = any(
                            self._frag_site_compatible_with_scaffold_site(
                                frag, d, ss,
                            )
                            for ss in range(S_before)
                        )
                        if not compatible:
                            mask[d] = 1

                self.current_action_mask = mask

            # --- Case 1B: Existing open site chosen at Level 0 ---
            elif l0_action > self.K:
                source_idx = l0_action - (self.K + 1)
                mask = np.zeros(S, dtype=bool)
                mask[source_idx] = 1
                for t in range(S):
                    if t == source_idx:
                        continue
                    if not self._sites_compatible(source_idx, t):
                        mask[t] = 1
                self.current_action_mask = mask
            else:
                raise RuntimeError(
                    f"Unexpected Level 0 action {l0_action} at Level 1"
                )

        # ==============================================================
        # LEVEL 2
        # ==============================================================
        elif self.current_action_level == 2:
            l0_action = self.history[-2]

            # --- Case 2A: Fragment attachment ---
            if 1 <= l0_action <= self.K:
                frag = self.fragment_vocabulary[l0_action - 1]
                frag_site = self.history[-1]

                S_before = self._s_before_fragment_insertion

                if S_before is None or S_before == 0:
                    mask = np.zeros(0, dtype=bool)
                else:
                    mask = np.zeros(S_before, dtype=bool)
                    for s in range(S_before):
                        if not self._frag_site_compatible_with_scaffold_site(
                            frag, frag_site, s,
                        ):
                            mask[s] = 1

                self.current_action_mask = mask

            # --- Case 2B: Site-to-site bonding ---
            elif l0_action > self.K:
                source_idx = l0_action - (self.K + 1)
                target_idx = self.history[-1]

                site_a = self.open_attachment_sites[source_idx]
                site_b = self.open_attachment_sites[target_idx]
                bond_a = site_a[1]
                bond_b = site_b[1]

                if bond_a > 0 or bond_b > 0:
                    self.current_action_mask = np.zeros(1, dtype=bool)
                else:
                    max_a = self._site_max_bond_order(source_idx)
                    max_b = self._site_max_bond_order(target_idx)
                    max_order = min(max_a, max_b, 3)

                    mask = np.zeros(3, dtype=bool)
                    mask[max_order:] = 1
                    self.current_action_mask = mask
            else:
                raise RuntimeError(
                    f"Unexpected Level 0 action {l0_action} at Level 2"
                )

    def _update_action_mask_atomic(self):
        """Legacy atomic-mode action masking."""
        atom_valence = np.array([self.vocabulary_valence[x] for x in self.atoms])
        atom_valence_remaining = atom_valence - self.bonds[:, 1:].sum(axis=1)
        ex_action_idx = self.pick_existing_atoms_start_action_idx_lvl_0

        if self.current_action_level == 0:
            self.current_action_mask = np.zeros(
                len(self.vocabulary_atom_idcs) + len(self.atoms), dtype=bool
            )
            if len(self.atoms) <= 2:
                self.current_action_mask[0] = 1
            self.current_action_mask[1:ex_action_idx] = self.atom_feasibility_mask
            if (self.upper_limit_atoms is not None and
                    len(self.atoms) - 1 == self.upper_limit_atoms) or \
                    (not np.any(atom_valence_remaining[1:])):
                self.current_action_mask[1:ex_action_idx] = 1
            self.current_action_mask[ex_action_idx:][
                np.where(atom_valence_remaining[1:] <= 0)
            ] = 1
            bond_indicator = np.zeros_like(self.bonds[1:, 1:])
            bond_indicator[np.where(self.bonds[1:, 1:] == 0)] = 1
            np.fill_diagonal(bond_indicator, 0)
            # Use atleast_1d to handle the case where there's only 1 real atom
            # (squeeze() on a (1,1) array produces a 0d scalar, which np.where can't handle)
            has_free_nonneighbor = np.atleast_1d(
                np.matmul(
                    bond_indicator, (atom_valence_remaining[1:] > 0)[:, None]
                ).squeeze()
            )
            self.current_action_mask[ex_action_idx:][
                np.where(has_free_nonneighbor == 0)
            ] = 1

        elif self.current_action_level == 1:
            self.current_action_mask = np.zeros(len(self.atoms) - 1, dtype=bool)
            atom_picked = (
                len(self.atoms) - 2
                if self.history[-1] < ex_action_idx
                else self.history[-1] - ex_action_idx
            )
            self.current_action_mask[atom_picked] = 1
            self.current_action_mask[np.where(atom_valence_remaining[1:] < 1)] = 1
            self.current_action_mask[
                np.where(self.bonds[atom_picked + 1, 1:] > 0)
            ] = 1

        elif self.current_action_level == 2:
            self.current_action_mask = np.zeros(self.maximum_bond_order, dtype=bool)
            atom_picked_0 = (
                len(self.atoms) - 2
                if self.history[-2] < ex_action_idx
                else self.history[-2] - ex_action_idx
            )
            atom_picked_1 = self.history[-1]
            max_bond_order = min(
                atom_valence_remaining[atom_picked_0 + 1],
                atom_valence_remaining[atom_picked_1 + 1],
            )
            self.current_action_mask[int(max_bond_order):] = 1

    # ════════════════════════════════════════════════════════════════
    # ACTION-COUNT HELPERS
    # ════════════════════════════════════════════════════════════════

    def _num_actions_level_1(self) -> int:
        if self.current_action_level != 1 or len(self.history) == 0:
            return 0
        l0_action = self.history[-1]
        if 1 <= l0_action <= self.K:
            frag = self.fragment_vocabulary[l0_action - 1]
            return frag.num_attachment_sites
        elif l0_action > self.K:
            return len(self.open_attachment_sites)
        return 0

    def _num_actions_level_2(self) -> int:
        if self.current_action_level != 2 or len(self.history) < 2:
            return 0
        l0_action = self.history[-2]
        if 1 <= l0_action <= self.K:
            return self._s_before_fragment_insertion or 0
        elif l0_action > self.K:
            source_idx = l0_action - (self.K + 1)
            target_idx = self.history[-1]
            S = len(self.open_attachment_sites)
            if source_idx >= S or target_idx >= S:
                return 0
            site_a = self.open_attachment_sites[source_idx]
            site_b = self.open_attachment_sites[target_idx]
            if site_a[1] > 0 or site_b[1] > 0:
                return 1
            return 3
        return 0

    # ════════════════════════════════════════════════════════════════
    # RDKit MOLECULE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def update_rdkit_mol(
        self,
        new_atom: Optional[int] = None,
        set_bond: Optional[Tuple[int, int, int]] = None,
    ):
        """Updates the RDKit mol (atomic mode only)."""
        if new_atom is not None:
            atom_idx = new_atom
            atom_config = self.atom_vocabulary[
                self.vocabulary_atom_names[atom_idx - 1]
            ]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config:
                a.SetFormalCharge(atom_config["formal_charge"])
            if "chiral_tag" in atom_config:
                if atom_config["chiral_tag"] == 1:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
                elif atom_config["chiral_tag"] == 2:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            self.rdkit_mol.AddAtom(a)
        elif set_bond is not None:
            i, j, bond_order = set_bond
            self.rdkit_mol.AddBond(i, j, self.bond_types[bond_order])

    def _real_atom_numpy_idx_for_dummy(self, dummy_rd_idx: int) -> Optional[int]:
        """Return numpy index (1-based, 0=virtual) of the real atom bonded
        to the dummy at ``dummy_rd_idx``. Returns None if no real neighbor."""
        dummy = self.rdkit_mol.GetAtomWithIdx(dummy_rd_idx)
        for neighbor in dummy.GetNeighbors():
            if neighbor.GetAtomicNum() != 0:
                rd_neighbor = neighbor.GetIdx()
                for ni in range(1, len(self._numpy_to_rdkit)):
                    if self._numpy_to_rdkit[ni] == rd_neighbor:
                        return ni
        return None

    # ════════════════════════════════════════════════════════════════
    # STATE SYNCHRONISATION (AMORTIX 2.0)
    # ════════════════════════════════════════════════════════════════

    def _rebuild_numpy_state_from_rdkit(self):
        """
        Reconstruct numpy state from self.rdkit_mol.

        After building open_attachment_sites, relaxes fragment-sourced
        dummies (frag_id >= 0) by clearing their BRICS isotopes.
        This prevents the compatibility cascade where each attached
        fragment's remaining dummies restrict future fragment choices
        via isotope matching.

        Bond type is preserved; only the retrosynthetic environment
        (isotope) is relaxed to 0 (permissive).

        This ensures:
        - First attachment: fragment's original isotope checked against
          permissive scaffold sites → always compatible (bond type permitting)
        - Subsequent attachments: remaining fragment dummies are already
          permissive → no isotope cascade builds up
        - Bond order and valence constraints remain fully enforced
        """
        mol = self.rdkit_mol

        # Update property cache so GetIsAromatic / GetImplicitValence work
        mol.UpdatePropertyCache(strict=False)

        # ── Classify atoms ──────────────────────────────────────
        real_rd: List[int] = []
        dummy_rd: List[int] = []
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:
                dummy_rd.append(atom.GetIdx())
            else:
                real_rd.append(atom.GetIdx())

        n_real = len(real_rd)

        # ── RDKit → numpy index mapping ──────────────────────────
        rd_to_np: dict = {rd: ni for ni, rd in enumerate(real_rd, start=1)}

        # ── atoms: [virtual=0] + vocabulary indices ──────────────
        self.atoms = np.zeros(n_real + 1, dtype=np.uint8)
        self.atoms[0] = 0
        for ni, rd in enumerate(real_rd, start=1):
            self.atoms[ni] = self._rdkit_atom_to_vocab_idx(
                mol.GetAtomWithIdx(rd)
            )

        # ── _numpy_to_rdkit ─────────────────────────────────────
        self._numpy_to_rdkit = np.zeros(n_real + 1, dtype=np.int32)
        self._numpy_to_rdkit[0] = -1
        for ni, rd in enumerate(real_rd, start=1):
            self._numpy_to_rdkit[ni] = rd

        # ── bonds: virtual connections + real–real bonds ────────
        N = n_real + 1
        self.bonds = np.zeros((N, N), dtype=np.uint8)
        self.bonds[0, 1:] = self.virtual_bond_idx
        self.bonds[1:, 0] = self.virtual_bond_idx
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if i in rd_to_np and j in rd_to_np:
                ni, nj = rd_to_np[i], rd_to_np[j]
                if bond.GetIsAromatic():
                    order = self.aromatic_bond_idx
                else:
                    order = int(bond.GetBondTypeAsDouble())
                self.bonds[ni, nj] = order
                self.bonds[nj, ni] = order

        # ── atom_to_fragment ─────────────────────────────────────
        self.atom_to_fragment = [-1]
        for ni, rd in enumerate(real_rd, start=1):
            a = mol.GetAtomWithIdx(rd)
            fid = a.GetIntProp("_frag_id") if a.HasProp("_frag_id") else -1
            self.atom_to_fragment.append(fid)

        # ── open_attachment_sites ────────────────────────────────
        self.open_attachment_sites = []
        for rd in dummy_rd:
            a = mol.GetAtomWithIdx(rd)
            isotope = (
                a.GetIntProp("_brics_isotope")
                if a.HasProp("_brics_isotope") else 0
            )
            bond_type = (
                a.GetIntProp("_brics_bond_type")
                if a.HasProp("_brics_bond_type") else 0
            )
            fid = a.GetIntProp("_frag_id") if a.HasProp("_frag_id") else -1
            self.open_attachment_sites.append((rd, bond_type, isotope, fid))

        # ── _atom_has_open_site: which real atoms have dummy neighbors ──
        self._atom_has_open_site = np.zeros(n_real + 1, dtype=np.uint8)
        self._atom_has_open_site[0] = 0  # virtual atom
        for ni, rd in enumerate(real_rd, start=1):
            atom = mol.GetAtomWithIdx(rd)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetAtomicNum() == 0:
                    self._atom_has_open_site[ni] = 1
                    break

        # ════════════════════════════════════════════════════════════
        # RELAXATION: Clear BRICS isotopes on fragment-sourced dummies
        # ════════════════════════════════════════════════════════════
        # After a fragment is inserted into the scaffold, its remaining
        # dummies carry BRICS isotopes that encode the retrosynthetic
        # environment of the original bond. In de novo generation, we
        # don't need retrosynthetic matching — we only need forward
        # chemical validity (bond type + valence + sanitize).
        #
        # By clearing the isotope (setting to 0 = permissive), we prevent
        # the "compatibility cascade" where each attached fragment's
        # isotopes restrict future fragment choices, depleting the
        # action space.
        #
        # Bond type is PRESERVED (1/2/3) — this is a real chemical
        # constraint (single must match single, etc.).
        # Only the isotope (retrosynthetic label) is relaxed.
        #
        # This affects only fragment-sourced dummies (frag_id >= 0).
        # Scaffold-sourced dummies (from from_smiles) already have
        # isotope=0 and are unaffected.
        relaxed_sites = []
        for site in self.open_attachment_sites:
            rd_idx, bond_type, isotope, frag_id = site
            if isotope > 0 and frag_id >= 0:
                # Fragment-sourced dummy with BRICS isotope → relax
                atom = mol.GetAtomWithIdx(int(rd_idx))
                atom.SetIntProp("_brics_isotope", 0)
                relaxed_sites.append((rd_idx, bond_type, 0, frag_id))
            else:
                relaxed_sites.append(site)
        self.open_attachment_sites = relaxed_sites

    # ════════════════════════════════════════════════════════════════
    # ATTACHMENT-SITE COMPATIBILITY & VALENCE
    # ════════════════════════════════════════════════════════════════

    def _site_max_bond_order(self, site_idx: int) -> int:
        """
        Maximum bond order achievable at this attachment site.

        Formula: min(implicit_valence + dummy_bond_order, 3)

        - Zero-order dummy (scaffold):  implicit_val + 0 = implicit_val
        - Single-bond dummy (fragment): implicit_val + 1
        """
        site = self.open_attachment_sites[site_idx]
        dummy_rd = site[0]
        dummy = self.rdkit_mol.GetAtomWithIdx(dummy_rd)
        for neighbor in dummy.GetNeighbors():
            if neighbor.GetAtomicNum() != 0:
                bond = self.rdkit_mol.GetBondBetweenAtoms(
                    dummy_rd, neighbor.GetIdx()
                )
                dummy_bond_order = int(bond.GetBondTypeAsDouble())
                return min(
                    neighbor.GetImplicitValence() + dummy_bond_order, 3
                )
        return 0

    @staticmethod
    def _site_pair_compatible(
        bond_a: int, isotope_a: int,
        bond_b: int, isotope_b: int,
    ) -> bool:
        """Check whether two attachment sites are BRICS-compatible."""
        if isotope_a > 0 and isotope_b > 0:
            from core.fragment import brincs_bond_order
            if brincs_bond_order(isotope_a, isotope_b) == 0:
                return False
        if bond_a > 0 and bond_b > 0:
            return bond_a == bond_b
        return True

    def _frag_site_compatible_with_scaffold_site(
        self, frag, frag_site_idx: int, scaffold_site_idx: int,
    ) -> bool:
        """Check fragment site vs scaffold site compatibility."""
        scaffold_site = self.open_attachment_sites[scaffold_site_idx]

        if not self._site_pair_compatible(
                bond_a=frag.attachment_bond_types[frag_site_idx],
                isotope_a=frag.attachment_isotopes[frag_site_idx],
                bond_b=scaffold_site[1],
                isotope_b=scaffold_site[2],
        ):
            return False

        if scaffold_site[1] == 0:  # flexible
            frag_bond_type = frag.attachment_bond_types[frag_site_idx]
            if self._site_max_bond_order(scaffold_site_idx) < frag_bond_type:
                return False

        return True

    def _has_any_compatible_site_pair(self, frag) -> bool:
        """Check if any fragment site is compatible with any scaffold site."""
        S = len(self.open_attachment_sites)
        if S == 0:
            return False
        for f_site in range(frag.num_attachment_sites):
            for s_site in range(S):
                if self._frag_site_compatible_with_scaffold_site(
                    frag, f_site, s_site,
                ):
                    return True
        return False

    def _sites_compatible(self, site_idx_a: int, site_idx_b: int) -> bool:
        """
        Check whether two scaffold open attachment sites are mutually
        compatible (for site–site bonding / cyclisation).

        Checks:
          1. Different site indices
          2. Different real atoms (no self-loops)
          3. Real atoms not already bonded (no duplicate bonds)
          4. BRICS isotope compatibility
          5. Bond type compatibility + valence feasibility
        """
        if site_idx_a == site_idx_b:
            return False

        # Check different real atoms
        real_a = self._real_atom_numpy_idx_for_dummy(
            self.open_attachment_sites[site_idx_a][0]
        )
        real_b = self._real_atom_numpy_idx_for_dummy(
            self.open_attachment_sites[site_idx_b][0]
        )
        if real_a is not None and real_b is not None:
            if real_a == real_b:
                return False

            # Check if the real atoms are already bonded
            # Cast numpy.int32 to Python int for RDKit compatibility
            rd_a = int(self._numpy_to_rdkit[real_a])
            rd_b = int(self._numpy_to_rdkit[real_b])
            if self.rdkit_mol.GetBondBetweenAtoms(rd_a, rd_b) is not None:
                return False

        site_a = self.open_attachment_sites[site_idx_a]
        site_b = self.open_attachment_sites[site_idx_b]
        bond_a, iso_a = site_a[1], site_a[2]
        bond_b, iso_b = site_b[1], site_b[2]

        # BRICS isotope check
        if iso_a > 0 and iso_b > 0:
            from core.fragment import brincs_bond_order
            if brincs_bond_order(iso_a, iso_b) == 0:
                return False

        # Bond type check
        if bond_a > 0 and bond_b > 0:
            return bond_a == bond_b
        elif bond_a > 0:
            return self._site_max_bond_order(site_idx_b) >= bond_a
        elif bond_b > 0:
            return self._site_max_bond_order(site_idx_a) >= bond_b
        else:
            return self._site_max_bond_order(site_idx_a) >= 1 \
                and self._site_max_bond_order(site_idx_b) >= 1

    @staticmethod
    def _site_pair_bond_order(
        site_a: Tuple[int, int, int, int],
        site_b: Tuple[int, int, int, int],
    ) -> int:
        """Return bond order for connecting two sites where at least
        one has a fixed bond type."""
        bond_a, iso_a = site_a[1], site_a[2]
        bond_b, iso_b = site_b[1], site_b[2]

        if iso_a > 0 and iso_b > 0:
            from core.fragment import brincs_bond_order
            return brincs_bond_order(iso_a, iso_b)
        if bond_a > 0:
            return bond_a
        if bond_b > 0:
            return bond_b
        raise ValueError(
            "_site_pair_bond_order called when both sites are flexible. "
            "Bond order should be chosen by the policy at Level 2."
        )

    # ════════════════════════════════════════════════════════════════
    # FRAGMENT FUSION (AMORTIX 2.0)
    # ════════════════════════════════════════════════════════════════

    def fuse_fragment(
        self, fragment_idx: int, frag_site_idx: int, scaffold_site_idx: int,
    ):
        """
        Fuse a fragment onto the growing molecule.

        The fragment was already inserted into the RWMol at Level 0.
        This method bonds the real atoms adjacent to the two dummy
        attachment points, then removes both dummies.
        """
        S_before = self._s_before_fragment_insertion
        assert S_before is not None, (
            "_s_before_fragment_insertion is None at fuse_fragment"
        )

        # ── Fragment side: find dummy via open_attachment_sites ──
        frag_site_open_idx = S_before + frag_site_idx
        assert frag_site_open_idx < len(self.open_attachment_sites), (
            f"frag_site_open_idx={frag_site_open_idx} out of range "
            f"(len={len(self.open_attachment_sites)})"
        )

        frag_site = self.open_attachment_sites[frag_site_open_idx]
        frag_dummy_rd = frag_site[0]

        # Find fragment real atom (neighbor of dummy)
        frag_dummy_atom = self.rdkit_mol.GetAtomWithIdx(frag_dummy_rd)
        frag_real_rd = None
        for neighbor in frag_dummy_atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 0:
                frag_real_rd = neighbor.GetIdx()
                break
        assert frag_real_rd is not None, (
            f"Fragment site {frag_site_idx}: dummy atom has no real-atom neighbour"
        )

        # ── Scaffold side: find dummy and real-atom neighbour ────
        scaffold_site = self.open_attachment_sites[scaffold_site_idx]
        scaffold_dummy_rd = scaffold_site[0]

        scaffold_dummy_atom = self.rdkit_mol.GetAtomWithIdx(scaffold_dummy_rd)
        scaffold_real_rd = None
        for neighbor in scaffold_dummy_atom.GetNeighbors():
            if neighbor.GetAtomicNum() != 0:
                scaffold_real_rd = neighbor.GetIdx()
                break
        assert scaffold_real_rd is not None, (
            f"Scaffold site {scaffold_site_idx}: "
            f"dummy atom has no real-atom neighbour"
        )

        # ── Resolve bond type ────────────────────────────────────
        scaffold_bond_type = scaffold_site[1]
        frag_bond_type = frag_site[1]

        if scaffold_bond_type > 0:
            bond_type = scaffold_bond_type
        elif frag_bond_type > 0:
            bond_type = frag_bond_type
        else:
            bond_type = 1

        # ── Bond the REAL atoms ──────────────────────────────────
        self.rdkit_mol.AddBond(
            scaffold_real_rd,
            frag_real_rd,
            self.bond_types[bond_type],
        )

        # ── Remove both dummies (descending order) ───────────────
        first, second = sorted(
            [scaffold_dummy_rd, frag_dummy_rd], reverse=True,
        )
        self.rdkit_mol.RemoveAtom(first)
        self.rdkit_mol.RemoveAtom(second)

        # ── Rebuild numpy state ──────────────────────────────────
        self._rebuild_numpy_state_from_rdkit()

    # ════════════════════════════════════════════════════════════════
    # MASKED LOG-PROBABILITIES
    # ════════════════════════════════════════════════════════════════

    def masked_log_probs_for_current_action_level(
        self, logits: np.array
    ) -> np.array:
        mask = self.current_action_mask
        logits[mask] = -np.inf
        with np.errstate(divide='ignore'):
            log_probs = np.log(softmax(logits))
        return log_probs

    # ════════════════════════════════════════════════════════════════
    # ACTION EXECUTION — Fragment-based (AMORTIX 2.0)
    # ════════════════════════════════════════════════════════════════

    def take_action(self, action: int, log_prob: Optional[float] = None):
        """Execute an action at the current action level."""
        assert not self.synthesis_done, (
            "Taking action on already terminated design."
        )

        # ── Debug guard ─────────────────────────────────────────
        if self.current_action_mask[action] != 0:
            print(f"[DEBUG] take_action: About to fail assertion!", flush=True)
            print(f"[DEBUG]   Action: {action}, "
                  f"Level: {self.current_action_level}", flush=True)
            print(f"[DEBUG]   Mask (len {len(self.current_action_mask)}): "
                  f"{self.current_action_mask}", flush=True)

        assert self.current_action_mask[action] == 0, (
            f"Trying to take action {action} on level "
            f"{self.current_action_level}, but it is set to infeasible"
        )

        # ── Atomic mode dispatch ─────────────────────────────────
        if not self._is_fragment_mode:
            self._take_action_atomic(action, log_prob)
            return

        # ── Fragment mode ────────────────────────────────────────
        S = len(self.open_attachment_sites)
        next_level = 0

        # ==========================================================
        # LEVEL 0
        # ==========================================================
        if self.current_action_level == 0:
            if action == 0:
                self.synthesis_done = True
                self.finalize()

            elif 1 <= action <= self.K:
                frag_idx = action - 1
                frag = self.fragment_vocabulary[frag_idx]
                self._s_before_fragment_insertion = S
                self.rdkit_mol.InsertMol(frag.rdkit_mol)
                self._rebuild_numpy_state_from_rdkit()

                if S == 0:
                    next_level = 0
                    self._s_before_fragment_insertion = None
                else:
                    next_level = 1

            elif action > self.K:
                next_level = 1
            else:
                raise ValueError(f"Unexpected Level 0 action: {action}")

        # ==========================================================
        # LEVEL 1
        # ==========================================================
        elif self.current_action_level == 1:
            next_level = 2

        # ==========================================================
        # LEVEL 2
        # ==========================================================
        elif self.current_action_level == 2:
            l0_action = self.history[-2]
            l1_action = self.history[-1]

            # --- Case 2A: Fragment attachment ---
            if 1 <= l0_action <= self.K:
                frag_idx = l0_action - 1
                frag_site_idx = l1_action
                scaffold_site_idx = action

                self.fuse_fragment(
                    fragment_idx=frag_idx,
                    frag_site_idx=frag_site_idx,
                    scaffold_site_idx=scaffold_site_idx,
                )
                self._s_before_fragment_insertion = None
                next_level = 0

            # --- Case 2B: Site-to-site bonding ---
            elif l0_action > self.K:
                source_idx = l0_action - (self.K + 1)
                target_idx = l1_action

                site_a = self.open_attachment_sites[source_idx]
                site_b = self.open_attachment_sites[target_idx]
                bond_a = site_a[1]
                bond_b = site_b[1]

                # Determine bond order
                if bond_a > 0:
                    bond_order = bond_a
                elif bond_b > 0:
                    bond_order = bond_b
                else:
                    bond_order = action + 1

                # Find real atoms adjacent to dummies
                dummy_a = self.rdkit_mol.GetAtomWithIdx(site_a[0])
                real_a = None
                for neighbor in dummy_a.GetNeighbors():
                    if neighbor.GetAtomicNum() != 0:
                        real_a = neighbor.GetIdx()
                        break

                dummy_b = self.rdkit_mol.GetAtomWithIdx(site_b[0])
                real_b = None
                for neighbor in dummy_b.GetNeighbors():
                    if neighbor.GetAtomicNum() != 0:
                        real_b = neighbor.GetIdx()
                        break

                assert real_a is not None and real_b is not None, (
                    "Cannot find real atom neighbors of dummy atoms "
                    f"(rd_a={site_a[0]}, rd_b={site_b[0]})"
                )
                assert real_a != real_b, (
                    "Cannot bond an atom to itself via site-to-site bonding"
                )

                # Bond the REAL atoms
                self.rdkit_mol.AddBond(
                    real_a, real_b, self.bond_types[bond_order],
                )

                # Remove both dummies (descending order)
                first, second = sorted([site_a[0], site_b[0]], reverse=True)
                self.rdkit_mol.RemoveAtom(first)
                self.rdkit_mol.RemoveAtom(second)

                self._rebuild_numpy_state_from_rdkit()
                next_level = 0
            else:
                raise ValueError(
                    f"Unexpected Level 0 action {l0_action} at Level 2"
                )

        # ==========================================================
        # COMMON
        # ==========================================================
        self.history.append(int(action))
        if log_prob is not None:
            self.log_probs_history.append(log_prob)
        self.current_action_level = next_level
        self.update_action_mask()

    def _take_action_atomic(self, action: int, log_prob: Optional[float]):
        """Legacy atomic-mode action execution."""
        ex_action_idx = self.pick_existing_atoms_start_action_idx_lvl_0
        next_level = 0

        if self.current_action_level == 0:
            if action == 0:
                self.synthesis_done = True
                self.finalize()
            elif 1 <= action < ex_action_idx:
                self.atoms = np.append(self.atoms, action)
                self.bonds = np.pad(
                    self.bonds, [(0, 1), (0, 1)], mode='constant', constant_values=0
                )
                new_atom_idx = len(self.atoms) - 1
                self.bonds[0, new_atom_idx] = self.bonds[new_atom_idx, 0] = self.virtual_bond_idx
                self.update_rdkit_mol(new_atom=action)

                # FIX: Update _numpy_to_rdkit and _atom_has_open_site for the new atom
                # The new RDKit atom index is len(self.atoms) - 2 (since virtual atom is not in RDKit)
                self._numpy_to_rdkit = np.append(
                    self._numpy_to_rdkit, len(self.atoms) - 2
                )
                self._atom_has_open_site = np.append(
                    self._atom_has_open_site, 0
                )
                self.atom_to_fragment.append(-1)

                next_level = 1
            else:
                next_level = 1

        elif self.current_action_level == 1:
            next_level = 2

        elif self.current_action_level == 2:
            atom_a = self.history[-1]
            atom_b = self.history[-2]
            if atom_b < ex_action_idx:
                atom_b = len(self.atoms) - 2
            else:
                atom_b = atom_b - ex_action_idx
            bond_order = action + 1
            self.bonds[atom_a + 1, atom_b + 1] = self.bonds[atom_b + 1, atom_a + 1] = bond_order
            self.update_rdkit_mol(set_bond=(atom_a, atom_b, bond_order))

        self.history.append(int(action))
        if log_prob is not None:
            self.log_probs_history.append(log_prob)
        self.current_action_level = next_level
        self.update_action_mask()

    # ════════════════════════════════════════════════════════════════
    # FINALISATION & VALIDATION
    # ════════════════════════════════════════════════════════════════

    def finalize(self, assert_feasible: bool = False):
        """Called when terminating.  Removes remaining dummies, sanitizes
        the molecule, and creates the SMILES string."""
        if assert_feasible:
            self.assert_feasible()

        # Remove all remaining dummy atoms (unused attachment sites)
        dummy_indices = [
            a.GetIdx() for a in self.rdkit_mol.GetAtoms()
            if a.GetAtomicNum() == 0
        ]
        for idx in sorted(dummy_indices, reverse=True):
            self.rdkit_mol.RemoveAtom(idx)

        # Rebuild numpy state to reflect dummy removal
        if self._is_fragment_mode and dummy_indices:
            self._rebuild_numpy_state_from_rdkit()

        try:
            Chem.SanitizeMol(self.rdkit_mol)
        except Exception:
            self.infeasibility_flag = True

        if not self.infeasibility_flag:
            self.smiles_string = Chem.MolToSmiles(self.rdkit_mol)
            if self.smiles_string == "C":
                self.infeasibility_flag = True

    def assert_feasible(self):
        """Checks whether the current molecule is feasible."""
        assert self.atoms[0] == 0, "First atom should be virtual (0)"
        assert np.all([not self.atom_feasibility_mask[x - 1]
                        for x in self.atoms[1:]]) \
            and np.all(self.atoms[1:] > 0), \
            "Only allowed atoms permitted"
        assert self.upper_limit_atoms is None \
            or len(self.atoms) - 1 <= self.upper_limit_atoms, \
            "Exceeded maximum number of atoms"
        assert np.all(self.bonds[0, 1:] == self.virtual_bond_idx) \
            and np.all(self.bonds[1:, 0] == self.virtual_bond_idx), \
            "Virtual atom must be connected to all other atoms"
        assert not np.any(self.bonds.diagonal()), \
            "Atom may not be connected to itself"
        assert not np.any(self.bonds - self.bonds.T), \
            "Bond matrix must be symmetric"

        # Valence check — use RDKit's explicit valence (handles aromatic bonds)
        for ni in range(1, len(self.atoms)):
            rd_idx = int(self._numpy_to_rdkit[ni])
            atom = self.rdkit_mol.GetAtomWithIdx(rd_idx)
            explicit_val = atom.GetExplicitValence()
            max_val = self.vocabulary_valence[self.atoms[ni]]
            assert explicit_val <= max_val, (
                f"Atom {ni} (RDKit idx {rd_idx}) has explicit valence "
                f"{explicit_val} > max {max_val}"
            )

        if self.current_action_level == 0 and len(self.atoms) > 2:
            assert np.all(self.bonds[1:, 1:].sum(axis=1) > 0), \
                "An atom must be connected to at least one other atom"

    # ════════════════════════════════════════════════════════════════
    # UTILITY
    # ════════════════════════════════════════════════════════════════

    def to_rdkit_mol(self, sanitize=True) -> Chem.RWMol:
        """@Deprecated — use ``self.rdkit_mol`` directly."""
        mol = Chem.RWMol()
        num_atoms = len(self.atoms) - 1
        for atom_idx in self.atoms[1:]:
            atom_config = self.atom_vocabulary[
                self.vocabulary_atom_names[atom_idx - 1]
            ]
            a = Chem.Atom(atom_config["atomic_number"])
            if "formal_charge" in atom_config:
                a.SetFormalCharge(atom_config["formal_charge"])
            if "chiral_tag" in atom_config:
                if atom_config["chiral_tag"] == 1:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CW)
                elif atom_config["chiral_tag"] == 2:
                    a.SetChiralTag(Chem.CHI_TETRAHEDRAL_CCW)
            mol.AddAtom(a)
        bonds = self.bonds[1:, 1:]
        for i in range(num_atoms):
            for j in range(i, num_atoms):
                if bonds[i, j] > 0:
                    mol.AddBond(i, j, self.bond_types[bonds[i, j]])
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                self.infeasibility_flag = True
        return mol

    def is_terminable(self):
        return self.current_action_level == 0 and not self.synthesis_done

    def to_smiles(self) -> str:
        return Chem.MolToSmiles(self.rdkit_mol)

    # ════════════════════════════════════════════════════════════════
    # BaseTrajectory INTERFACE
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def init_batch_from_instance_list(
            config: MoleculeConfig,
            instances: List[int],
            network: nn.Module,
            device: torch.device,
    ):
        if getattr(config, 'use_fragment_action_space', False):
            return [
                MoleculeDesign(config=config, initial_fragment=frag)
                for frag in instances
            ]
        else:
            return [
                MoleculeDesign(config=config, initial_atom=atom)
                for atom in instances
            ]

    @staticmethod
    def log_probability_fn(
        trajectories: List['MoleculeDesign'],
        network: nn.Module,
    ) -> List[np.ndarray]:
        log_probs_to_return: List[np.ndarray] = []
        network.eval()
        with torch.no_grad():
            batch = MoleculeDesign.list_to_batch(
                molecules=trajectories, device=network.device,
            )
            batch_logits_per_level = list(network(batch))
            for lvl in range(3):
                batch_logits_per_level[lvl] = \
                    batch_logits_per_level[lvl].float().cpu().numpy()

            for i, mol in enumerate(trajectories):
                logits = batch_logits_per_level[mol.current_action_level][i]
                logits = logits[:len(mol.current_action_mask)]
                log_probs = mol.masked_log_probs_for_current_action_level(logits)
                if not np.isfinite(log_probs).all():
                    bad = ~np.isfinite(log_probs)
                    log_probs[bad] = -np.inf
                log_probs_to_return.append(log_probs)
        return log_probs_to_return

    @staticmethod
    def from_smiles_with_frozen_core(
        config: MoleculeConfig,
        smiles: str,
        frozen_smarts: List,
        do_finish: bool = False,
    ) -> 'MoleculeDesign':
        """
        Create a ``MoleculeDesign`` from a SMILES string, preserving
        specified substructures as a "frozen core".

        **Fragment mode only.**  Non-frozen atoms are removed, and
        zero-order-bonded dummies are added at the cut points, creating
        open attachment sites where the removed substructures used to be.
        The policy can then add new fragments at these sites.

        This enables:
        - **Substituent replacement:** Remove -OCH₃, add -OCF₃
        - **Scaffold hopping:** Remove old ring, add new ring
        - **Selective decoration:** Freeze core, decorate periphery
        - **Linker redesign:** Remove old linker, add new one

        SMARTS Instance Selection
        -------------------------
        ``frozen_smarts`` accepts a list where each entry is either:

        - ``str`` — freeze **all** instances of this pattern.
        - ``(str, int)`` — freeze only the match at this index.
        - ``(str, list[int])`` — freeze matches at these indices.

        Negative indices are supported (``-1`` = last match).

        Match ordering is deterministic for canonical SMILES: matches
        are sorted by their first atom index (via
        ``GetSubstructMatches(uniquify=True)``).

        Parameters
        ----------
        config : MoleculeConfig
            Configuration with ``use_fragment_action_space=True`` and
            ``fragment_vocabulary`` populated.
        smiles : str
            Input SMILES string.
        frozen_smarts : List[Union[str, Tuple[str, Union[int, List[int]]]]]
            SMARTS patterns identifying substructures to preserve.
            Atoms matching *any* pattern are frozen; all others are
            removed.  A single string is also accepted (wrapped in a
            list).
        do_finish : bool, default False
            If True, call ``finalize()`` immediately (for testing).

        Returns
        -------
        MoleculeDesign
            Initialized with the frozen scaffold + open attachment sites
            at the locations of removed substructures.

        Raises
        ------
        ValueError
            If SMILES is invalid, a SMARTS pattern is invalid, or no
            atoms match any SMARTS pattern.
        IndexError
            If an instance index is out of range.
        TypeError
            If an entry in ``frozen_smarts`` has an invalid format.

        Warns
        -----
        UserWarning
            If the frozen atoms are not connected in the input molecule.

        Examples
        --------
        Freeze all benzene rings:

        >>> design = MoleculeDesign.from_smiles_with_frozen_core(
        ...     config,
        ...     smiles="c1ccccc1Cc2ccccc2O",
        ...     frozen_smarts=["c1ccccc1"],
        ... )
        # Both benzene rings frozen; only -CH₂- and -OH removed

        Freeze only the first benzene ring (by index):

        >>> design = MoleculeDesign.from_smiles_with_frozen_core(
        ...     config,
        ...     smiles="c1ccccc1Cc2ccccc2O",
        ...     frozen_smarts=[("c1ccccc1", 0)],
        ... )
        # Only the first ring frozen; second ring + -CH₂- + -OH removed

        Freeze the first and third benzene rings:

        >>> design = MoleculeDesign.from_smiles_with_frozen_core(
        ...     config,
        ...     smiles="c1ccccc1Cc2ccccc2Cc3ccccc3",
        ...     frozen_smarts=[("c1ccccc1", [0, 2])],
        ... )

        Mix of all-instances and specific-instance patterns:

        >>> design = MoleculeDesign.from_smiles_with_frozen_core(
        ...     config,
        ...     smiles="c1ccccc1Oc2ccccc2C(=O)N",
        ...     frozen_smarts=["C(=O)N", ("c1ccccc1", 0)],
        ... )
        # All amide groups frozen; only first benzene ring frozen

        Use negative indexing to freeze the last match:

        >>> design = MoleculeDesign.from_smiles_with_frozen_core(
        ...     config,
        ...     smiles="c1ccccc1Cc2ccccc2O",
        ...     frozen_smarts=[("c1ccccc1", -1)],
        ... )
        # Only the last (second) benzene ring frozen

        Notes
        -----
        - Bond orders of removed bridge bonds are *not* preserved.  The
          zero-order dummy allows the policy to choose any valid bond
          order at the attachment site.
        - If a frozen atom had a double bond to a removed atom, the
          frozen atom retains 2 units of free valence at that site.
        - Frozen atoms that had implicit hydrogens also get attachment
          dummies for those free valence units, matching the behavior
          of ``from_smiles``.
        """
        # ── Accept a single SMARTS string for convenience ─────────
        if isinstance(frozen_smarts, str):
            frozen_smarts = [frozen_smarts]

        # ── 1. Parse and canonicalize input ────────────────────────
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                f"Invalid SMILES: '{smiles}' — RDKit could not parse it"
            )
        Chem.SanitizeMol(mol)
        canonical_smiles = Chem.MolToSmiles(mol)
        if smiles != canonical_smiles:
            mol = Chem.MolFromSmiles(canonical_smiles)
            Chem.SanitizeMol(mol)

        # ── 2. Resolve frozen atoms via the helper ────────────────
        frozen_atoms = MoleculeDesign._resolve_frozen_atoms(
            mol, frozen_smarts,
        )

        if not frozen_atoms:
            raise ValueError(
                "No atoms matched the frozen SMARTS patterns"
            )

        # ── 3. Check connectivity of frozen atoms ──────────────────
        start = min(frozen_atoms)
        visited = {start}
        queue = [start]
        while queue:
            current = queue.pop(0)
            atom = mol.GetAtomWithIdx(current)
            for neighbor in atom.GetNeighbors():
                nidx = neighbor.GetIdx()
                if nidx in frozen_atoms and nidx not in visited:
                    visited.add(nidx)
                    queue.append(nidx)

        if visited != frozen_atoms:
            import warnings
            warnings.warn(
                f"Frozen atoms are not connected "
                f"({len(visited)} of {len(frozen_atoms)} reachable via "
                f"frozen-only paths). The resulting scaffold will have "
                f"disconnected fragments. If this is intentional "
                f"(e.g., fragment linking), this warning can be ignored.",
                UserWarning,
                stacklevel=2,
            )

        # ── 4. Build RWMol and stamp properties ────────────────────
        rw_mol = Chem.RWMol(mol)

        for atom in rw_mol.GetAtoms():
            atom.SetIntProp("_frag_id", -1)

        # ── 5. Process bridge bonds (frozen ↔ non-frozen) ─────────
        bridge_bonds: List[Tuple[int, int]] = []
        for bond in rw_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            if (i in frozen_atoms) != (j in frozen_atoms):  # XOR
                bridge_bonds.append((i, j))

        dummy_indices: set = set()
        for i, j in bridge_bonds:
            frozen_idx = i if i in frozen_atoms else j

            rw_mol.RemoveBond(i, j)

            dummy = Chem.Atom(0)
            dummy.SetIntProp("_frag_id", -1)
            dummy.SetIntProp("_brics_isotope", 0)
            dummy.SetIntProp("_brics_bond_type", 0)
            dummy_idx = rw_mol.AddAtom(dummy)
            dummy_indices.add(dummy_idx)

            rw_mol.AddBond(
                frozen_idx, dummy_idx, Chem.BondType.ZERO,
            )

        # ── 6. Remove non-frozen, non-dummy atoms ─────────────────
        atoms_to_remove = sorted(
            [
                idx for idx in range(rw_mol.GetNumAtoms())
                if idx not in frozen_atoms and idx not in dummy_indices
            ],
            reverse=True,
        )
        for idx in atoms_to_remove:
            rw_mol.RemoveAtom(idx)

        # ── 7. Update property cache ───────────────────────────────
        rw_mol.UpdatePropertyCache(strict=False)

        # ── 8. Add dummies for remaining free valence ──────────────
        real_atoms_snapshot = [
            a for a in rw_mol.GetAtoms() if a.GetAtomicNum() != 0
        ]

        for atom in real_atoms_snapshot:
            free = atom.GetImplicitValence()
            for _ in range(free):
                dummy = Chem.Atom(0)
                dummy.SetIntProp("_frag_id", -1)
                dummy.SetIntProp("_brics_isotope", 0)
                dummy.SetIntProp("_brics_bond_type", 0)
                dummy_idx = rw_mol.AddAtom(dummy)
                rw_mol.AddBond(
                    atom.GetIdx(), dummy_idx, Chem.BondType.ZERO,
                )

        rw_mol.UpdatePropertyCache(strict=False)

        # ── 9. Delegate to shared initialization ──────────────────
        return MoleculeDesign._create_fragment_design(
            config=config,
            rw_mol=rw_mol,
            canonical_smiles=canonical_smiles,
            do_finish=do_finish,
        )

    # ════════════════════════════════════════════════════════════════
    # PRIVATE: Resolve SMARTS patterns + instance indices → atom set
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _resolve_frozen_atoms(
        mol: Chem.Mol,
        frozen_smarts: List,
    ) -> set:
        """
        Resolve a list of SMARTS patterns (with optional instance
        indices) into a set of frozen atom indices.

        Each entry in ``frozen_smarts`` can be:

        - ``str`` — freeze **all** instances of this pattern.
        - ``(str, int)`` — freeze only the match at this index.
        - ``(str, list[int])`` — freeze matches at these indices.

        Negative indices are supported (Python-style: ``-1`` = last
        match).

        Match ordering is determined by ``GetSubstructMatches`` with
        ``uniquify=True``: matches are sorted by their first atom index.
        For canonical SMILES this ordering is deterministic.

        Parameters
        ----------
        mol : Chem.Mol
            The molecule to search.
        frozen_smarts : List
            List of patterns and optional instance selectors.

        Returns
        -------
        set[int]
            Set of atom indices to freeze.

        Raises
        ------
        TypeError
            If an entry is not a str or (str, int/list) tuple.
        ValueError
            If a SMARTS pattern is invalid or doesn't match.
        IndexError
            If an instance index is out of range.
        """
        frozen_atoms: set = set()

        for entry in frozen_smarts:
            # ── Parse entry format ───────────────────────────────
            if isinstance(entry, str):
                pattern_str = entry
                instance_idcs = None  # None = all instances

            elif isinstance(entry, tuple) and len(entry) == 2:
                pattern_str, raw_idcs = entry

                if isinstance(raw_idcs, int):
                    instance_idcs = [raw_idcs]
                elif isinstance(raw_idcs, (list, tuple)):
                    instance_idcs = list(raw_idcs)
                else:
                    raise TypeError(
                        f"Instance indices must be int or list[int], "
                        f"got {type(raw_idcs).__name__}"
                    )
            else:
                raise TypeError(
                    f"Each entry must be a str or (str, int/list) tuple, "
                    f"got {type(entry).__name__}: {entry!r}"
                )

            # ── Parse SMARTS ──────────────────────────────────────
            pattern = Chem.MolFromSmarts(pattern_str)
            if pattern is None:
                raise ValueError(
                    f"Invalid SMARTS pattern: '{pattern_str}'"
                )

            # ── Get unique matches ────────────────────────────────
            # uniquify=True prevents overlapping rotated matches
            # (e.g., benzene matching 6 times in the same ring)
            matches = mol.GetSubstructMatches(pattern, uniquify=True)

            if not matches:
                raise ValueError(
                    f"SMARTS '{pattern_str}' does not match in molecule "
                    f"'{Chem.MolToSmiles(mol)}'"
                )

            # ── Select instances ──────────────────────────────────
            if instance_idcs is None:
                # Freeze all instances
                for match in matches:
                    frozen_atoms.update(match)
            else:
                n_matches = len(matches)
                for idx in instance_idcs:
                    # Support negative indexing
                    if idx < 0:
                        idx += n_matches
                    if idx < 0 or idx >= n_matches:
                        raise IndexError(
                            f"Instance index {idx} out of range for "
                            f"SMARTS '{pattern_str}' "
                            f"(has {n_matches} matches, valid: "
                            f"[{-n_matches}, {n_matches - 1}])"
                        )
                    frozen_atoms.update(matches[idx])

        return frozen_atoms

    # ════════════════════════════════════════════════════════════════
    # PRIVATE: Fragment-mode design creation from a prepared RWMol
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _create_fragment_design(
        config: MoleculeConfig,
        rw_mol: Chem.RWMol,
        canonical_smiles: str,
        do_finish: bool = False,
    ) -> 'MoleculeDesign':
        """
        Create a fragment-mode ``MoleculeDesign`` from a prepared RWMol.

        The RWMol must already have:
        - Real atoms with ``_frag_id`` property set to -1
        - Zero-order-bonded dummies at attachment points
        - ``UpdatePropertyCache(strict=False)`` called

        This is shared between ``from_smiles`` and
        ``from_smiles_with_frozen_core`` to avoid code duplication.
        """
        design = MoleculeDesign.__new__(MoleculeDesign)

        # Shared / immutable references
        design.config = config
        design.atom_vocabulary = config.atom_vocabulary
        design.vocabulary_atom_idcs = list(
            range(1, len(config.atom_vocabulary) + 1),
        )
        design.vocabulary_atom_names = list(config.atom_vocabulary.keys())
        design.vocabulary_valence = [-1] + [
            config.atom_vocabulary[x]["valence"]
            for x in design.vocabulary_atom_names
        ]
        design.atom_feasibility_mask = [
            not config.atom_vocabulary[x]["allowed"]
            for x in design.vocabulary_atom_names
        ]
        design.upper_limit_atoms = config.max_num_atoms
        design._is_fragment_mode = True
        design._build_atom_lookup()

        design.fragment_vocabulary = config.fragment_vocabulary
        design.K = (
            len(config.fragment_vocabulary)
            if config.fragment_vocabulary else 0
        )

        # Attach RWMol and rebuild numpy state
        design.rdkit_mol = rw_mol
        design._rebuild_numpy_state_from_rdkit()

        # Remaining state
        design.initial_fragment = None
        design.pick_existing_atoms_start_action_idx_lvl_0 = design.K + 1
        design.synthesis_done = False
        design.smiles_string = None
        design.current_objective = float("-inf")
        design.current_action_level = 0
        design.current_action_mask = None
        design.history = []
        design.log_probs_history = []
        design.objective = None
        design.sa_score = 0.0
        design.infeasibility_flag = False
        design._s_before_fragment_insertion = None
        design._D_max = (
            max(f.num_attachment_sites for f in design.fragment_vocabulary)
            if design.fragment_vocabulary else 2
        )
        design._lvl0_pad_size = (
            1 + design.K + config.max_open_attachment_sites
        )
        design._lvl1_pad_size = max(
            design._D_max, config.max_open_attachment_sites,
        )
        design._lvl2_pad_size = max(
            config.max_open_attachment_sites, 3,
        )
        design.prompt_smiles = canonical_smiles if not do_finish else None

        design.update_action_mask()
        return design

    # ════════════════════════════════════════════════════════════════
    # SHALLOW CLONE
    # ════════════════════════════════════════════════════════════════

    def _shallow_clone(self) -> 'MoleculeDesign':
        new = self.__class__.__new__(self.__class__)

        # ── Immutable / shared ────────────────────────────────────
        new.config = self.config
        new.atom_vocabulary = self.atom_vocabulary
        new.vocabulary_atom_idcs = self.vocabulary_atom_idcs
        new.vocabulary_atom_names = self.vocabulary_atom_names
        new.vocabulary_valence = self.vocabulary_valence
        new.atom_feasibility_mask = self.atom_feasibility_mask
        new.upper_limit_atoms = self.upper_limit_atoms

        new._atom_key_to_vocab_idx = self._atom_key_to_vocab_idx
        new._atomic_num_to_vocab_idx = self._atomic_num_to_vocab_idx

        new.fragment_vocabulary = self.fragment_vocabulary
        new.K = self.K
        new._is_fragment_mode = self._is_fragment_mode

        # ── Mutable state (copy) ──────────────────────────────────
        new.atoms = self.atoms.copy()
        new.bonds = self.bonds.copy()
        new.rdkit_mol = Chem.RWMol(self.rdkit_mol)
        new._numpy_to_rdkit = self._numpy_to_rdkit.copy()
        new.open_attachment_sites = self.open_attachment_sites.copy()
        new.atom_to_fragment = self.atom_to_fragment.copy()
        new._atom_has_open_site = self._atom_has_open_site.copy()

        new.synthesis_done = self.synthesis_done
        new.smiles_string = self.smiles_string
        new.current_objective = self.current_objective
        new.current_action_level = self.current_action_level
        new.current_action_mask = (
            None if self.current_action_mask is None
            else self.current_action_mask.copy()
        )
        new.history = self.history.copy()
        new.log_probs_history = self.log_probs_history.copy()
        new.objective = self.objective
        new.sa_score = self.sa_score
        new.infeasibility_flag = self.infeasibility_flag
        new._s_before_fragment_insertion = self._s_before_fragment_insertion
        new._D_max = self._D_max
        new._lvl0_pad_size = self._lvl0_pad_size
        new._lvl1_pad_size = self._lvl1_pad_size
        new._lvl2_pad_size = self._lvl2_pad_size
        new.prompt_smiles = self.prompt_smiles

        new.pick_existing_atoms_start_action_idx_lvl_0 = \
            self.pick_existing_atoms_start_action_idx_lvl_0
        new.initial_fragment = self.initial_fragment
        return new

    def transition_fn(
        self, action: int, log_prob: Optional[float] = None,
    ) -> Tuple['BaseTrajectory', bool]:
        copied_molecule = self._shallow_clone()
        copied_molecule.take_action(action, log_prob)
        return copied_molecule, copied_molecule.synthesis_done

    def to_max_evaluation_fn(self) -> float:
        if self.objective is None:
            raise ValueError(
                "Objective is ``None``. Evaluate molecule with "
                "``MoleculeObjectiveEvaluator`` first."
            )
        return self.objective

    def num_actions(self) -> int:
        return int((1 - self.current_action_mask).sum())

    # ════════════════════════════════════════════════════════════════
    # BATCHING
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def list_to_batch(
        molecules: List['MoleculeDesign'],
        device: torch.device = None,
        include_feasibility_masks: bool = False,
    ) -> dict:
        """
        Given a list of molecule designs, prepares a batch for the network.

        ``picked_atom_mhe`` uses 3 values:
            0 = padding / nothing
            1 = picked at L0 (fragment atoms or source site)
            2 = picked at L1 (fragment site or target site)

        ``open_sites_mask`` indicates which atoms have open attachment
        sites (dummy neighbors).  0 = no open site, 1 = has open site.
        """
        atoms_padding_idx = len(molecules[0].vocabulary_atom_idcs) + 1
        degree_padding_idx = max(molecules[0].vocabulary_valence) + 1
        bond_padding_idx = MoleculeDesign.aromatic_bond_idx + 1  # = 9

        device = torch.device("cpu") if device is None else device
        num_atoms = [len(mol.atoms) for mol in molecules]
        max_num_atoms = max(num_atoms)

        batch_level_idx = [mol.current_action_level == 0 for mol in molecules]

        # ════════════════════════════════════════════════════════════
        # picked_atom_mhe — marks atoms picked in the current cycle
        # ════════════════════════════════════════════════════════════
        batch_picked_atom_mhe = np.zeros(
            (len(molecules), max_num_atoms), dtype=int,
        )
        K = molecules[0].K

        for i, mol in enumerate(molecules):
            if mol.current_action_level == 0:
                pass

            elif mol.current_action_level == 1 and mol.history:
                l0 = mol.history[-1]

                if 1 <= l0 <= K:
                    frag_idx = l0 - 1
                    frag = mol.fragment_vocabulary[frag_idx]
                    n_frag_atoms = frag.num_atoms
                    if n_frag_atoms > 0:
                        start = len(mol.atoms) - n_frag_atoms
                        end = len(mol.atoms)
                        batch_picked_atom_mhe[i, start:end] = 1

                elif l0 > K:
                    source_site_idx = l0 - (K + 1)
                    if source_site_idx < len(mol.open_attachment_sites):
                        site = mol.open_attachment_sites[source_site_idx]
                        np_idx = mol._real_atom_numpy_idx_for_dummy(site[0])
                        if np_idx is not None and np_idx < max_num_atoms:
                            batch_picked_atom_mhe[i, np_idx] = 1

            elif mol.current_action_level == 2 and len(mol.history) >= 2:
                l0 = mol.history[-2]
                l1 = mol.history[-1]

                if 1 <= l0 <= K:
                    # Mark fragment atoms with 1
                    frag_idx = l0 - 1
                    frag = mol.fragment_vocabulary[frag_idx]
                    n_frag_atoms = frag.num_atoms
                    if n_frag_atoms > 0:
                        start = len(mol.atoms) - n_frag_atoms
                        end = len(mol.atoms)
                        batch_picked_atom_mhe[i, start:end] = 1

                    # Mark the FRAGMENT SITE's real atom with 2
                    sb = mol._s_before_fragment_insertion
                    if sb is not None:
                        frag_site_open_idx = sb + l1
                        if frag_site_open_idx < len(mol.open_attachment_sites):
                            site = mol.open_attachment_sites[frag_site_open_idx]
                            np_idx = mol._real_atom_numpy_idx_for_dummy(site[0])
                            if np_idx is not None and np_idx < max_num_atoms:
                                batch_picked_atom_mhe[i, np_idx] = 2

                elif l0 > K:
                    source_idx = l0 - (K + 1)
                    target_idx = l1
                    for which, site_idx in [(1, source_idx), (2, target_idx)]:
                        if site_idx < len(mol.open_attachment_sites):
                            site = mol.open_attachment_sites[site_idx]
                            np_idx = mol._real_atom_numpy_idx_for_dummy(
                                site[0])
                            if np_idx is not None and np_idx < max_num_atoms:
                                batch_picked_atom_mhe[i, np_idx] = which

        # ════════════════════════════════════════════════════════════
        # Standard tensors
        # ════════════════════════════════════════════════════════════
        batch_atoms = np.stack([
            np.concatenate((
                mol.atoms,
                np.full(max_num_atoms - num_atoms[i],
                        fill_value=atoms_padding_idx, dtype=int),
            ))
            for i, mol in enumerate(molecules)
        ])

        batch_atoms_degree = np.stack([
            np.concatenate((
                (mol.bonds > 0).sum(axis=1) - 1,
                np.full(max_num_atoms - num_atoms[i],
                        fill_value=degree_padding_idx, dtype=int),
            ))
            for i, mol in enumerate(molecules)
        ])

        # ── open_sites_mask: which atoms have open attachment sites ──
        batch_open_sites = np.stack([
            np.concatenate((
                mol._atom_has_open_site,
                np.zeros(max_num_atoms - num_atoms[i], dtype=int),
            ))
            for i, mol in enumerate(molecules)
        ])

        bonds_list = []
        for i, mol in enumerate(molecules):
            padded_bonds = np.pad(
                mol.bonds,
                [(0, max_num_atoms - num_atoms[i]),
                 (0, max_num_atoms - num_atoms[i])],
                mode="constant", constant_values=bond_padding_idx,
            )
            np.fill_diagonal(padded_bonds, bond_padding_idx)
            bonds_list.append(padded_bonds)
        batch_bonds = np.stack(bonds_list)

        additive_padding_masks = []
        for i, mol in enumerate(molecules):
            mask = np.zeros_like(mol.bonds).astype(float)
            mask = np.pad(
                mask,
                [(0, max_num_atoms - num_atoms[i]),
                 (0, max_num_atoms - num_atoms[i])],
                mode="constant", constant_values=-np.inf,
            )
            np.fill_diagonal(mask, 0)
            additive_padding_masks.append(mask)
        batch_additive_padding_attn_mask = np.stack(additive_padding_masks)

        return_dict = dict(
            level_idx=torch.tensor(batch_level_idx, dtype=torch.long,
                                   device=device),
            picked_atom_mhe=torch.from_numpy(batch_picked_atom_mhe)
                .long().to(device),
            num_atoms=torch.tensor(num_atoms, dtype=torch.long,
                                   device=device),
            atoms=torch.from_numpy(batch_atoms).long().to(device),
            atoms_degree=torch.from_numpy(batch_atoms_degree).long()
                .to(device),
            open_sites_mask=torch.from_numpy(batch_open_sites).long()
                .to(device),
            bonds=torch.from_numpy(batch_bonds).long().to(device),
            additive_padding_attn_mask=torch.from_numpy(
                batch_additive_padding_attn_mask,
            ).float().to(device),
        )

        # ════════════════════════════════════════════════════════════
        # Feasibility masks — fixed global padding
        # ════════════════════════════════════════════════════════════
        if include_feasibility_masks:
            lvl0_pad = molecules[0]._lvl0_pad_size
            lvl1_pad = molecules[0]._lvl1_pad_size
            lvl2_pad = molecules[0]._lvl2_pad_size

            for lvl, pad_size, label in [
                (0, lvl0_pad, "feasibility_mask_level_zero"),
                (1, lvl1_pad, "feasibility_mask_level_one"),
                (2, lvl2_pad, "feasibility_mask_level_two"),
            ]:
                masks = []
                for mol in molecules:
                    if mol.current_action_level == lvl:
                        raw = mol.current_action_mask
                        padded = np.ones(pad_size, dtype=bool)
                        copy_len = min(len(raw), pad_size)
                        padded[:copy_len] = raw[:copy_len]
                        masks.append(padded)
                    else:
                        masks.append(np.zeros(pad_size, dtype=bool))
                return_dict[label] = torch.from_numpy(
                    np.stack(masks)
                ).bool().to(device)

        return return_dict

    @staticmethod
    def batch_to_device(batch: dict, device: torch.device):
        return {k: v.to(device) for k, v in batch.items()}

    # ════════════════════════════════════════════════════════════════
    # INITIALISATION HELPERS — Fragment mode
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def get_empty_graph(config: MoleculeConfig) -> 'MoleculeDesign':
        return MoleculeDesign(config, initial_fragment=None)

    @staticmethod
    def get_empty_graph_batch(
        config: MoleculeConfig, batch_size: int,
    ) -> List['MoleculeDesign']:
        template = MoleculeDesign(config, initial_fragment=None)
        return [template._shallow_clone() for _ in range(batch_size)]

    @staticmethod
    def get_top_fragment_seeds(
        config: MoleculeConfig,
        top_n: int = 10,
        repeat: int = 1,
    ) -> List['MoleculeDesign']:
        K = len(config.fragment_vocabulary)
        n = min(top_n, K)
        instances = []
        for k in range(n):
            for _ in range(repeat):
                instances.append(k)
        return MoleculeDesign.init_batch_from_instance_list(
            config, instances, None, None,
        )

    # ════════════════════════════════════════════════════════════════
    # LEGACY HELPERS (atomic mode)
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def get_c_chains(config: MoleculeConfig) -> List['MoleculeDesign']:
        """[Atomic mode] Carbon-chain starting points."""
        if getattr(config, 'use_fragment_action_space', False):
            raise RuntimeError(
                "get_c_chains is not supported in fragment mode. "
                "Use get_empty_graph or get_top_fragment_seeds instead."
            )
        carbon_atom_idx = list(config.atom_vocabulary.keys()).index("C") + 1
        instance_list = []
        for num_c_to_add in range(
            min(config.max_num_atoms - 1, config.start_c_chain_max_len)
        ):
            mol = MoleculeDesign(config, initial_atom=1)
            for i in range(num_c_to_add):
                mol.take_action(carbon_atom_idx)
                mol.take_action(len(mol.atoms) - 3)
                mol.take_action(0)
            instance_list.append(mol)
        return instance_list

    @staticmethod
    def get_single_atom_molecules(
        config: MoleculeConfig, repeat: int = 1,
    ) -> List['MoleculeDesign']:
        """[Atomic mode] Single-atom starting points."""
        if getattr(config, 'use_fragment_action_space', False):
            raise RuntimeError(
                "get_single_atom_molecules is not supported in fragment mode. "
                "Use get_empty_graph or get_top_fragment_seeds instead."
            )
        atoms = []
        for i, atom in enumerate(config.atom_vocabulary.keys()):
            if config.atom_vocabulary[atom]["allowed"]:
                atoms.append(i + 1)
        return MoleculeDesign.init_batch_from_instance_list(
            config, atoms * repeat, None, None,
        )

    @staticmethod
    def random_atom_order_in_smiles(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES input.")
        num_atoms = mol.GetNumAtoms()
        atom_indices = list(range(num_atoms))
        random.shuffle(atom_indices)
        reordered_mol = Chem.RenumberAtoms(mol, atom_indices)
        return Chem.MolToSmiles(reordered_mol, isomericSmiles=True,
                                canonical=False)

    # ════════════════════════════════════════════════════════════════
    # SMILES → MoleculeDesign
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def from_smiles(
        config: MoleculeConfig,
        smiles: str,
        do_finish: bool = False,
        compare_smiles: bool = False,
    ) -> 'MoleculeDesign':
        """
        Create a ``MoleculeDesign`` from a SMILES string.

        In **fragment mode** the molecule is preserved intact and
        attachment dummies are added at free-valence positions using
        **zero-order bonds** (preserving implicit valence for bond-order
        flexibility).

        In **atomic mode** the molecule is reconstructed atom-by-atom.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(
                f"Invalid SMILES: '{smiles}' — RDKit could not parse it"
            )
        Chem.SanitizeMol(mol)
        canonical_smiles = Chem.MolToSmiles(mol)
        if smiles != canonical_smiles:
            mol = Chem.MolFromSmiles(canonical_smiles)
            Chem.SanitizeMol(mol)

        # ── Atomic mode: delegate to legacy path ─────────────────
        if not getattr(config, 'use_fragment_action_space', False):
            design = MoleculeDesign.from_rdkit_mol(
                config, mol, canonical_smiles, do_finish, compare_smiles,
            )
            if not do_finish:
                design.prompt_smiles = canonical_smiles
            return design

        # ── Fragment mode: build RWMol with zero-order dummies ────
        rw_mol = Chem.RWMol(mol)

        # Stamp _frag_id = -1 on every existing atom
        for atom in rw_mol.GetAtoms():
            atom.SetIntProp("_frag_id", -1)

        # Snapshot real atoms before adding dummies (avoid live-iterator bug)
        real_atoms_snapshot = [
            a for a in rw_mol.GetAtoms() if a.GetAtomicNum() != 0
        ]

        for atom in real_atoms_snapshot:
            free = atom.GetImplicitValence()
            for _ in range(free):
                dummy = Chem.Atom(0)
                dummy.SetIntProp("_frag_id", -1)
                dummy.SetIntProp("_brics_isotope", 0)
                dummy.SetIntProp("_brics_bond_type", 0)
                dummy_idx = rw_mol.AddAtom(dummy)
                rw_mol.AddBond(
                    atom.GetIdx(), dummy_idx,
                    Chem.BondType.ZERO,
                )

        rw_mol.UpdatePropertyCache(strict=False)

        # ── Delegate to shared initialization ─────────────────────
        return MoleculeDesign._create_fragment_design(
            config=config,
            rw_mol=rw_mol,
            canonical_smiles=canonical_smiles,
            do_finish=do_finish,
        )

    @staticmethod
    def from_rdkit_mol(
        config: MoleculeConfig,
        rdkit_mol: Chem.RWMol,
        smiles: str,
        do_finish=True,
        compare_smiles=True,
    ) -> 'MoleculeDesign':
        """
        Creates an instance of ``MoleculeDesign`` from an RDKit molecule.

        **Atomic mode only.** In fragment mode, use ``from_smiles`` instead.
        """
        if getattr(config, 'use_fragment_action_space', False):
            raise RuntimeError(
                "from_rdkit_mol is not supported in fragment mode. "
                "Use from_smiles instead."
            )

        Chem.Kekulize(rdkit_mol)
        atoms = rdkit_mol.GetAtoms()
        atom_idcs_for_design = []
        adjacency_matrix: np.ndarray = Chem.rdmolops.GetAdjacencyMatrix(
            rdkit_mol, useBO=True,
        )

        atomic_num_to_atom_idx = dict()
        for i, atom_name in enumerate(config.atom_vocabulary.keys()):
            k = config.atom_vocabulary[atom_name]["atomic_number"]
            if "formal_charge" in config.atom_vocabulary[atom_name]:
                k = f"{k}_{config.atom_vocabulary[atom_name]['formal_charge']}"
            if "chiral_tag" in config.atom_vocabulary[atom_name]:
                k = f"{k}@{config.atom_vocabulary[atom_name]['chiral_tag']}"
            atomic_num_to_atom_idx[k] = i + 1

        for atom in atoms:
            k = atom.GetAtomicNum()
            formal_charge = int(atom.GetFormalCharge())
            if formal_charge != 0:
                k = f"{k}_{formal_charge}"
            chiral_tag = int(atom.GetChiralTag())
            if chiral_tag != 0:
                k = f"{k}@{chiral_tag}"
            atom_idx = atomic_num_to_atom_idx[k]
            atom_idcs_for_design.append(atom_idx)

        design = MoleculeDesign(config, initial_atom=atom_idcs_for_design[0])
        for i in range(1, len(atom_idcs_for_design)):
            atom_to_add = atom_idcs_for_design[i]
            atom_is_placed = False
            for j in range(0, i):
                desired_bond_order = adjacency_matrix[i, j]
                if desired_bond_order > 0:
                    if not atom_is_placed:
                        design.take_action(atom_to_add)
                        atom_is_placed = True
                    else:
                        design.take_action(
                            1 + len(config.atom_vocabulary.keys())
                            + len(design.atoms) - 2
                        )
                    design.take_action(j)
                    design.take_action(int(desired_bond_order - 1))

        if do_finish:
            design.take_action(0)
            if compare_smiles:
                assert Chem.CanonSmiles(design.smiles_string) == \
                    Chem.CanonSmiles(smiles), \
                    f"Converted: {Chem.CanonSmiles(design.smiles_string)}, " \
                    f"RDKit: {Chem.CanonSmiles(smiles)}"
            design.assert_feasible()
        else:
            design.history = []
            design.log_probs_history = []

        return design