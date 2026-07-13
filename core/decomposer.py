"""
Pre-training trajectory decomposer for AMORTIX 2.0 (USES engine).

Decomposes target SMILES into fragment action sequences for supervised
pre-training of the MoleculeTransformer.

Algorithm:
  1. BRICS decompose the target molecule into fragments
  2. Match each fragment to the vocabulary (by canonical SMILES)
  3. Build a fragment connection graph
  4. Find a spanning tree (assembly order) + ring-closure edges
  5. Simulate assembly using MoleculeDesign, recording actions
"""

from typing import List, Optional, Tuple, Dict
from collections import defaultdict, deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

from config import MoleculeConfig
from core.fragment import brincs_bond_order
from molecule_design import MoleculeDesign


class TrajectoryDecomposer:
    """
    Decomposes target SMILES into fragment action trajectories for pre-training.
    """

    def __init__(self, config: MoleculeConfig):
        self.config = config
        self.vocab = config.fragment_vocabulary
        assert self.vocab is not None, "fragment_vocabulary must be loaded"
        self._build_lookups()

    def _build_lookups(self):
        """Build SMILES → fragment_id and fragment_id → entry lookups."""
        self.smiles_to_id: Dict[str, int] = {}
        self.id_to_entry: Dict[int, 'FragmentEntry'] = {}

        for frag in self.vocab:
            canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(frag.smiles))
            self.smiles_to_id[canon_smi] = frag.fragment_id
            self.id_to_entry[frag.fragment_id] = frag

    # ════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ════════════════════════════════════════════════════════════════

    def decompose(self, target_smiles: str) -> Optional[List[int]]:
        """Decompose a target SMILES into a fragment action sequence."""
        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            return None

        # 1. Find BRICS bonds
        brics_bonds_raw = list(BRICS.FindBRICSBonds(mol))
        if not brics_bonds_raw:
            return None

        # Parse: [((a, b), (env_a_str, env_b_str)), ...]
        brics_bonds = []
        for bond_id, (atoms, envs) in enumerate(brics_bonds_raw):
            a, b = atoms
            env_a, env_b = int(envs[0]), int(envs[1])
            brics_bonds.append((a, b, env_a, env_b, bond_id))

        # 2. Break bonds and extract fragments
        frag_data, frag_assignment, num_frags = self._get_fragments(
            mol, brics_bonds_raw
        )

        if num_frags < 2:
            return None

        # 3. Match fragments to vocabulary
        frag_info = self._match_all_fragments(frag_data, num_frags)
        if frag_info is None:
            return None

        # 4. Build connection graph
        connections = self._build_connections(frag_assignment, brics_bonds)

        # 5. Find spanning tree
        tree_edges, ring_edges, root = self._find_spanning_tree(
            connections, num_frags
        )

        # 6. Simulate assembly
        actions = self._simulate_assembly(
            mol, frag_assignment, frag_info, connections,
            tree_edges, ring_edges, root
        )

        return actions

    def decompose_batch(
        self,
        smiles_list: List[str],
        verbose: bool = False,
    ) -> List[Tuple[str, List[int]]]:
        """Decompose a batch of SMILES into action trajectories."""
        results = []
        n_success = 0
        n_fail = 0

        for smi in smiles_list:
            actions = self.decompose(smi)
            if actions is not None:
                results.append((smi, actions))
                n_success += 1
            else:
                n_fail += 1

        if verbose:
            total = len(smiles_list)
            print(
                f"Decomposed {n_success}/{total} molecules "
                f"({n_fail} failed)"
            )

        return results

    # ════════════════════════════════════════════════════════════════
    # STEP 1-2: BRICS DECOMPOSITION + FRAGMENT EXTRACTION
    # ════════════════════════════════════════════════════════════════

    def _get_fragments(
        self,
        mol: Chem.Mol,
        brics_bonds_raw: List,
    ) -> Tuple[List, List[int], int]:
        """
        Break BRICS bonds and extract fragments.

        Uses GetMolFrags with fragsMolAtomMapping (passing empty list
        that RDKit fills in-place).
        """
        broken_mol = BRICS.BreakBRICSBonds(mol)
        n_original = mol.GetNumAtoms()

        # FIX: GetMolFrags with fragsMolAtomMapping requires an empty list
        # that RDKit fills in-place (NOT a boolean)
        frags_mol_atom_mapping = []
        frag_mols = Chem.GetMolFrags(
            broken_mol,
            asMols=True,
            fragsMolAtomMapping=frags_mol_atom_mapping,
        )
        # frag_mols: tuple of Mol objects
        # frags_mol_atom_mapping: list of tuples, each tuple has frag atom indices
        #   mapping to broken_mol atom indices

        frag_groups = Chem.GetMolFrags(broken_mol, asMols=False)
        num_frags = len(frag_groups)

        # Map original atoms to fragments
        frag_assignment = [0] * n_original
        for frag_idx, group in enumerate(frag_groups):
            for atom_idx in group:
                if atom_idx < n_original:
                    frag_assignment[atom_idx] = frag_idx

        # Build per-fragment data
        frag_data = []
        for frag_idx in range(num_frags):
            frag_mol = frag_mols[frag_idx]
            # frags_mol_atom_mapping[frag_idx] is a tuple of broken_mol indices
            # corresponding to frag_mol's atoms (in order)
            mapping = frags_mol_atom_mapping[frag_idx]

            # Build atom_map: {broken_mol_idx: frag_atom_idx}
            atom_map = {broken_idx: frag_idx_local
                       for frag_idx_local, broken_idx in enumerate(mapping)}

            # Find dummies and their attached original atoms
            dummy_info = []
            for frag_atom_idx in range(frag_mol.GetNumAtoms()):
                atom = frag_mol.GetAtomWithIdx(frag_atom_idx)
                if atom.GetAtomicNum() == 0:  # Dummy atom
                    isotope = atom.GetIsotope()
                    # Find the real atom it's attached to
                    for neighbor in atom.GetNeighbors():
                        if neighbor.GetAtomicNum() != 0:
                            neighbor_frag_idx = neighbor.GetIdx()
                            neighbor_broken_idx = mapping[neighbor_frag_idx]
                            if neighbor_broken_idx < n_original:
                                dummy_info.append(
                                    (neighbor_broken_idx, isotope, frag_atom_idx)
                                )

            frag_data.append((frag_mol, atom_map, dummy_info))

        return frag_data, frag_assignment, num_frags

    # ════════════════════════════════════════════════════════════════
    # STEP 3: VOCABULARY MATCHING
    # ════════════════════════════════════════════════════════════════

    def _match_all_fragments(
        self,
        frag_data: List,
        num_frags: int,
    ) -> Optional[List[Dict]]:
        """Match all fragments to vocabulary and build atom/dummy mappings."""
        frag_info_list = []

        for frag_idx in range(num_frags):
            frag_mol, atom_map, dummy_info = frag_data[frag_idx]

            # Get canonical SMILES
            smi = Chem.MolToSmiles(frag_mol)

            if smi not in self.smiles_to_id:
                return None  # Fragment not in vocabulary

            frag_id = self.smiles_to_id[smi]
            frag_entry = self.id_to_entry[frag_id]

            # Match our fragment to the canonical form
            canon_mol = Chem.MolFromSmiles(smi)
            match = frag_mol.GetSubstructMatch(canon_mol)

            if not match or len(match) != canon_mol.GetNumAtoms():
                return None

            # match[canon_idx] = frag_mol_idx

            # Build dummy_mapping: for each vocab dummy, find the
            # corresponding (orig_atom, isotope)
            dummy_mapping = []
            for canon_dummy_idx in frag_entry.attachment_atom_indices:
                frag_mol_dummy_idx = match[canon_dummy_idx]
                # Find this dummy in dummy_info
                found = False
                for orig_atom, isotope, dummy_idx in dummy_info:
                    if dummy_idx == frag_mol_dummy_idx:
                        dummy_mapping.append((orig_atom, isotope))
                        found = True
                        break
                if not found:
                    return None

            # Build atom_mapping: {orig_atom_idx: canon_idx}
            # match[canon_idx] = frag_mol_idx
            # atom_map = {broken_mol_idx: frag_mol_idx}
            # So: broken_mol_idx → frag_mol_idx → canon_idx
            atom_mapping = {}
            for broken_idx, frag_mol_idx in atom_map.items():
                for canon_idx, matched_frag_idx in enumerate(match):
                    if matched_frag_idx == frag_mol_idx:
                        atom_mapping[broken_idx] = canon_idx
                        break

            frag_info_list.append({
                'frag_id': frag_id,
                'frag_entry': frag_entry,
                'dummy_mapping': dummy_mapping,
                'atom_mapping': atom_mapping,
            })

        return frag_info_list

    # ════════════════════════════════════════════════════════════════
    # STEP 4: CONNECTION GRAPH
    # ════════════════════════════════════════════════════════════════

    def _build_connections(
        self,
        frag_assignment: List[int],
        brics_bonds: List[Tuple],
    ) -> List[Tuple]:
        """Build the fragment connection graph."""
        connections = []
        for a, b, env_a, env_b, bond_id in brics_bonds:
            frag_a = frag_assignment[a]
            frag_b = frag_assignment[b]
            connections.append(
                (frag_a, frag_b, a, b, env_a, env_b, bond_id)
            )
        return connections

    # ════════════════════════════════════════════════════════════════
    # STEP 5: SPANNING TREE
    # ════════════════════════════════════════════════════════════════

    def _find_spanning_tree(
        self,
        connections: List[Tuple],
        num_frags: int,
    ) -> Tuple[List, List, int]:
        """Find a spanning tree of the fragment graph using BFS."""
        adj = defaultdict(list)
        for conn in connections:
            fa, fb = conn[0], conn[1]
            adj[fa].append(conn)
            adj[fb].append(conn)

        root = 0
        visited = set()
        tree_edges = []
        ring_edges = []
        seen_bonds = set()

        queue = deque([root])
        visited.add(root)

        while queue:
            curr = queue.popleft()
            for conn in adj[curr]:
                neighbor = conn[1] if conn[0] == curr else conn[0]
                bond_id = conn[6]
                if bond_id in seen_bonds:
                    continue
                seen_bonds.add(bond_id)
                if neighbor not in visited:
                    visited.add(neighbor)
                    tree_edges.append(conn)
                    queue.append(neighbor)
                else:
                    ring_edges.append(conn)

        return tree_edges, ring_edges, root

    # ════════════════════════════════════════════════════════════════
    # STEP 6: ASSEMBLY SIMULATION
    # ════════════════════════════════════════════════════════════════

    def _simulate_assembly(
        self,
        mol: Chem.Mol,
        frag_assignment: List[int],
        frag_info: List[Dict],
        connections: List[Tuple],
        tree_edges: List[Tuple],
        ring_edges: List[Tuple],
        root: int,
    ) -> Optional[List[int]]:
        """
        Simulate the assembly process using MoleculeDesign, recording actions.

        Key insight: When BRICS breaks a bond between atom_a (env X) and
        atom_b (env Y):
          - atom_a gets dummy with isotope Y (the OTHER side's env)
          - atom_b gets dummy with isotope X

        So when searching for sites:
          - Site on atom_a has isotope = env_b
          - Site on atom_b has isotope = env_a
        """
        assembled = set()
        assembled.add(root)

        # Create MoleculeDesign with root fragment
        root_frag_id = frag_info[root]['frag_id']
        design = MoleculeDesign(self.config, initial_fragment=root_frag_id)

        # Set _decomp_orig_idx on root fragment's atoms
        self._set_orig_idx_properties(
            design, frag_info[root], is_root=True, n_before=0
        )

        actions = []

        # ── Process tree edges (fragment additions) ────────────────
        for conn in tree_edges:
            (
                frag_a, frag_b, atom_a, atom_b,
                env_a, env_b, bond_id
            ) = conn

            # Determine parent (in scaffold) and child (to add)
            if frag_a in assembled:
                parent_frag, child_frag = frag_a, frag_b
                parent_atom, child_atom = atom_a, atom_b
                parent_env, child_env = env_a, env_b
            else:
                parent_frag, child_frag = frag_b, frag_a
                parent_atom, child_atom = atom_b, atom_a
                parent_env, child_env = env_b, env_a

            # Record atom count before adding
            n_before = design.rdkit_mol.GetNumAtoms()

            # L0: Add child fragment
            child_frag_id = frag_info[child_frag]['frag_id']
            action_l0 = 1 + child_frag_id
            design.take_action(action_l0)
            actions.append(action_l0)

            # Set _decomp_orig_idx on child fragment's atoms
            self._set_orig_idx_properties(
                design, frag_info[child_frag],
                is_root=False, n_before=n_before
            )

            # L1: Find and select child's site
            # FIX: The dummy on child_atom has isotope = parent_env
            # (because BRICS assigns the OTHER side's environment)
            child_site_idx = self._find_site_for_connection(
                design, child_atom, parent_env
            )
            if child_site_idx is None:
                return None

            design.take_action(child_site_idx)
            actions.append(child_site_idx)

            # L2: Find and select parent's site
            # FIX: The dummy on parent_atom has isotope = child_env
            parent_site_idx = self._find_site_for_connection(
                design, parent_atom, child_env
            )
            if parent_site_idx is None:
                return None

            design.take_action(parent_site_idx)
            actions.append(parent_site_idx)

            assembled.add(child_frag)

        # ── Process ring edges (ring closures) ─────────────────────
        K = len(self.vocab)

        for conn in ring_edges:
            (
                frag_a, frag_b, atom_a, atom_b,
                env_a, env_b, bond_id
            ) = conn

            # L0: Select site on atom_a
            # FIX: The dummy on atom_a has isotope = env_b
            site_a_idx = self._find_site_for_connection(
                design, atom_a, env_b
            )
            if site_a_idx is None:
                return None

            action_l0 = 1 + K + site_a_idx
            design.take_action(action_l0)
            actions.append(action_l0)

            # L1: Select site on atom_b
            # FIX: The dummy on atom_b has isotope = env_a
            site_b_idx = self._find_site_for_connection(
                design, atom_b, env_a
            )
            if site_b_idx is None:
                return None

            design.take_action(site_b_idx)
            actions.append(site_b_idx)

            # L2: Select bond order
            bond_order = brincs_bond_order(env_a, env_b)
            if bond_order == 0:
                return None

            if self._is_deterministic_bond(design, site_a_idx, site_b_idx):
                action_l2 = 0
            else:
                action_l2 = bond_order - 1

            design.take_action(action_l2)
            actions.append(action_l2)

        # ── Terminate ──────────────────────────────────────────────
        design.take_action(0)
        actions.append(0)

        # ── Verify result ──────────────────────────────────────────
        if not design.synthesis_done:
            return None

        target_smi = Chem.MolToSmiles(mol)
        try:
            result_smi = Chem.CanonSmiles(design.smiles_string)
            if Chem.CanonSmiles(target_smi) != result_smi:
                return None
        except Exception:
            return None

        return actions

    # ════════════════════════════════════════════════════════════════
    # HELPER: Set _decomp_orig_idx properties
    # ════════════════════════════════════════════════════════════════

    def _set_orig_idx_properties(
        self,
        design: MoleculeDesign,
        frag_info_dict: Dict,
        is_root: bool,
        n_before: int,
    ):
        """
        Set _decomp_orig_idx property on real atoms in the design's RWMol.

        For the ROOT fragment:
        - _init_fragment_mode copies real atoms in order (skipping dummies)
        - Real atoms get RDKit indices 0, 1, 2, ... in order of appearance
        - Dummies are appended at the end
        - So: k-th real atom in fragment → RDKit index k

        For NON-ROOT fragments (inserted via InsertMol at L0):
        - The fragment's rdkit_mol (with dummies) is inserted as-is
        - Atoms are appended in fragment rdkit_mol order
        - So: canon_idx i → RDKit index (n_before + i)
        """
        atom_mapping = frag_info_dict['atom_mapping']  # {orig_idx: canon_idx}
        frag_entry = frag_info_dict['frag_entry']
        frag_mol = frag_entry.rdkit_mol

        if is_root:
            # For root: _init_fragment_mode strips BRICS dummies and
            # re-adds them. Real atoms are at indices 0, 1, 2, ...
            # (in order of appearance in the fragment, skipping dummies).

            # Build list of real atom indices in fragment's rdkit_mol
            frag_real_indices = []
            for i in range(frag_mol.GetNumAtoms()):
                if frag_mol.GetAtomWithIdx(i).GetAtomicNum() != 0:
                    frag_real_indices.append(i)

            # Map canon_idx → position among real atoms
            canon_to_pos = {c: p for p, c in enumerate(frag_real_indices)}

            for orig_idx, canon_idx in atom_mapping.items():
                if canon_idx not in canon_to_pos:
                    continue  # Skip dummies
                real_pos = canon_to_pos[canon_idx]
                # numpy index: 0 = virtual, 1+ = real atoms
                numpy_idx = 1 + real_pos
                if numpy_idx < len(design._numpy_to_rdkit):
                    rdkit_idx = int(design._numpy_to_rdkit[numpy_idx])
                    atom = design.rdkit_mol.GetAtomWithIdx(rdkit_idx)
                    if atom.GetAtomicNum() != 0:
                        atom.SetIntProp("_decomp_orig_idx", orig_idx)
        else:
            # For non-root: fragment's rdkit_mol (with dummies) is
            # inserted via InsertMol. Atoms are appended in order.
            # canon_idx i → RDKit index (n_before + i)
            for orig_idx, canon_idx in atom_mapping.items():
                rdkit_idx = n_before + canon_idx
                if rdkit_idx < design.rdkit_mol.GetNumAtoms():
                    atom = design.rdkit_mol.GetAtomWithIdx(rdkit_idx)
                    if atom.GetAtomicNum() != 0:
                        atom.SetIntProp("_decomp_orig_idx", orig_idx)

    # ════════════════════════════════════════════════════════════════
    # HELPER: Find open site for a connection
    # ════════════════════════════════════════════════════════════════

    def _find_site_for_connection(
        self,
        design: MoleculeDesign,
        orig_atom: int,
        isotope: int,
    ) -> Optional[int]:
        """
        Find the open attachment site index for a specific connection.

        Identifies the site by:
        1. Matching the isotope (or flexible if isotope=0)
        2. Finding the dummy's attached real atom
        3. Checking if the real atom has _decomp_orig_idx == orig_atom
        """
        for site_idx, site in enumerate(design.open_attachment_sites):
            rwmol_dummy_idx = site[0]
            site_isotope = site[2]

            # Isotope must match (or be flexible)
            if site_isotope != isotope and site_isotope != 0:
                continue

            # Get the dummy atom from RWMol
            dummy_atom = design.rdkit_mol.GetAtomWithIdx(int(rwmol_dummy_idx))

            # Get the real atom the dummy is attached to
            neighbors = dummy_atom.GetNeighbors()
            if not neighbors:
                continue
            real_atom = neighbors[0]

            # Check if this is the correct original atom
            if real_atom.HasProp("_decomp_orig_idx"):
                if real_atom.GetIntProp("_decomp_orig_idx") == orig_atom:
                    return site_idx

        return None

    # ════════════════════════════════════════════════════════════════
    # HELPER: Check if bond order is deterministic
    # ════════════════════════════════════════════════════════════════

    def _is_deterministic_bond(
        self,
        design: MoleculeDesign,
        site_a_idx: int,
        site_b_idx: int,
    ) -> bool:
        """Check if the bond order between two sites is deterministic."""
        site_a = design.open_attachment_sites[site_a_idx]
        site_b = design.open_attachment_sites[site_b_idx]

        bond_type_a = site_a[1]  # 0 = flexible, 1/2/3 = fixed
        bond_type_b = site_b[1]

        return bond_type_a != 0 and bond_type_b != 0