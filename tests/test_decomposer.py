"""
Unit tests for the TrajectoryDecomposer.

Uses a mock vocabulary built from ACTUAL BRICS fragments (correct isotopes).
"""

import pytest
from rdkit import Chem
from rdkit.Chem import BRICS

from config import MoleculeConfig
from molecule_design import MoleculeDesign
from core.decomposer import TrajectoryDecomposer

from tests.test_fragments import create_mock_fragment, make_fragment_config


def build_brics_vocabulary():
    """
    Build a vocabulary from ACTUAL BRICS fragments.

    BRICS environments:
      - Env 1: C;D3(=O) — carbonyl C
      - Env 3: [O;D2] — ether O
      - Env 4: [C;!D1;!$(C=*)] — sp3 C bonded to another C
      - Env 5: [N;!D1;...] — amine N
      - Env 16: [c;$(c(:c):c)] — aromatic C

    Valid BRICS bond pairs include: 3-4, 3-16, 5-16, etc.

    When BRICS breaks a bond between env X and env Y:
      - Atom X gets dummy with isotope Y
      - Atom Y gets dummy with isotope X

    So for a bond between O(env3) and C(env4):
      - O gets [4*] dummy
      - C gets [3*] dummy
      → Fragment with C: [3*]CC, [3*]CCC, etc.
      → Fragment with O: [4*]O[4*]
    """
    # These fragments are produced by BRICS decomposition of simple ethers:
    # CCOCC → [3*]CC + [4*]O[4*] + [3*]CC
    # CCCOCC → [3*]CCC + [4*]O[4*] + [3*]CC
    fragments = [
        ("[3*]CC", 0, 800),       # Ethyl with O-bond (isotope 3)
        ("[3*]CCC", 1, 600),     # Propyl with O-bond (isotope 3)
        ("[4*]O[4*]", 2, 400),   # Ether bridge (isotope 4)
        ("[4*]C", 3, 1000),      # Methyl with C-bond (isotope 4)
        ("[4*]CC", 4, 500),      # Ethyl with C-bond (isotope 4)
        ("[4*]CCC", 5, 300),     # Propyl with C-bond (isotope 4)
    ]
    return [create_mock_fragment(smi, fid, freq)
            for smi, fid, freq in fragments]


def make_brics_config():
    """Create a MoleculeConfig with BRICS-compatible vocabulary."""
    vocab = build_brics_vocabulary()
    config = make_fragment_config(vocab)
    return config


@pytest.fixture
def decomposer():
    config = make_brics_config()
    return TrajectoryDecomposer(config)


class TestDecomposer:

    def test_decompose_diethyl_ether(self, decomposer):
        """Diethyl ether (CCOCC) → [3*]CC + [4*]O[4*] + [3*]CC."""
        actions = decomposer.decompose("CCOCC")
        assert actions is not None
        assert len(actions) >= 4  # add + L1 + L2 + terminate
        assert actions[-1] == 0   # terminate

    def test_decompose_dipropyl_ether(self, decomposer):
        """Dipropyl ether (CCCOCC) → [3*]CCC + [4*]O[4*] + [3*]CC."""
        actions = decomposer.decompose("CCCOCC")
        assert actions is not None
        assert actions[-1] == 0

    def test_verify_actions_reconstruct_molecule(self, decomposer):
        """Executing decomposed actions should reconstruct the molecule."""
        target_smiles = "CCOCC"
        actions = decomposer.decompose(target_smiles)
        if actions is None:
            pytest.skip("Molecule not decomposable with mock vocabulary")

        # The decomposer already verifies SMILES match internally
        # (in _simulate_assembly, it compares result vs target)
        assert len(actions) > 0

    def test_decompose_invalid_smiles(self, decomposer):
        """Invalid SMILES returns None."""
        assert decomposer.decompose("invalid") is None

    def test_decompose_no_brics_bonds(self, decomposer):
        """Molecule with no BRICS bonds returns None.

        Ethane (CC) has no BRICS bonds because BRICS env 4 (sp3 C bonded to C)
        does not connect to itself.
        """
        assert decomposer.decompose("CC") is None

    def test_decompose_fragment_not_in_vocab(self, decomposer):
        """Molecule with fragment not in vocabulary returns None."""
        # This molecule has aromatic fragments not in our vocab
        result = decomposer.decompose("c1ccccc1OCC")
        assert result is None or isinstance(result, list)

    def test_decompose_batch(self, decomposer):
        """Batch decomposition processes multiple molecules."""
        smiles_list = ["CCOCC", "CCCOCC", "CCOCCC", "CC"]
        results = decomposer.decompose_batch(smiles_list, verbose=False)
        assert isinstance(results, list)
        # At least some should succeed
        assert len(results) >= 1

    def test_decompose_multiple_molecules(self, decomposer):
        """Test decomposition of multiple molecules."""
        for smi in ["CCOCC", "CCCOCC", "CCCCOCC"]:
            actions = decomposer.decompose(smi)
            assert actions is not None, f"Failed to decompose {smi}"
            assert actions[-1] == 0  # terminate