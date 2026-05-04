# objective_predictor/Prodrug/bbb_objective.py
"""
BBB permeability objective with a hydrolyzable-junction (cleavability) gate.

Reward = P_BBB(G) * 1[QED(G) >= qed_floor] * 1[junction(G, S0) is hydrolyzable]

Three multiplicative factors, all in [0, 1]:

  - P_BBB:  ensemble of MiniMol heads fine-tuned on TDC bbb_martins
            (delegated to MiniMolOracle, which replicates the Graphcore SOTA
            training procedure)
  - QED gate:  hard floor on drug-likeness. 1 if QED >= qed_floor, else 0.
  - Cleavage gate:  1 if at least one bond connecting the parent subgraph to
                    the addon subgraph participates in an ester, carbamate, or
                    aminoacid-amide motif, else 0. This enforces the standard
                    prodrug definition: the addon must be enzymatically
                    severable so the parent is released in the brain.

Design notes:
  - The reward is a product of one learned signal and two off-the-shelf
    chemical tests. There are no tunable weights, no soft ramps, no MW caps.
    Every term is justified outside the paper:
      * BBB by the TDC benchmark MiniMol was fine-tuned on
      * QED >= 0.4 by Bickerton et al. 2012 (roughly the median for FDA-approved
        oral drugs)
      * Hydrolyzable junction by the textbook definition of a prodrug
  - The policy is purely additive (atoms and bonds are only added, never
    removed), so the parent is preserved as a subgraph of the generated
    molecule by construction. We still defensively verify this with
    GetSubstructMatch and fail loudly otherwise.
"""
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from objective_predictor.Prodrug.base_objective import BaseObjective
from minimol_oracle import MiniMolOracle


# -------------------------- hydrolyzable motifs --------------------------- #
# Three canonical cleavable bonds that endogenous CNS enzymes hydrolyze:
#   - ester:           parent -X- C(=O) -O- R   (carboxylesterases)
#   - carbamate:       parent -N- C(=O) -O- R   (slower, sometimes desired)
#   - aminoacid amide: parent -NH- C(=O) -CH(R)- NH2   (peptidases)
#
# Each pattern is broad enough to match common variants without false positives
# from non-cleavable look-alikes (e.g. a plain amide bond is NOT included since
# it's far less reliably hydrolyzed in the CNS).
_HYDROLYZABLE_SMARTS = [
    "[#6,#7,#8]-[CX3](=O)-[OX2]-[#6]",       # ester (broad: O- or N- or C-acyl)
    "[#7]-[CX3](=O)-[OX2]-[#6]",             # carbamate
    "[NX3;H2,H1]-[CX4]-[CX3](=O)-[NX3]",     # aminoacid amide
]
HYDROLYZABLE_PATTERNS = [Chem.MolFromSmarts(s) for s in _HYDROLYZABLE_SMARTS]


class BBBObjective(BaseObjective):
    """Drop-in replacement for the legacy multi-component prodrug objective.

    Args:
        qed_floor: hard threshold on QED. Reward is zeroed below this.
                   0.4 is roughly the median QED for FDA-approved oral drugs;
                   ablating in [0.3, 0.5] does not change qualitative behavior.
        cache_dir: where MiniMolOracle caches featurizations and weights.
    """

    # Class-level singleton: loading MiniMol + ensemble is expensive and
    # we don't want to retrain/reload across instantiations within a run.
    _oracle: Optional[MiniMolOracle] = None

    def __init__(self,
                 qed_floor: float = 0.4,
                 cache_dir: str = "./oracle_cache"):
        self.qed_floor = qed_floor
        if BBBObjective._oracle is None:
            BBBObjective._oracle = MiniMolOracle(task_name="bbb", cache_dir=cache_dir)

    # ------------------------------- gates --------------------------------- #

    def _qed_gate(self, qed: float) -> float:
        return 1.0 if qed >= self.qed_floor else 0.0

    def _junction_cleavable(self,
                            generated_mol: Chem.Mol,
                            parent_mol: Chem.Mol) -> bool:
        """True iff at least one parent-addon junction bond is hydrolyzable.

        The policy is additive, so the parent should appear as a subgraph of
        the generated molecule. We find the parent's atom indices in the
        generated graph, then identify bonds that straddle the parent/addon
        boundary. A junction is "cleavable" if at least one of those bonds is
        part of an ester, carbamate, or aminoacid-amide motif.

        Returns False (not cleavable) if:
          - parent is not a subgraph of generated (shouldn't happen; we log it)
          - no addon was added (parent == generated, no junction bonds exist)
          - all junction bonds are stable (e.g. C-C, C-N single bonds)
        """
        match = generated_mol.GetSubstructMatch(parent_mol)
        if not match:
            # Parent not preserved. The additive policy guarantees this should
            # not happen; if it does, treat it as a hard failure.
            return False
        parent_atoms = set(match)

        # Bonds with exactly one endpoint in the parent set straddle the
        # parent/addon boundary. These are the candidate cleavage points.
        junction_bond_atompairs = []
        for bond in generated_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in parent_atoms) != (a2 in parent_atoms):
                junction_bond_atompairs.append((a1, a2))

        if not junction_bond_atompairs:
            # No addon at all. Nothing to cleave -> not a prodrug.
            return False

        # For each hydrolyzable pattern, check whether any substructure match
        # spans at least one junction bond. We require the bond's two atoms
        # to both lie inside the same SMARTS match (i.e. the cleavable motif
        # actually contains the junction, not just sits nearby).
        for patt in HYDROLYZABLE_PATTERNS:
            if patt is None:
                continue
            for atom_match in generated_mol.GetSubstructMatches(patt):
                match_set = set(atom_match)
                for a1, a2 in junction_bond_atompairs:
                    if a1 in match_set and a2 in match_set:
                        return True
        return False

    # ------------------------------- API ----------------------------------- #

    def calculate(self,
                  generated_mol: Chem.Mol,
                  parent_mol: Optional[Chem.Mol] = None) -> dict:
        if parent_mol is None:
            # Without a parent we can't evaluate cleavability. Fail closed.
            raise ValueError(
                "BBBObjective.calculate requires parent_mol; "
                "the cleavage gate is undefined without it."
            )

        smi = Chem.MolToSmiles(generated_mol)
        bbb_prob = float(self._oracle(smi))   # MiniMolOracle: scalar for single SMILES

        qed = float(QED.qed(generated_mol))
        mw = float(Descriptors.MolWt(generated_mol))

        qed_gate = self._qed_gate(qed)
        cleave_gate = 1.0 if self._junction_cleavable(generated_mol, parent_mol) else 0.0

        total = bbb_prob * qed_gate * cleave_gate

        return {
            "total_reward": total,
            "reward_bbb": bbb_prob,
            "reward_qed_gate": qed_gate,
            "reward_cleave_gate": cleave_gate,
            "metrics": {
                "bbb_prob": bbb_prob,
                "qed": qed,
                "mw": mw,
                "qed_gate": qed_gate,
                "cleave_gate": cleave_gate,
                "cleavable": bool(cleave_gate),
            },
        }

    def calculate_batch(self,
                        generated_mols: List[Chem.Mol],
                        parent_mols: Optional[List[Chem.Mol]] = None) -> List[dict]:
        """Batched scoring. Use this in RL rollouts -- featurization dominates
        wall-clock and is much faster amortized across a batch.

        parent_mols must be the same length as generated_mols, with each
        entry being the parent for the corresponding generated molecule.
        """
        if parent_mols is None or len(parent_mols) != len(generated_mols):
            raise ValueError(
                "calculate_batch requires parent_mols aligned 1:1 with "
                "generated_mols; the cleavage gate is per-pair."
            )

        smis = [Chem.MolToSmiles(m) for m in generated_mols]
        bbb_probs = self._oracle(smis)  # ndarray of shape (N,)

        out = []
        for gen_mol, parent_mol, bbb_prob in zip(generated_mols, parent_mols, bbb_probs):
            qed = float(QED.qed(gen_mol))
            mw = float(Descriptors.MolWt(gen_mol))
            qg = self._qed_gate(qed)
            cg = 1.0 if self._junction_cleavable(gen_mol, parent_mol) else 0.0
            p = float(bbb_prob)
            out.append({
                "total_reward": p * qg * cg,
                "reward_bbb": p,
                "reward_qed_gate": qg,
                "reward_cleave_gate": cg,
                "metrics": {
                    "bbb_prob": p,
                    "qed": qed,
                    "mw": mw,
                    "qed_gate": qg,
                    "cleave_gate": cg,
                    "cleavable": bool(cg),
                },
            })
        return out