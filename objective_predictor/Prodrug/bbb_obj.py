# objective_predictor/Prodrug/bbb_objective.py
"""
BBB permeability objective with a junction-purity (cleavability) gate.

Reward = P_BBB(G) * 1[QED(G) >= qed_floor] * purity(G, S0)

Three multiplicative factors, all in [0, 1]:

  - P_BBB:  ensemble of MiniMol heads fine-tuned on TDC bbb_martins.
  - QED gate:  hard floor on drug-likeness (1 if QED >= qed_floor, else 0).
  - Junction purity:  fraction of parent-addon junction bonds that are
                      hydrolyzable (ester / carbamate / aminoacid amide).
                      1.0 means every bond connecting parent to addon is
                      cleavable -> the parent is fully recoverable on
                      hydrolysis. 0.5 means half cleavable. 0 means no
                      cleavable junction at all (or no addon).

Design notes:
  - Purity (continuous in [0,1]) replaces the previous binary cleave gate.
    The binary form rewarded *any* cleavable junction even when the policy
    also added permanent modifications (e.g. methyl ether on a phenol while
    esterifying a carboxyl). Purity demands full cleavability with partial
    credit during training -- a clean, defensible specification.
  - No tunable weights, no soft ramps, no MW caps. Every term is justified
    outside the paper:
      * BBB by the TDC benchmark
      * QED >= 0.4 by Bickerton et al. 2012
      * Purity by the textbook definition of a prodrug (parent recoverable)
"""
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from objective_predictor.Prodrug.base_objective import BaseObjective
from minimol_oracle import MiniMolOracle


# Three canonical hydrolyzable bonds.
_HYDROLYZABLE_SMARTS = [
    "[#6,#7,#8]-[CX3](=O)-[OX2]-[#6]",       # ester (broad)
    "[#7]-[CX3](=O)-[OX2]-[#6]",             # carbamate
    "[NX3;H2,H1]-[CX4]-[CX3](=O)-[NX3]",     # aminoacid amide
]
HYDROLYZABLE_PATTERNS = [Chem.MolFromSmarts(s) for s in _HYDROLYZABLE_SMARTS]


class BBBObjective(BaseObjective):
    """Drop-in replacement for the binary-cleave-gate objective.

    Args:
        qed_floor: hard threshold on QED. Reward zeroed below this.
        cache_dir: where MiniMolOracle caches featurizations and weights.
    """

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

    def _junction_purity(self,
                         generated_mol: Chem.Mol,
                         parent_mol: Chem.Mol) -> float:
        """Fraction of parent-addon junction bonds that participate in a
        hydrolyzable motif.

        Returns:
            1.0  -> every junction bond is cleavable (pure prodrug)
            x    -> x fraction of junction bonds are cleavable, 0 < x < 1
            0.0  -> no junction bond is cleavable, OR no addon at all,
                    OR parent not preserved as subgraph
        """
        # useChirality=False because additive policies sometimes alter the
        # local stereo perception without removing atoms; we want the match
        # to succeed so we can verify atom-level preservation.
        match = generated_mol.GetSubstructMatch(parent_mol, useChirality=False)
        if not match:
            return 0.0

        parent_atoms = set(match)
        junction_bonds = []
        for bond in generated_mol.GetBonds():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a1 in parent_atoms) != (a2 in parent_atoms):
                junction_bonds.append((a1, a2))

        if not junction_bonds:
            return 0.0  # no addon -> not a prodrug

        # Collect the set of junction bonds that lie inside a hydrolyzable
        # SMARTS match. Set semantics: a bond is counted at most once even
        # if multiple patterns claim it.
        cleavable = set()
        for patt in HYDROLYZABLE_PATTERNS:
            if patt is None:
                continue
            for atom_match in generated_mol.GetSubstructMatches(patt):
                ms = set(atom_match)
                for jb in junction_bonds:
                    a1, a2 = jb
                    if a1 in ms and a2 in ms:
                        cleavable.add(jb)

        return float(len(cleavable)) / float(len(junction_bonds))

    # ------------------------------- API ----------------------------------- #

    def calculate(self,
                  generated_mol: Chem.Mol,
                  parent_mol: Optional[Chem.Mol] = None) -> dict:
        if parent_mol is None:
            raise ValueError(
                "BBBObjective.calculate requires parent_mol; "
                "the purity gate is undefined without it."
            )

        smi = Chem.MolToSmiles(generated_mol)
        bbb_prob = float(self._oracle(smi))
        qed = float(QED.qed(generated_mol))
        mw = float(Descriptors.MolWt(generated_mol))

        qed_gate = self._qed_gate(qed)
        purity = self._junction_purity(generated_mol, parent_mol)

        total = bbb_prob * qed_gate * purity

        return {
            "total_reward": total,
            "reward_bbb": bbb_prob,
            "reward_qed_gate": qed_gate,
            "reward_purity": purity,
            "metrics": {
                "bbb_prob": bbb_prob,
                "qed": qed,
                "mw": mw,
                "qed_gate": qed_gate,
                "purity": purity,
                # Convenience flag matching the binary semantics for any
                # downstream code that wants a yes/no:
                "is_pure_prodrug": purity == 1.0,
            },
        }

    def calculate_batch(self,
                        generated_mols: List[Chem.Mol],
                        parent_mols: Optional[List[Chem.Mol]] = None) -> List[dict]:
        if parent_mols is None or len(parent_mols) != len(generated_mols):
            raise ValueError(
                "calculate_batch requires parent_mols aligned 1:1 with "
                "generated_mols; the purity gate is per-pair."
            )

        smis = [Chem.MolToSmiles(m) for m in generated_mols]
        bbb_probs = self._oracle(smis)

        out = []
        for gen_mol, parent_mol, bbb_prob in zip(generated_mols, parent_mols, bbb_probs):
            qed = float(QED.qed(gen_mol))
            mw = float(Descriptors.MolWt(gen_mol))
            qg = self._qed_gate(qed)
            pur = self._junction_purity(gen_mol, parent_mol)
            p = float(bbb_prob)
            out.append({
                "total_reward": p * qg * pur,
                "reward_bbb": p,
                "reward_qed_gate": qg,
                "reward_purity": pur,
                "metrics": {
                    "bbb_prob": p,
                    "qed": qed,
                    "mw": mw,
                    "qed_gate": qg,
                    "purity": pur,
                    "is_pure_prodrug": pur == 1.0,
                },
            })
        return out