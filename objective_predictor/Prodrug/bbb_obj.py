# objective_predictor/Prodrug/bbb_objective.py
"""
BBB permeability objective using MiniMol foundation model.

Reward = BBB_prob * QED_gate * Size_gate

- BBB_prob:  ensemble of MiniMol heads fine-tuned on TDC bbb_martins
             (delegated to MiniMolOracle, which replicates the Graphcore SOTA
             training procedure)
- QED_gate:  drug-likeness guard; flat at 1.0 above qed_floor, linear ramp below
- Size_gate: prevents trivial chain elongation; flat at 1.0 below mw_soft_cap,
             linear ramp to 0 at mw_hard_cap, hard 0 above

The multiplicative form prevents reward hacking: the model cannot trade off
drug-likeness or molecule size to inflate the BBB score. Either gate hitting
zero zeros the whole reward.
"""
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors, QED

from objective_predictor.Prodrug.base_objective import BaseObjective
from minimol_oracle import MiniMolOracle


class BBBObjective(BaseObjective):
    """Drop-in replacement for the legacy multi-component prodrug objective."""

    # Class-level singleton: loading MiniMol + ensemble is expensive and
    # we don't want to retrain/reload across instantiations within a run.
    _oracle: Optional[MiniMolOracle] = None

    def __init__(self,
                 qed_floor: float = 0.6,
                 mw_soft_cap: float = 500.0,
                 mw_hard_cap: float = 600.0,
                 cache_dir: str = "./oracle_cache"):
        self.qed_floor = qed_floor
        self.mw_soft_cap = mw_soft_cap
        self.mw_hard_cap = mw_hard_cap
        if BBBObjective._oracle is None:
            BBBObjective._oracle = MiniMolOracle(task_name="bbb", cache_dir=cache_dir)

    # ------------------------------- gates --------------------------------- #

    def _qed_gate(self, qed: float) -> float:
        if qed >= self.qed_floor:
            return 1.0
        return float(qed / self.qed_floor)

    def _size_gate(self, mw: float) -> float:
        if mw <= self.mw_soft_cap:
            return 1.0
        if mw >= self.mw_hard_cap:
            return 0.0
        return float((self.mw_hard_cap - mw) / (self.mw_hard_cap - self.mw_soft_cap))

    # ------------------------------- API ----------------------------------- #

    def calculate(self, generated_mol: Chem.Mol,
                  parent_mol: Optional[Chem.Mol] = None) -> dict:
        smi = Chem.MolToSmiles(generated_mol)
        bbb_prob = float(self._oracle(smi))   # MiniMolOracle returns scalar for single SMILES

        qed = float(QED.qed(generated_mol))
        mw = float(Descriptors.MolWt(generated_mol))

        qed_gate = self._qed_gate(qed)
        size_gate = self._size_gate(mw)
        # total = bbb_prob * qed_gate * size_gate
        total = bbb_prob * qed_gate

        return {
            "total_reward": total,
            "reward_bbb": bbb_prob,
            "reward_qed_gate": qed_gate,
            "reward_size_gate": size_gate,
            "metrics": {
                "bbb_prob": bbb_prob,
                "qed": qed,
                "mw": mw,
                "qed_gate": qed_gate,
                "size_gate": size_gate,
            },
        }

    def calculate_batch(self, generated_mols: List[Chem.Mol]) -> List[dict]:
        """Batched scoring. Use this in RL rollouts — featurisation dominates
        wall-clock and is much faster amortised across a batch."""
        smis = [Chem.MolToSmiles(m) for m in generated_mols]
        bbb_probs = self._oracle(smis)  # ndarray of shape (N,)

        out = []
        for mol, p in zip(generated_mols, bbb_probs):
            qed = float(QED.qed(mol))
            mw = float(Descriptors.MolWt(mol))
            qg, sg = self._qed_gate(qed), self._size_gate(mw)
            out.append({
                "total_reward": float(p) * qg * sg,
                "reward_bbb": float(p),
                "reward_qed_gate": qg,
                "reward_size_gate": sg,
                "metrics": {
                    "bbb_prob": float(p),
                    "qed": qed,
                    "mw": mw,
                    "qed_gate": qg,
                    "size_gate": sg,
                },
            })
        return out
