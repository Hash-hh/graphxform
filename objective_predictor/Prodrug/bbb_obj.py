from objective_predictor.Prodrug.base_objective import BaseObjective
from rdkit import Chem
import rdkit.Chem.Crippen as Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Descriptors, QED


class BBBObjective(BaseObjective):
    """Objective class to evaluate Blood-Brain Barrier (BBB) permeability of prodrugs

    It calculates a reward based on these components:
    1. LogP Change: We want to increase lipophilicity i.e. make it fattier.
    2. H-Bong Change: We want to decrease hydrogen bonds i.e. remove -OH and -NH.
    3  Add Ester: We want to add a cleavable ester group.
    4. MW Penalty: Penalize if Molecular Weight exceeds a threshold (prevent infinite chains).
    5. QED Score: Reward for drug-likeness.
    """

    def __init__(self,
                 weight_logp_delta: float = 1.0,
                 weight_hdonor_delta: float = 1.0,
                 weight_cleavable: float = 1.0,
                 weight_mw_penalty: float = 5.0,  # Heavy penalty
                 max_mw: float = 600.0,  # Max MW threshold
                 weight_qed: float = 2.0  # Reward for drug-likeness
                 ):
                 self.weight_logp_delta = weight_logp_delta
                 self.weight_hdonor_delta = weight_hdonor_delta
                 self.weight_cleavable = weight_cleavable
                 self.weight_mw_penalty = weight_mw_penalty
                 self.max_mw = max_mw
                 self.weight_qed = weight_qed

                 self.ester_smarts = Chem.MolFromSmarts('[#6]C(=O)O[#6]')  # [Any Carbon]-C(=O)-O-[Any Carbon]
                 self.amide_smarts = Chem.MolFromSmarts('[#6]C(=O)N[#6]')  # [Any Carbon]-C(=O)-N-[Any Carbon]

    def _calculate_property_delta(self, mol_gen, mol_parent):
        logp_parent = Crippen.MolLogP(mol_parent)
        logp_gen = Crippen.MolLogP(mol_gen)
        logp_delta = logp_gen - logp_parent  # change in logP -> +iv is better

        hdonor_parent = rdMolDescriptors.CalcNumHBD(mol_parent)
        hdonor_gen = rdMolDescriptors.CalcNumHBD(mol_gen)
        hdonor_delta = hdonor_parent - hdonor_gen  # change in TPSA -> -iv is better

        return {
            'logp_parent': logp_parent,
            'logp_gen': logp_gen,
            'logp_delta': logp_delta,
            'hdonor_parent': hdonor_parent,
            'hdonor_gen': hdonor_gen,
            'hdonor_delta': hdonor_delta
        }

    def _calculate_cleavable_reward(self, mol_gen, mol_parent) -> float:
        """Check if a new ester OR amide bond was added."""
        parent_ester_count = len(mol_parent.GetSubstructMatches(self.ester_smarts))
        gen_ester_count = len(mol_gen.GetSubstructMatches(self.ester_smarts))

        parent_amide_count = len(mol_parent.GetSubstructMatches(self.amide_smarts))
        gen_amide_count = len(mol_gen.GetSubstructMatches(self.amide_smarts))

        if (gen_ester_count > parent_ester_count) or (gen_amide_count > parent_amide_count):
            return 1.0  # Reward for adding ester
        else:
            return 0.0

    def _calculate_physchem(self, mol_gen):
        """Calculate MW and QED."""
        mw = Descriptors.MolWt(mol_gen)
        qed = QED.qed(mol_gen)
        return mw, qed

    def calculate(self, generated_mol: Chem.Mol, parent_mol: Chem.Mol) -> dict:
        prop_deltas = self._calculate_property_delta(generated_mol, parent_mol)
        mw, qed = self._calculate_physchem(generated_mol)

        reward_logp = prop_deltas['logp_delta'] * self.weight_logp_delta
        reward_hdonor = prop_deltas['hdonor_delta'] * self.weight_hdonor_delta

        reward_cleavable = (self._calculate_cleavable_reward(generated_mol, parent_mol)
                            * self.weight_cleavable)

        reward_qed = qed * self.weight_qed

        # Penalty: If MW > max, subtract penalty.
        # TODO: We could also make it proportional to excess, but step is fine for now
        penalty_mw = -self.weight_mw_penalty if mw > self.max_mw else 0.0

        total_score = reward_logp + reward_hdonor + reward_cleavable + reward_qed + penalty_mw

        return {
            'total_reward': total_score,
            'reward_logp_weighted': reward_logp,
            'reward_hdonor_weighted': reward_hdonor,
            'reward_cleavable_weighted': reward_cleavable,
            'metrics': {
                **prop_deltas,
                'cleavable_bond_added': bool(reward_cleavable > 0),
                'mw': mw,
                'qed': qed,
                'penalty_mw': penalty_mw
            }
        }

