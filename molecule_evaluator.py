import math
from typing import List, Union
import os
import ray
import torch
import numpy as np
import pandas as pd
from config import MoleculeConfig
from rdkit import Chem, RDLogger
from rdkit.Contrib.SA_Score import sascorer

from molecule_design import MoleculeDesign
from objective_predictor.GH_GNN_IDAC.src.models.utilities.mol2graph import get_dataloader_pairs_T, sys2graph, atom_features, n_atom_features, n_bond_features
from objective_predictor.GH_GNN_IDAC.src.models.GHGNN_architecture import GHGNN
from objective_predictor.Prodrug.bbb_obj import BBBObjective
from objective_predictor.tdc.jnk import JNK3Objective
from objective_predictor.tdc.gsk import GSKObjective
from objective_predictor.tdc.qed import QEDObjective
from objective_predictor.tdc.sa import SAObjective
from objective_predictor.tdc.kinase_mpo import KinaseMPOObjective
from objective_predictor.tdc.guacamol_hard import GuacaMolHardObjective

from guacamol.benchmark_suites import goal_directed_suite_v2

@ray.remote
class PredictorWorker:
    def __init__(self, config: MoleculeConfig, device: torch.device):
        # Silence RDKit warnings
        RDLogger.DisableLog('rdApp.*')

        if config.CUDA_VISIBLE_DEVICES:
            # override ray's limiting of GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = config.CUDA_VISIBLE_DEVICES

        self.device = device
        self.config = config
        self.model = self._load_model() if hasattr(config, "GHGNN_model_path") and config.GHGNN_model_path else None
        # Pre-calculate molecules from SMILES:
        self.pre_molecules = {
            "COC1=CC(=CC(=C1)C=O)OC": Chem.MolFromSmiles("COC1=CC(=CC(=C1)C=O)OC"),
            "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC": Chem.MolFromSmiles(
                "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"),
            "O": Chem.MolFromSmiles("O"),
            "CC(C)CO": Chem.MolFromSmiles("CC(C)CO")
        }

    def predict_objectives_from_rdkit_mols(self, feasible_molecules: List[Chem.RWMol]):
        constraint_value = self.predict_constraint(feasible_molecules)  # must be exp(.) > 4
        if self.config.objective_type == "DMBA_TMB":
            ln_y_DMBA_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                               l_smiles_solute=["COC1=CC(=CC(=C1)C=O)OC"] * len(feasible_molecules))
            ln_y_TMB_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                              l_smiles_solute=[
                                                                  "COC=1C=C(C=C(C1)OC)C(=O)[C@H](O)C1=CC(=CC(=C1)OC)OC"] * len(
                                                  feasible_molecules))
            with np.errstate(divide='ignore'):
                objs = np.where(
                    constraint_value > np.exp(4),
                    np.exp(ln_y_DMBA_solv) / np.exp(ln_y_TMB_solv),
                    -np.inf
                )
        elif self.config.objective_type == "IBA":
            ln_y_IPA_solv = self.predict_IDAC(l_solvent=feasible_molecules,
                                              l_smiles_solute=["CC(C)CO"] * len(feasible_molecules))
            with np.errstate(divide='ignore'):
                objs = np.where(
                    constraint_value > np.exp(4),
                    1. / np.exp(ln_y_IPA_solv),
                    -np.inf
                )
        else:
            raise ValueError("Objective type unknown")

        if self.config.synthetic_accessibility_in_objective_scale > 0:
            sa_scores = np.array([self.calc_SA_score(x) for x in feasible_molecules])
            objs = objs - self.config.synthetic_accessibility_in_objective_scale * sa_scores

        return objs

    def calc_SA_score(self, mol: Chem.RWMol):
        """
        SA score: Synthetic accessibility of drug-like molecules (or ease of synthesis) http://www.jcheminf.com/content/1/1/8
        Gives a score of of 1 (easiest) to 10 (hardest)
        Breaks down molecule into fragments and calculates score based on "ease of synthesis" divded by number of fragments
            Therefore small molecules like methane (C) and water (O) have high scores.
        Because this value was developed for drug-like (water-soluble) molecules, and we are designing water-insoluble
            molecules, this may not be useful in the end but worth trying
        """
        return sascorer.calculateScore(mol)

    def predict_constraint(self, l_mols: List[Chem.RWMol]) -> np.array:
        ln_y_water_solv = self.predict_IDAC(l_solvent=l_mols, l_smiles_solute=["O"] * len(l_mols))
        ln_y_solv_water = self.predict_IDAC(l_solvent=[self.pre_molecules["O"]] * len(l_mols), l_smiles_solute=l_mols)

        constr_value = np.exp(ln_y_water_solv) * np.exp(ln_y_solv_water)
        return constr_value

    def predict_IDAC(self, l_solvent: List[Chem.RWMol], l_smiles_solute: List[Union[str, Chem.RWMol]], l_T=None):
        # Preprocess data
        mol_solvents = l_solvent
        mol_solutes = []
        for solute in l_smiles_solute:
            if isinstance(solute, str):
                mol_solutes.append(self.pre_molecules[solute] if solute in self.pre_molecules else Chem.MolFromSmiles(solute))
            else:
                # is Chem.RWMol
                mol_solutes.append(solute)
        Temps = l_T if l_T is not None else [25] * len(mol_solvents)
        ys = [np.nan] * len(mol_solvents)

        ## Create dummy df to be able to use original data processing functions
        mol_column_solvent = 'Molecule_Solvent'
        mol_column_solute = 'Molecule_Solute'
        target = 'log-gamma'
        df = pd.DataFrame(
            {
                mol_column_solvent: mol_solvents,
                mol_column_solute: mol_solutes,
                "T": Temps,
                "log-gamma": ys
            }
        )

        graphs_solv, graphs_solu = 'g_solv', 'g_solu'
        df[graphs_solv], df[graphs_solu] = sys2graph(
            df=df,
            mol_column_1=mol_column_solvent,
            mol_column_2=mol_column_solute,
            target=target,
            y_scaler=None,
            single_system=False,
            silent=True
        )
        ## Dataloader
        indices = df.index.tolist()
        with torch.no_grad():
            predict_loader = get_dataloader_pairs_T(df,
                                                    indices,
                                                    graphs_solv,
                                                    graphs_solu,
                                                    batch_size=self.config.objective_predictor_batch_size,
                                                    shuffle=False,
                                                    drop_last=False)

            # Batch-wise prediction
            y_pred_final = np.array([])
            for batch_solvent, batch_solute, batch_T in predict_loader:
                batch_solvent = batch_solvent.to(self.device)
                batch_solute = batch_solute.to(self.device)
                batch_T = batch_T.to(self.device)
                with torch.no_grad():
                    y_pred = self.model(
                        batch_solvent.to(self.device), batch_solute.to(self.device), batch_T.to(self.device),
                        scaler=None, ln_gamma=True).reshape(
                        -1, ).cpu().numpy()
                    y_pred_final = np.concatenate((y_pred_final, y_pred))

        return y_pred_final

    def _load_model(self):
        v_in = n_atom_features()
        e_in = n_bond_features()
        u_in = 3  # ap, bp, topopsa
        model = GHGNN(v_in, e_in, u_in, self.config.GHGNN_hidden_dim, device=self.device)
        model.load_state_dict(torch.load(self.config.GHGNN_model_path, map_location="cpu"))
        model = model.to(self.device)
        model.eval()
        return model

@ray.remote
class OracleTracker:
    def __init__(self):
        self.seen_smiles = set()

    def register_and_count(self, smiles_list: List[str]) -> int:
        """
        Registers new SMILES and returns the total count of unique SMILES seen so far.
        """
        for s in smiles_list:
            self.seen_smiles.add(s)
        return len(self.seen_smiles)

    def get_count(self) -> int:
        return len(self.seen_smiles)

class MoleculeObjectiveEvaluator:
    def __init__(self, config: MoleculeConfig, device: torch.device = None, oracle_tracker=None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.oracle_tracker = oracle_tracker
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)]

        if getattr(self.config, 'objective_type', '') == 'prodrug_bbb':
            # Use weights from config, defaulting to 1.0 if not set
            self.bbb_objective = BBBObjective(
                weight_logp_delta=getattr(self.config, 'bbb_weight_logp', 1.0),
                weight_hdonor_delta=getattr(self.config, 'bbb_weight_hdonor', 1.0),
                weight_cleavable=getattr(self.config, 'bbb_weight_cleavable', 1.0),
                weight_mw_penalty=getattr(self.config, 'bbb_weight_mw_penalty', 5.0),
                max_mw=getattr(self.config, 'bbb_max_mw', 600.0),
                weight_qed=getattr(self.config, 'bbb_weight_qed', 2.0)
            )

        # TDC objectives
        if getattr(self.config, 'objective_type', '') == 'jnk3':
            self.jnk3_objective = JNK3Objective()

        if getattr(self.config, 'objective_type', '') == 'gsk':
            self.gsk_objective = GSKObjective()

        if getattr(self.config, 'objective_type', '') == 'qed':
            self.qed_objective = QEDObjective()

        if getattr(self.config, 'objective_type', '') == 'sa':
            self.sa_objective = SAObjective()

        if getattr(self.config, 'objective_type', '') == 'kinase_mpo':
            self.kinase_mpo_objective = KinaseMPOObjective()

        if 'guacamol' in getattr(self.config, 'objective_type', ''):
            task_name = self.config.objective_type.replace('guacamol_', '')
            self.guacamol_objective = GuacaMolHardObjective(task_name=task_name)

        # initialize GuacaMol benchmarks
        guacamol_goal_directed_suite = goal_directed_suite_v2()
        self.guacamol_benchmarks = dict(
            celecoxib_rediscovery=guacamol_goal_directed_suite[0],
            troglitazone_rediscovery=guacamol_goal_directed_suite[1],
            thiothixene_rediscovery=guacamol_goal_directed_suite[2],
            aripiprazole_similarity=guacamol_goal_directed_suite[3],
            albuterol_similarity=guacamol_goal_directed_suite[4],
            mestranol_similarity=guacamol_goal_directed_suite[5],
            isomers_c11h24=guacamol_goal_directed_suite[6],
            isomers_c9h10n2o2pf2cl=guacamol_goal_directed_suite[7],
            median_camphor_menthol=guacamol_goal_directed_suite[8],
            median_tadalafil_sildenafil=guacamol_goal_directed_suite[9],
            osimertinib_mpo=guacamol_goal_directed_suite[10],
            fexofenadine_mpo=guacamol_goal_directed_suite[11],
            ranolazine_mpo=guacamol_goal_directed_suite[12],
            perindopril_rings=guacamol_goal_directed_suite[13],
            amlodipine_rings=guacamol_goal_directed_suite[14],
            sitagliptin_replacement=guacamol_goal_directed_suite[15],
            zaleplon_mpo=guacamol_goal_directed_suite[16],
            valsartan_smarts=guacamol_goal_directed_suite[17],
            deco_hop=guacamol_goal_directed_suite[18],
            scaffold_hop=guacamol_goal_directed_suite[19]
        )

    def predict_objective(self, molecule_designs: List[Union[MoleculeDesign, str]]) -> np.array:
        """
        Takes list of molecules (either as `MoleculeDesign` or directly as SMILES string
        and predicts the objective function on them. Returns the objectives as a numpy array, but also sets the
        objective directly on the objects.
        """
        # Get molecules that are known to be feasible for the predictor / RDKit / by the constraints,
        # i.e., molecules that could be sanitized and are not single carbon atoms.
        feasible_molecules: List[Chem.RWMol] = []
        feasible_idcs = []  # indices of feasible molecules in the original `molecule_designs` list
        feasible_smiles = []

        for i, mol in enumerate(molecule_designs):
            if isinstance(mol, MoleculeDesign):
                # print("Mol is a MoleculeDesign")
                assert mol.synthesis_done
                if not self.infeasible_by_special_constraints(mol):
                    feasible_idcs.append(i)
                    feasible_molecules.append(mol.rdkit_mol)
                    feasible_smiles.append(mol.smiles_string)
            elif mol != "C":
                print("Mol is a SMILES string")
                # is a string
                try:
                    mol = Chem.MolFromSmiles(mol)
                    Chem.SanitizeMol(mol)
                    feasible_idcs.append(i)
                    feasible_molecules.append(mol)
                    feasible_smiles.append(Chem.MolToSmiles(mol))
                except:
                    continue

        # Oracle Counting Logic
        # We perform this right here, after feasibility checks but before any prediction model runs.
        if self.oracle_tracker is not None and len(feasible_molecules) > 0:
            # Fire and forget (or wait if you want strict synchronization, but not strictly necessary for logging)
            # We use .remote() to send the update to the global actor.
            self.oracle_tracker.register_and_count.remote(feasible_smiles)

        if getattr(self.config, 'objective_type', '') == 'prodrug_bbb':
            objs = []
            # We iterate over the indices of feasible molecules
            for i, idx in enumerate(feasible_idcs):
                mol_obj = molecule_designs[idx]
                gen_mol = feasible_molecules[i]  # The RDKit mol from the filtering list

                # 1. Retrieve Parent SMILES
                parent_smiles = None
                if isinstance(mol_obj, MoleculeDesign):
                    # The prompt_smiles was stored during generation (see GumbeldoreDataset logic)
                    parent_smiles = getattr(mol_obj, 'prompt_smiles', None)

                # 2. Calculate Score
                if parent_smiles:
                    parent_mol = Chem.MolFromSmiles(parent_smiles)
                    # BBBObjective.calculate returns a dict with 'total_reward' and 'metrics'
                    results = self.bbb_objective.calculate(gen_mol, parent_mol)
                    score = results['total_reward']

                    # 3. Attach detailed info for logging
                    if isinstance(mol_obj, MoleculeDesign):
                        # We create a new attribute 'aux_metrics' to hold this info
                        mol_obj.aux_metrics = results['metrics']
                        # Add the weighted rewards too if you want to see them
                        mol_obj.aux_metrics.update({
                            'reward_logp': results['reward_logp_weighted'],
                            'reward_hdonor': results['reward_hdonor_weighted'],
                            'reward_cleavable': results['reward_cleavable_weighted']
                        })
                else:
                    # Fallback/Penalty if no parent is found (e.g. pure random gen)
                    score = -10.0

                objs.append(score)
            objs = np.array(objs)

        elif self.config.objective_type in self.guacamol_benchmarks:
            # Drug design tasks
            objs = np.array([
                self.guacamol_benchmarks[self.config.objective_type].objective.score(
                    Chem.MolToSmiles(rdkit_mol)
                )
                for rdkit_mol in feasible_molecules
            ])

        elif getattr(self.config, 'objective_type', '') == 'jnk3':
            objs = np.array([
                self.jnk3_objective.score(Chem.MolToSmiles(rdkit_mol))
                for rdkit_mol in feasible_molecules
            ])

        elif getattr(self.config, 'objective_type', '') == 'gsk':
            objs = np.array([
                self.gsk_objective.score(Chem.MolToSmiles(rdkit_mol))
                for rdkit_mol in feasible_molecules
            ])

        elif getattr(self.config, 'objective_type', '') == 'kinase_mpo':
            objs = np.array([
                self.kinase_mpo_objective.score(Chem.MolToSmiles(rdkit_mol))
                for rdkit_mol in feasible_molecules
            ])

        elif 'guacamol' in getattr(self.config, 'objective_type', ''):
            objs = np.array([
                self.guacamol_objective.score(Chem.MolToSmiles(rdkit_mol))
                for rdkit_mol in feasible_molecules
            ])

        else:
            # Distribute the list of feasible molecules to the predictor workers.
            num_per_worker = math.ceil(len(feasible_molecules) / len(self.predictor_workers))
            future_objs = [
                worker.predict_objectives_from_rdkit_mols.remote(feasible_molecules[i * num_per_worker: (i+1) * num_per_worker])
                for i, worker in enumerate(self.predictor_workers)
            ]
            future_objs = ray.get(future_objs)
            objs = np.concatenate(future_objs)
        all_objs = np.array([-np.inf] * len(molecule_designs))
        all_objs[feasible_idcs] = objs

        return all_objs

    def infeasible_by_special_constraints(self, mol: MoleculeDesign) -> bool:
        """
        We check special constraints such as number of rings, nitrogen-to-nitrogen bond order, etc. and if the
        constraints are not satisfied, return true.
        """
        if mol.infeasibility_flag:
            return True

        try:
            atoms = mol.rdkit_mol.GetAtoms()
            node_f = [atom_features(atom) for atom in atoms]
        except:
            return True

        if self.config.objective_type in ["IBA", "DMBA_TMB"] and self.config.include_structural_constraints:
            """
            Check for a ring with more than 6 atoms or less than 5
            """
            for ring in mol.rdkit_mol.GetRingInfo().AtomRings():
                if len(ring) < 5 or len(ring) > 6: # adjust according to max/min ring size
                    return True
            """
            Check for a O-O single bond in the molecule
            """
            for bond in mol.rdkit_mol.GetBonds():
                if (bond.GetBondType() == Chem.BondType.SINGLE and
                    mol.rdkit_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 8 and
                        mol.rdkit_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 8):
                    return True
            """
            Check for an N-N single bond
            """
            for bond in mol.rdkit_mol.GetBonds():
                if (bond.GetBondType() == Chem.BondType.SINGLE and
                    mol.rdkit_mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomicNum() == 7 and
                        mol.rdkit_mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomicNum() == 7):
                    return True
            """
            Check for an N-C-N bond (with exception for C=0)
            """
            for atom in mol.rdkit_mol.GetAtoms():
                if atom.GetAtomicNum() == 6:
                    neighbors = atom.GetNeighbors()
                    nitrogen_count = sum(1 for nbr in neighbors if
                                         nbr.GetAtomicNum() == 7 and
                                         mol.rdkit_mol.GetBondBetweenAtoms(atom.GetIdx(),
                                                                 nbr.GetIdx()).GetBondType() == Chem.BondType.SINGLE)

                    # Check if carbon is also double-bonded to oxygen (C=O)
                    has_carbonyl = any(
                        nbr.GetAtomicNum() == 8 and  # Oxygen
                        mol.rdkit_mol.GetBondBetweenAtoms(atom.GetIdx(), nbr.GetIdx()).GetBondType() == Chem.BondType.DOUBLE
                        for nbr in neighbors
                    )

                    if nitrogen_count >= 2 and not has_carbonyl:
                        return True
            """
            Don't allow O-C(X)-N
            """
            for atom in mol.rdkit_mol.GetAtoms():
                if atom.GetAtomicNum() == 6:  # Carbon atom
                    neighbors = atom.GetNeighbors()

                    # Count the types of bonded atoms
                    n_count = sum(1 for nbr in neighbors if nbr.GetAtomicNum() == 7)  # Nitrogen
                    o_count = sum(1 for nbr in neighbors if nbr.GetAtomicNum() == 8)  # Oxygen
                    h_count = atom.GetTotalNumHs()  # Hydrogen

                    # Condition: Carbon is bonded to both N and O and has exactly 1 H
                    if n_count >= 1 and o_count >= 1 and h_count == 1:
                        return True  # Restriction is violated

        return False
