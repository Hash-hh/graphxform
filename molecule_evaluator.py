"""
Provides functions to evaluate a completed molecule (validity checks, computing properties like QED or similarity, calling GuacaMol scoring functions, etc.) and assign the objective score.
The MoleculeDesign.objective is set here after generation.
"""

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
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem

from molecule_design import MoleculeDesign
from objective_predictor.GH_GNN_IDAC.src.models.utilities.mol2graph import get_dataloader_pairs_T, sys2graph, atom_features, n_atom_features, n_bond_features
from objective_predictor.GH_GNN_IDAC.src.models.GHGNN_architecture import GHGNN

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
        self.model = self._load_model()

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


class MoleculeObjectiveEvaluator:
    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.predictor_workers = [PredictorWorker.remote(self.config, self.device) for _ in range(self.config.num_predictor_workers)]

        # --- New: Setup for Similarity Objective ---
        self.use_similarity_objective = self.config.reference_smiles is not None
        if self.use_similarity_objective:
            self.reference_mol = Chem.MolFromSmiles(self.config.reference_smiles)
            if self.reference_mol:
                self.reference_fp = AllChem.GetMorganFingerprintAsBitVect(self.reference_mol, 2, nBits=2048)
                print(f"Similarity objective enabled. Reference: {self.config.reference_smiles}")
            else:
                self.use_similarity_objective = False
                print(f"Warning: Could not parse reference SMILES '{self.config.reference_smiles}'. Similarity objective is disabled.")

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

    def _calculate_similarity(self, gen_mol: Chem.RWMol) -> float:
        """Calculates Tanimoto similarity to the reference molecule."""
        if not self.use_similarity_objective or gen_mol is None:
            return 0.0
        try:
            gen_fp = AllChem.GetMorganFingerprintAsBitVect(gen_mol, 2, nBits=2048)
            return TanimotoSimilarity(self.reference_fp, gen_fp)
        except Exception:
            # Handle cases where fingerprint generation might fail
            return 0.0

    def predict_objective(self, molecule_designs: List[Union[MoleculeDesign, str]]) -> List[dict]:
        """
        Takes list of molecules and predicts the objective functions on them.
        For Pareto optimality, this now returns a list of dictionaries, where each
        dictionary contains the SMILES and a list of the two objectives.
        e.g. [{'smiles': 'CCO', 'objectives': [0.8, 0.9]}, ...]
        """
        feasible_molecules: List[Chem.RWMol] = []
        feasible_smiles: List[str] = []

        for mol_design in molecule_designs:
            mol = None
            smiles = None
            if isinstance(mol_design, MoleculeDesign):
                if not self.infeasible_by_special_constraints(mol_design):
                    mol = mol_design.rdkit_mol
                    smiles = Chem.MolToSmiles(mol)
            elif isinstance(mol_design, str) and mol_design != "C":
                try:
                    mol = Chem.MolFromSmiles(mol_design)
                    Chem.SanitizeMol(mol)
                    smiles = mol_design
                except:
                    continue # Skip infeasible SMILES

            if mol and smiles:
                feasible_molecules.append(mol)
                feasible_smiles.append(smiles)

        if not feasible_molecules:
            return []

        # --- 1. Calculate Primary Objective ---
        if self.config.objective_type in self.guacamol_benchmarks:
            primary_objs = np.array([
                self.guacamol_benchmarks[self.config.objective_type].objective.score(smi)
                for smi in feasible_smiles
            ])
        else:
            num_per_worker = math.ceil(len(feasible_molecules) / len(self.predictor_workers))
            future_objs = [
                worker.predict_objectives_from_rdkit_mols.remote(feasible_molecules[i * num_per_worker: (i+1) * num_per_worker])
                for i, worker in enumerate(self.predictor_workers)
            ]
            future_objs = ray.get(future_objs)
            primary_objs = np.concatenate(future_objs)

        # --- 2. Calculate Similarity Objective ---
        similarity_objs = np.array([self._calculate_similarity(mol) for mol in feasible_molecules])

        # --- 3. Combine into final list of dictionaries ---
        results = []
        for i in range(len(feasible_smiles)):
            # Ensure primary objective is a float, not numpy float
            primary_obj_float = float(primary_objs[i])
            if not np.isfinite(primary_obj_float):
                continue # Skip molecules with non-finite primary objectives

            results.append({
                "smiles": feasible_smiles[i],
                "objectives": [primary_obj_float, float(similarity_objs[i])]
            })

        return results

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