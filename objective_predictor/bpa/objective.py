import numpy as np
import os
import warnings
import pandas as pd
import re  # For parsing PolyNC output

from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
from rdkit.Contrib.SA_Score import sascorer

# --- NEW IMPORTS for PolyNC ---
try:
    import torch
    from transformers import T5ForConditionalGeneration, T5Tokenizer
except ImportError:
    print("Error: 'torch' or 'transformers' package not found.")
    print("Please install them using: pip install torch transformers")
    exit()

# --- NEW IMPORT for ADMET-AI ---
# try:
from admet_ai import ADMETModel
# except ImportError:
#     print("Error: 'admet-ai' package not found.")
#     print("Please install it using: pip install admet-ai")
#     exit()

# --- Setup and Configuration ---
warnings.filterwarnings("ignore")

try:
    MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    MODULE_DIR = os.getcwd()  # Fallback for interactive environments


class BPA_Scorer:
    """
    Calculates a multi-objective reward for BPA alternatives, optimizing for:
    1.  Low Toxicity (admet-ai, 18 endpoints, thresholded)
    2.  High Glass Transition Temp (Tg) (PolyNC model)
    3.  High Synthetic Accessibility (SA Score) - Avoids "weird" molecules
    4.  Hard Structural Constraints (rings, -OH groups, distance)

    Also maintains a 'leaderboard' of the top 20 molecules seen.
    """

    # --- WEIGHTAGE CONFIGURATION ---
    # Adjust these values to change the priority of each component.
    # The code will automatically normalize them to sum to 1.0.
    # WEIGHT_TOXICITY = 0.4  # Priority for low toxicity
    WEIGHT_TOXICITY = 0.4 # Priority for low toxicity
    WEIGHT_TG = 0.4  # Priority for high Tg
    # WEIGHT_TG = 0.2  # Priority for high Tg
    # WEIGHT_SA = 0.4  # Priority for ease of synthesis
    WEIGHT_SA = 0.2 # Priority for ease of synthesis

    def __init__(self, config):

        print("Initializing BPA-Free Polymer Objective (Toxicity + Tg + SA)...")

        # --- 1. ADMET-AI (Toxicity) Initialization ---
        print("  Loading 'admet-ai' model suite...")
        try:
            self.admet_model = ADMETModel()
            print("  ✅ 'admet-ai' loaded.")
        except Exception as e:
            print(f"Error initializing ADMETModel: {e}")
            raise

        polync_device = config.polync_device

        # --- 2. PolyNC (Tg) Initialization ---
        print(f"  Loading 'PolyNC' model onto {polync_device}...")
        try:
            self.polync_device = torch.device(polync_device)
            self.polync_tokenizer = T5Tokenizer.from_pretrained("hkqiu/PolyNC")
            self.polync_model = T5ForConditionalGeneration.from_pretrained("hkqiu/PolyNC").to(self.polync_device)
            self.polync_model.eval()
            print("  ✅ 'PolyNC' model loaded.")
        except Exception as e:
            print(f"Error initializing PolyNC model: {e}")
            raise

        # --- 3. Define Toxicity Endpoints (Probabilities) ---
        self.tox_prob_columns = [
            'hERG', 'ClinTox', 'AMES', 'DILI', 'Carcinogens_Lagunin',
            'NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase',
            'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5',
            'SR-HSE', 'SR-MMP', 'SR-p53'
        ]

        # --- 4. Define Structural Constraint Constants ---
        self.MIN_OH_DISTANCE = 9
        # self.oh_pattern = Chem.MolFromSmarts('[OH]')
        # self.psmiles_oh_H_pattern = Chem.MolFromSmarts('[H][O]')

        # Pattern for Valid Hydroxyls (C-OH, not Acid)
        self.oh_pattern = Chem.MolFromSmarts('[OH;$(O[#6]);!$(OC=O)]')
        self.psmiles_oh_H_pattern = Chem.MolFromSmarts('[H][O;$(O[#6]);!$(OC=O)]')

        # --- NEW: Interfering Group Patterns (BAN LIST) ---
        self.interfering_patterns = [
            # 1. Carboxylic Acids (-C(=O)OH)
            Chem.MolFromSmarts('[CX3](=O)[OH]'),
            # 2. Amines (Primary/Secondary) - N with at least one H
            #    (We exclude Amides 'NC=O' from this simple check if you want,
            #     but for pure PC, banning all N-H is safer).
            Chem.MolFromSmarts('[NX3;H2,H1]'),
            # 3. Thiols (-SH)
            Chem.MolFromSmarts('[#16;H]'),
            # 4. Aldehydes (-CH=O)
            Chem.MolFromSmarts('[CX3H1](=O)'),
            # 5. Isocyanates (-N=C=O) - highly reactive
            Chem.MolFromSmarts('N=C=O'),
            # 6. Anhydrides
            Chem.MolFromSmarts('[CX3](=O)O[CX3](=O)')
        ]

        # --- 5. Define Objective Weights (Normalized) ---
        total_weight = self.WEIGHT_TOXICITY + self.WEIGHT_TG + self.WEIGHT_SA
        self.w_tox = self.WEIGHT_TOXICITY / total_weight
        self.w_tg = self.WEIGHT_TG / total_weight
        self.w_sa = self.WEIGHT_SA / total_weight

        # --- 6. Define Normalization Ranges ---
        self.tg_min = 50.0
        self.tg_max = 400.0
        self.tg_range = self.tg_max - self.tg_min

        # --- 7. NEW: Leaderboard Setup ---
        self.top_k = 20
        self.results_dir = os.path.join(MODULE_DIR, 'bpa_results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.leaderboard_file = os.path.join(self.results_dir, 'bpa_top20_leaderboard.txt')

        # Added 'sa_score' to columns
        self.leaderboard_cols = ['smiles', 'combined_score', 'tox_score', 'tg_score', 'sa_score', 'actual_tg']

        try:
            self.leaderboard_df = pd.read_csv(self.leaderboard_file, sep='\t')
            print(f"  Loaded existing leaderboard from {self.leaderboard_file}")
            # Ensure new column exists if loading old file
            if 'sa_score' not in self.leaderboard_df.columns:
                self.leaderboard_df['sa_score'] = 0.0
        except FileNotFoundError:
            self.leaderboard_df = pd.DataFrame(columns=self.leaderboard_cols)
            print("  Initialized new leaderboard.")

        print(f"  Tracking {len(self.tox_prob_columns)} toxicity endpoints.")
        print("  Tracking 1 Tg regression endpoint (PolyNC).")
        print("  Tracking Synthetic Accessibility (SA Score).")
        print(f"  Weights :: Toxicity: {self.w_tox:.2f}, Tg: {self.w_tg:.2f}, SA: {self.w_sa:.2f}")
        print("  Enforcing hard structural constraints (No Halogens, 2+ Rings, Size 4-6).")
        print("✅ BPA-Free Polymer Objective initialized.")

    def _get_rdkit_mol(self, smiles: str):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return None
            Chem.SanitizeMol(mol)
            return mol
        except Exception:
            return None

    def _check_constraints(self, mol: Chem.Mol) -> bool:
        try:
            # 1. No Halogens (F, Cl, Br, I)
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in [9, 17, 35, 53]:
                    return False

            # --- NEW CHECK: No Interfering Functional Groups ---
            for pattern in self.interfering_patterns:
                if mol.HasSubstructMatch(pattern):
                    return False  # Contains a banned reactive group

            # 2. Must have at least two rings
            ring_info = mol.GetRingInfo()
            atom_rings = ring_info.AtomRings()
            if len(atom_rings) < 2: return False

            # 3. Rings must be size 4, 5 or 6
            for ring in atom_rings:
                if len(ring) not in [4, 5, 6]:
                    return False

            # 4. Must have exactly 2 Hydroxyl groups
            matches = mol.GetSubstructMatches(self.oh_pattern)
            valid_oh_indices = []
            for match in matches:
                o_idx = match[0]
                atom = mol.GetAtomWithIdx(o_idx)
                if atom.GetTotalNumHs() > 0:
                    valid_oh_indices.append(o_idx)

            if len(valid_oh_indices) != 2:
                return False

            # 5. Topological Distance
            oh_oxygen_atoms = [valid_oh_indices[0], valid_oh_indices[1]]
            path = Chem.GetShortestPath(mol, oh_oxygen_atoms[0], oh_oxygen_atoms[1])
            distance = len(path) - 1

            if distance < self.MIN_OH_DISTANCE: return False

            return True
        except Exception:
            return False

    def _construct_psmiles(self, mol: Chem.Mol) -> str:
        try:
            mol_with_H = Chem.AddHs(mol)
            matches = mol_with_H.GetSubstructMatches(self.psmiles_oh_H_pattern)
            if len(matches) != 2:
                raise ValueError(f"Expected 2 OH-matches, found {len(matches)}")

            h1_idx, o1_idx = matches[0][0], matches[0][1]
            h2_idx, o2_idx = matches[1][0], matches[1][1]

            rw_mol = Chem.RWMol(mol_with_H)

            # Mod 1: First H -> *
            h1_atom = rw_mol.GetAtomWithIdx(h1_idx)
            h1_atom.SetAtomicNum(0)
            h1_atom.SetIsotope(0)
            h1_atom.SetFormalCharge(0)
            h1_atom.SetNoImplicit(True)

            # Mod 2: Second H -> Carbonate
            rw_mol.RemoveAtom(h2_idx)
            c_carb_idx = rw_mol.AddAtom(Chem.Atom(6))
            o_carb_idx = rw_mol.AddAtom(Chem.Atom(8))
            star_2_idx = rw_mol.AddAtom(Chem.Atom(0))

            star_2_atom = rw_mol.GetAtomWithIdx(star_2_idx)
            star_2_atom.SetIsotope(0)
            star_2_atom.SetFormalCharge(0)
            star_2_atom.SetNoImplicit(True)

            rw_mol.AddBond(o2_idx, c_carb_idx, Chem.BondType.SINGLE)
            rw_mol.AddBond(c_carb_idx, o_carb_idx, Chem.BondType.DOUBLE)
            rw_mol.AddBond(c_carb_idx, star_2_idx, Chem.BondType.SINGLE)

            final_mol = Chem.RemoveHs(rw_mol.GetMol())
            Chem.SanitizeMol(final_mol)

            return Chem.MolToSmiles(final_mol, isomericSmiles=False, canonical=True)

        except Exception as e:
            print(f"Error in _construct_psmiles: {e}")
            return None

    def _parse_polync_output(self, text_output: str) -> float:
        try:
            match = re.search(r"[-+]?\d*\.\d+|\d+", text_output)
            return float(match.group(0)) if match else -np.inf
        except ValueError:
            return -np.inf

    def _predict_tg(self, psmiles_list: list) -> list:
        if not psmiles_list: return []
        valid_psmiles_list = [ps for ps in psmiles_list if ps is not None]
        if not valid_psmiles_list: return [-np.inf] * len(psmiles_list)

        prompts = [f"Predict the Tg of the following SMILES: {s}" for s in valid_psmiles_list]
        try:
            inputs = self.polync_tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                                           max_length=150).to(self.polync_device)
            with torch.no_grad():
                outputs = self.polync_model.generate(**inputs, max_new_tokens=8)
            decoded = self.polync_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            scores = [self._parse_polync_output(out) for out in decoded]

            score_iter = iter(scores)
            return [-np.inf if ps is None else next(score_iter) for ps in psmiles_list]
        except Exception as e:
            print(f"Error during PolyNC prediction: {e}")
            return [-np.inf] * len(psmiles_list)

    def _update_leaderboard(self, batch_df: pd.DataFrame):
        if batch_df.empty: return
        try:
            combined_df = pd.concat([self.leaderboard_df, batch_df], ignore_index=True)
            combined_df = combined_df.sort_values(by='combined_score', ascending=False)
            combined_df = combined_df.drop_duplicates(subset='smiles', keep='first')
            self.leaderboard_df = combined_df.head(self.top_k).reset_index(drop=True)
            self.leaderboard_df.to_csv(self.leaderboard_file, sep='\t', index=False, float_format='%.4f')
        except Exception as e:
            print(f"Warning: Could not update leaderboard. {e}")

    def __call__(self, smiles_list: list) -> np.ndarray:
        final_scores = np.zeros(len(smiles_list))

        # --- 1. Pre-filter ---
        valid_mols_map = {}
        valid_smiles_to_idx = {}

        for i, smiles in enumerate(smiles_list):
            mol = self._get_rdkit_mol(smiles)
            if mol is None or not self._check_constraints(mol):
                final_scores[i] = 0.0
                continue
            valid_mols_map[i] = mol
            vs = Chem.MolToSmiles(mol)
            if vs not in valid_smiles_to_idx: valid_smiles_to_idx[vs] = []
            valid_smiles_to_idx[vs].append(i)

        if not valid_mols_map: return final_scores

        # --- 2. Predictions ---
        unique_mols = {smiles: valid_mols_map[indices[0]] for smiles, indices in valid_smiles_to_idx.items()}
        unique_monomer_smiles = list(unique_mols.keys())

        try:
            # A. Toxicity
            preds_df = self.admet_model.predict(smiles=unique_monomer_smiles)
            tox_probs = preds_df[self.tox_prob_columns]
            final_tox_score = (1.0 - (tox_probs - 0.5).mul(2.0).clip(lower=0.0)).mean(axis=1)

            # B. Tg
            unique_psmiles = [self._construct_psmiles(mol) for mol in unique_mols.values()]
            tg_raw = self._predict_tg(unique_psmiles)
            tg_series = pd.Series(tg_raw, index=unique_monomer_smiles)
            final_tg_score = ((tg_series - self.tg_min) / self.tg_range).clip(0.0, 1.0)

            # C. SA Score (Avoiding "Weird" Molecules)
            # sascorer returns 1 (easy) to 10 (hard).
            # We want to maximize, so we invert: (10 - SA) / 9.
            # Result: 1.0 = Very Easy, 0.0 = Very Hard.
            sa_raw = [sascorer.calculateScore(mol) for mol in unique_mols.values()]
            sa_series = pd.Series(sa_raw, index=unique_monomer_smiles)
            final_sa_score = (10.0 - sa_series) / 9.0
            final_sa_score = final_sa_score.clip(0.0, 1.0)  # Just in case

            # D. Combined Score
            combined_scores = (self.w_tox * final_tox_score) + \
                              (self.w_tg * final_tg_score) + \
                              (self.w_sa * final_sa_score)

            # E. Leaderboard
            batch_df = pd.DataFrame({
                'smiles': combined_scores.index,
                'combined_score': combined_scores.values,
                'tox_score': final_tox_score.reindex(combined_scores.index).values,
                'tg_score': final_tg_score.reindex(combined_scores.index).values,
                'sa_score': final_sa_score.reindex(combined_scores.index).values,  # Saved normalized score
                'actual_tg': tg_series.reindex(combined_scores.index).values
            })
            self._update_leaderboard(batch_df)

            # F. Map back
            for smiles, score in combined_scores.items():
                if pd.isna(score): score = 0.0
                for idx in valid_smiles_to_idx[smiles]:
                    final_scores[idx] = max(0.0, score)

        except Exception as e:
            print(f"Error during scoring: {e}")

        return np.array(final_scores)