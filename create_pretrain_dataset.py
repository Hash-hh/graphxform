from core.decomposer import TrajectoryDecomposer
from core.fragment import load_fragment_vocabulary
from config import MoleculeConfig
import pickle

# Load config and vocabulary
config = MoleculeConfig()
config.use_fragment_action_space = True
config.fragment_vocabulary = load_fragment_vocabulary(
    config.fragment_vocabulary_path,
    top_k=config.fragment_top_k,
)

# Create decomposer
decomposer = TrajectoryDecomposer(config)

# Load ChEMBL SMILES
for datatype in ["valid", "train"]:
    with open(f"data/chembl_{datatype}.smiles") as f:
        smiles_list = [line.strip() for line in f]

    # Decompose all molecules
    trajectories = decomposer.decompose_batch(smiles_list, verbose=True)
    print(f"Decomposed {len(trajectories)}/{len(smiles_list)} molecules")

    # Save trajectories for training

    with open(f"data/chembl/pretrain_sequences/chembl_{datatype}.pkl", "wb") as f:
        pickle.dump(trajectories, f)