"""
Defines the MoleculeConfig (model hyperparameters, vocabulary, search settings, etc.) and default experiment settings (initially set for the celecoxib_rediscovery GuacaMol benchmark).
Modify this for different tasks or to point to a pre-trained checkpoint.
"""

import os
import datetime


class MoleculeConfig:
    def __init__(self):
        self.seed = 42

        # Network and environment
        self.latent_dimension = 512
        self.num_transformer_blocks = 10
        self.num_heads = 16
        self.dropout = 0.
        self.use_rezero_transformer = True

        # Environment options
        self.wall_clock_limit = None  # in seconds. If no limit, set to None
        self.max_num_atoms = 50

        # self.atom_vocabulary = {  # Attention! Order matters!
        #     "C":    {"allowed": True, "atomic_number": 6, "valence": 4},
        #     "C-":   {"allowed": True, "atomic_number": 6, "valence": 3, "formal_charge": -1},
        #     "C+":   {"allowed": True, "atomic_number": 6, "valence": 5, "formal_charge": 1},
        #     "C@":   {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 1},
        #     "C@@":  {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 2},
        #
        #     "N":    {"allowed": True, "atomic_number": 7, "valence": 3},
        #     "N-":   {"allowed": True, "atomic_number": 7, "valence": 2, "formal_charge": -1},
        #     "N+":   {"allowed": True, "atomic_number": 7, "valence": 4, "formal_charge": 1},
        #
        #     "O":    {"allowed": True, "atomic_number": 8, "valence": 2},
        #     "O-":   {"allowed": True, "atomic_number": 8, "valence": 1, "formal_charge": -1},
        #     "O+":   {"allowed": True, "atomic_number": 8, "valence": 3, "formal_charge": 1},
        #
        #     "F":    {"allowed": True, "atomic_number": 9, "valence": 1},
        #
        #     "P":    {"allowed": True, "atomic_number": 15, "valence": 7},
        #     "P-":   {"allowed": True, "atomic_number": 15, "valence": 6, "formal_charge": -1},
        #     "P+":   {"allowed": True, "atomic_number": 15, "valence": 8, "formal_charge": 1},
        #
        #     "S":    {"allowed": True, "atomic_number": 16, "valence": 6},
        #     "S-":   {"allowed": True, "atomic_number": 16, "valence": 5, "formal_charge": -1},
        #     "S+":   {"allowed": True, "atomic_number": 16, "valence": 7, "formal_charge": 1},
        #     "S@":   {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 1},
        #     "S@@":  {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 2},
        #
        #     "Cl": {"allowed": True, "atomic_number": 17, "valence": 1},
        #     "Br": {"allowed": True, "atomic_number": 35, "valence": 1},
        #     "I": {"allowed": True, "atomic_number": 53, "valence": 1}
        # }

        self.atom_vocabulary = {  # Attention! Order matters!
            "C": {"allowed": True, "atomic_number": 6, "valence": 4},
            "N": {"allowed": True, "atomic_number": 7, "valence": 3},
            "O": {"allowed": True, "atomic_number": 8, "valence": 2}
        }

        self.start_from_c_chains = True
        self.start_c_chain_max_len = 1
        self.start_from_smiles = None  # Give SMILES and set `start_from_c_chains=False`.
        self.repeat_start_instances = 1
        # Positive value x, where the actual objective with our molecule score will be set to obj = score - x * SA_score
        self.synthetic_accessibility_in_objective_scale = 0
        # Enforce structural constraints (see molecule evaluator)
        self.include_structural_constraints = True

        # SMILES string of the molecule to be similar to.
        # Set to None to disable similarity objective.
        self.reference_smiles = "CCCCCCCCCCCCCCCCCO"  # best list screen for DMBA_TMB task

        # Strategy to select molecules from the Pareto front for training.
        # Options: 'crowding_distance' or 'random'
        self.pareto_selection_strategy = 'crowding_distance'

        # Objective molecule predictor
        self.GHGNN_model_path = os.path.join("objective_predictor/GH_GNN_IDAC/models/GHGNN.pth")
        self.GHGNN_hidden_dim = 113
        # self.objective_type = "celecoxib_rediscovery"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        self.objective_type = "DMBA_TMB"  # either "IBA" or "DMBA_TMB" for solvent design, or goal-directed task from GuacaMol (see README)
        self.num_predictor_workers = 1  # num of parallel workers that operate on a given list of molecules
        self.objective_predictor_batch_size = 64
        self.objective_gnn_device = "cpu"  # device on which the GNN should live

        # Loading trained checkpoints to resume training or evaluate
        self.load_checkpoint_from_path = None  # If given, model checkpoint is loaded from this path.
        self.load_optimizer_state = False  # If True, the optimizer state is also loaded.

        # Training
        self.num_dataloader_workers = 4  # Number of workers for creating batches for training
        self.CUDA_VISIBLE_DEVICES = "0"  # Must be set, as ray can have problems detecting multiple GPUs
        self.training_device = "cuda:0"  # Device on which to perform the supervised training
        self.num_epochs = 1000  # Number of epochs (i.e., passes through training set) to train
        self.scale_factor_level_one = 1.
        self.scale_factor_level_two = 1.
        self.batch_size_training = 64
        self.num_batches_per_epoch = 20  # Can be None, then we just do one pass through generated dataset

        # Optimizer
        self.optimizer = {
            "lr": 1e-4,  # learning rate
            "weight_decay": 0,
            "gradient_clipping": 1.,  # Clip gradient to given L2-norm. Set to 0 if no clipping should be performed.
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # Self-improvement sequence decoding
        self.gumbeldore_config = {
            # Number of trajectories with the the highest objective function evaluation to keep for training
            "num_trajectories_to_keep": 100,
            "keep_intermediate_trajectories": True,  # if True, we consider all intermediate, terminable trajectories
            "devices_for_workers": ["cuda:0"] * 1,
            "destination_path": "./data/generated_molecules.pickle",
            "batch_size_per_worker": 1,  # Keep at one, as we only have three atoms from which we can start
            "batch_size_per_cpu_worker": 1,
            "search_type": "tasar",
            "beam_width": 512,
            "replan_steps": 12,
            "num_rounds": 1,  # if it's a tuple, then we sample as long as it takes to obtain a better trajectory, but for a minimum of first entry rounds and a maximum of second entry rounds
            "deterministic": False,  # Only use for gumbeldore_eval=True below, switches to regular beam search.
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False
        }

        # Results and logging
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime(
                                             "%Y-%m-%d--%H-%M-%S"))  # Path to store the model weights
        self.log_to_file = True
