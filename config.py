import os
import datetime
import platform
import uuid

uid = uuid.uuid4().hex

class MoleculeConfig:

    @property
    def num_objectives(self):
        if self.objective_type in ['polypharmacy_2d', 'safety_2d']:
            return 2
        elif self.objective_type == 'tpp_3d':
            return 3
        return 1

    def __init__(self):
        self.seed = 42

        # =================================================================
        # NETWORK ARCHITECTURE
        # =================================================================
        self.latent_dimension = 512
        self.num_transformer_blocks = 10
        self.num_heads = 16
        self.dropout = 0.
        self.use_rezero_transformer = True

        # =================================================================
        # CONDITIONAL POLICY (Phase A: FiLM + corner-biased lambda)
        # =================================================================
        # FiLM conditioning applied per transformer block (gamma, beta modulation).
        # Zero-init so this is identity at start: safe for fine-tune from checkpoint.
        self.use_film = True
        # Keep the existing additive virtual-node lambda injection alongside FiLM.
        self.use_lambda_additive = True
        # Use corner/edge-biased lambda sampling during RL generation instead of
        # plain Dirichlet(1, 1, ..., 1). Addresses under-representation of corners.
        self.use_corner_sampling = True
        # If True, the sampler's "interior" bin is replaced by more edge samples
        # (A.4 in-distribution generalization test: train on extremes, eval on interior).
        self.restrict_training_lambda_to_extremes = False
        # If not None, use this fixed lambda at eval time (e.g. [0.5, 0.5]).
        # If None, each scaffold samples its own lambda at eval (random).
        self.eval_lambda = None

        # =================================================================
        # ENVIRONMENT
        # =================================================================
        self.disable_ray = False  # Set True to run without Ray (for debugging)
        self.wall_clock_limit = None
        self.max_num_atoms = 50

        self.atom_vocabulary = {  # Order matters!
            "C":    {"allowed": True, "atomic_number": 6, "valence": 4},
            "C-":   {"allowed": True, "atomic_number": 6, "valence": 3, "formal_charge": -1},
            "C+":   {"allowed": True, "atomic_number": 6, "valence": 5, "formal_charge": 1},
            "C@":   {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 1},
            "C@@":  {"allowed": True, "atomic_number": 6, "valence": 4, "chiral_tag": 2},

            "N":    {"allowed": True, "atomic_number": 7, "valence": 3},
            "N-":   {"allowed": True, "atomic_number": 7, "valence": 2, "formal_charge": -1},
            "N+":   {"allowed": True, "atomic_number": 7, "valence": 4, "formal_charge": 1},

            "O":    {"allowed": True, "atomic_number": 8, "valence": 2},
            "O-":   {"allowed": True, "atomic_number": 8, "valence": 1, "formal_charge": -1},
            "O+":   {"allowed": True, "atomic_number": 8, "valence": 3, "formal_charge": 1},

            "F":    {"allowed": True, "atomic_number": 9, "valence": 1},

            "P":    {"allowed": True, "atomic_number": 15, "valence": 7},
            "P-":   {"allowed": True, "atomic_number": 15, "valence": 6, "formal_charge": -1},
            "P+":   {"allowed": True, "atomic_number": 15, "valence": 8, "formal_charge": 1},

            "S":    {"allowed": True, "atomic_number": 16, "valence": 6},
            "S-":   {"allowed": True, "atomic_number": 16, "valence": 5, "formal_charge": -1},
            "S+":   {"allowed": True, "atomic_number": 16, "valence": 7, "formal_charge": 1},
            "S@":   {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 1},
            "S@@":  {"allowed": True, "atomic_number": 16, "valence": 6, "chiral_tag": 2},

            "Cl": {"allowed": True, "atomic_number": 17, "valence": 1},
            "Br": {"allowed": True, "atomic_number": 35, "valence": 1},
            "I": {"allowed": True, "atomic_number": 53, "valence": 1}
        }

        self.start_from_c_chains = True
        self.start_c_chain_max_len = 1
        self.start_from_smiles = None
        self.repeat_start_instances = 1
        self.synthetic_accessibility_in_objective_scale = 0
        self.include_structural_constraints = False

        # =================================================================
        # OBJECTIVE / TASK
        # =================================================================
        # Options: "polypharmacy_2d", "safety_2d", "tpp_3d", "kinase_mpo",
        #          "jnk3", "prodrug_bbb", or GuacaMol task names
        self.objective_type = "safety_2d"

        self.num_predictor_workers = 10
        self.objective_predictor_batch_size = 64
        self.objective_gnn_device = "cpu"

        # =================================================================
        # CHECKPOINTS
        # =================================================================
        self.load_checkpoint_from_path = "model/neurips/polypharmacy_2d.pt"
        # self.load_checkpoint_from_path = "results/2026-04-23--18-28-35/best_model182.pt"
        self.load_optimizer_state = False

        # =================================================================
        # TRAINING
        # =================================================================
        self.num_dataloader_workers = 10
        self.CUDA_VISIBLE_DEVICES = "0"
        self.training_device = "cuda:0"
        self.num_epochs = 500
        self.scale_factor_level_one = 1.
        self.scale_factor_level_two = 1.
        self.batch_size_training = 64
        self.num_batches_per_epoch = 20  # None = one pass through generated dataset

        self.optimizer = {
            "lr": 1e-4,
            "weight_decay": 0,
            "gradient_clipping": 1.,
            "schedule": {
                "decay_lr_every_epochs": 1,
                "decay_factor": 1
            }
        }

        # =================================================================
        # SEARCH (GUMBELDORE)
        # =================================================================
        self.gumbeldore_config = {
            "num_trajectories_to_keep": 100,
            "keep_intermediate_trajectories": False,
            "devices_for_workers": ["cuda:0"] * 1,
            "destination_path": f"./data/generated_molecules_{uid}.pickle",
            "batch_size_per_worker": 1,
            "batch_size_per_cpu_worker": 1,

            "search_type": "iid_mc",  # "beam_search" | "tasar" | "iid_mc" | "wor"
            "num_samples_per_instance": 32,
            "sampling_temperature": 1,

            "beam_width": 32,
            "replan_steps": 12,
            "num_rounds": 1,
            "deterministic": False,
            "nucleus_top_p": 1.,
            "pin_workers_to_core": False,
        }
        print("UID for this run:", uid)

        # =================================================================
        # RESULTS & LOGGING
        # =================================================================
        self.results_path = os.path.join("./results",
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
        print("Results path:", self.results_path)
        self.log_to_file = True

        # =================================================================
        # RL (Dr. GRPO)
        # =================================================================
        self.use_dr_grpo = True

        self.use_fragment_library = True
        self.fragment_library_path = "scaffold_splitting/zinc_splits_optimized/run_seed_42/train_scaffolds.txt"
        self.num_prompts_per_epoch = 10
        self.include_carbon_prompt = True

        self.evaluation_scaffolds_path = "scaffold_splitting/zinc_splits_optimized/run_seed_42/test_scaffolds_c.txt"
        self.validation_scaffolds_path = "scaffold_splitting/zinc_splits_optimized/run_seed_42/val_scaffolds.txt"
        self.use_validation_for_ckpt = True if self.use_dr_grpo else False

        self.ppo_epochs = 1
        self.rl_ppo_clip_epsilon = 0.2
        self.rl_entropy_beta = 0.

        self.rl_use_novelty_bonus = False
        self.rl_novelty_beta = 0.05
        self.rl_use_il_distillation = False

        self.rl_replay_microbatch_size = 128
        self.rl_streaming_backward = True
        self.rl_advantage_normalize = False
        self.rl_store_trajectories_path = None
        self.rl_max_group_size = None
        self.rl_log_advantages = False
        self.rl_assert_masks = False
        self.freeze_all_except_final_layer = False if self.use_dr_grpo else True
        self.use_grpo_grouping = True
        self.max_oracle_calls = None

        # Mixed precision
        self.use_amp = True
        self.amp_dtype = "bf16"
        self.use_amp_inference = True

        # =================================================================
        # PRODRUG-SPECIFIC
        # =================================================================
        self.prodrug_mode = False
        self.prodrug_parents_train = [
            "CN1CC[C@]23[C@@H]4[C@H]1CC5=C2C(=C(C=C5)O)O[C@H]3[C@H](C=C4)O",  # Morphine
            "C(CC(=O)O)CN",  # GABA
            "C1CNCCC1C(=O)O",  # Nipecotic Acid
            "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin
        ]
        self.prodrug_parents_test = [
            "C1=CC(=C(C=C1CCN)O)O",  # Dopamine
            "C1CC1CN2CC[C@]34[C@@H]5C(=O)CC[C@]3([C@H]2CC6=C4C(=C(C=C6)O)O5)O"  # Naltrexone
        ]
        self.bbb_weight_logp = 1.0
        self.bbb_weight_hdonor = 1.0
        self.bbb_weight_cleavable = 2.0
        self.bbb_weight_qed = 2.0
        self.bbb_weight_mw_penalty = 5.0
        self.bbb_max_mw = 600.0
        self.prodrug_parent_smiles = None
        self.prodrug_log_components = True

        # =================================================================
        # WANDB
        # =================================================================
        self.use_wandb = True
        self.wandb_project = "neurips"
        self.wandb_entity = "hasham"
        self.wandb_run_name = f"grxform_{self.objective_type}_Seed{self.seed}"

        if self.use_wandb == "auto":
            self.use_wandb = platform.system() == "Linux"
