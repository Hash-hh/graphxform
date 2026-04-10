import os
# =========================================================
# PREVENT WINDOWS RAM / PAGEFILE CRASH (WINERROR 1455)
# =========================================================
os.environ["LOKY_MAX_CPU_COUNT"] = "1"  # Restrict joblib to 1 parallel workers
os.environ["OMP_NUM_THREADS"] = "1"     # Prevent PyTorch from over-threading
# =========================================================
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from copy import deepcopy
import numpy as np
from hydra.core.global_hydra import GlobalHydra

# # =========================================================
# # GRAPHIUM / SCIPY FLOAT16 MONKEY-PATCH
# # =========================================================
# import scipy.sparse
# import scipy.sparse._coo
#
# _orig_coo = scipy.sparse.coo_matrix
#
#
# class SafeCOO(_orig_coo):
#     def __init__(self, arg1, shape=None, dtype=None, copy=False):
#         if dtype is not None:
#             if 'float16' in str(np.dtype(dtype).name):
#                 dtype = np.float32
#
#         if isinstance(arg1, tuple) and len(arg1) == 2:
#             data, ij = arg1
#             if hasattr(data, 'dtype') and 'float16' in str(data.dtype):
#                 arg1 = (data.astype(np.float32), ij)
#         elif hasattr(arg1, 'dtype') and 'float16' in str(arg1.dtype):
#             arg1 = arg1.astype(np.float32)
#
#         super().__init__(arg1, shape=shape, dtype=dtype, copy=copy)
#
#
# scipy.sparse.coo_matrix = SafeCOO
# scipy.sparse._coo.coo_matrix = SafeCOO
# # =========================================================

from minimol import Minimol
from tdc.benchmark_group import admet_group


# ---------------------------------------------------------
# 1. GRAPHCORE'S EXACT ARCHITECTURE
# ---------------------------------------------------------
class TaskHead(nn.Module):
    def __init__(self, hidden_dim=512, input_dim=512, dropout=0.1, depth=3, combine=True):
        super(TaskHead, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, hidden_dim)
        self.final_dense = nn.Linear(input_dim + hidden_dim, 1) if combine else nn.Linear(hidden_dim, 1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.combine = combine
        self.depth = depth

    def forward(self, x):
        original_x = x
        x = self.dense1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.dense2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        if self.depth == 4:
            x = self.dense3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.dropout(x)

        x = torch.cat((x, original_x), dim=1) if self.combine else x
        x = self.final_dense(x)
        return x


def model_factory(hidden_dim, depth, combine, lr, epochs=25, warmup=5, weight_decay=0.0001, device='cuda'):
    model = TaskHead(hidden_dim=hidden_dim, depth=depth, combine=combine).to(device)
    optimiser = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Restored EXACTLY to Graphcore's BCELoss
    loss_fn = nn.BCELoss()

    def lr_fn(epoch):
        if epoch < warmup:
            return epoch / warmup
        else:
            return (1 + math.cos(math.pi * (epoch - warmup) / (epochs - warmup))) / 2

    lr_scheduler = LambdaLR(optimiser, lr_lambda=lr_fn)
    return model, optimiser, lr_scheduler, loss_fn


class OracleDataset(Dataset):
    def __init__(self, embeddings, targets):
        self.embeddings = embeddings
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[idx], dtype=torch.float32), torch.tensor(self.targets[idx],
                                                                                     dtype=torch.float32)


# ---------------------------------------------------------
# 2. THE RL ORACLE WRAPPER
# ---------------------------------------------------------
class MiniMolOracle:
    def __init__(self, task_name, cache_dir="./oracle_cache", device=None):
        self.task_name = task_name.lower()
        self.cache_dir = cache_dir
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Graphcore Sweep Hyperparameters
        self.hparams = {
            'bbb': {'hidden_dim': 2048, 'depth': 3, 'combine': True, 'lr': 0.0001},
            'herg': {'hidden_dim': 512, 'depth': 3, 'combine': True, 'lr': 0.0003}
        }

        if self.task_name not in self.hparams:
            raise ValueError(f"Unsupported MiniMol task: {self.task_name}")

        # --- NEW CODE: Clear Hydra Singleton ---
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        # -------------------------------------

        print(f"[MiniMolOracle] Initializing Foundation Model...")
        self.featuriser = Minimol()
        self.model_path = os.path.join(self.cache_dir, f"minimol_{self.task_name}_ensemble.pt")

        self.ensemble = self._load_or_train()

    def _load_or_train(self):
        if os.path.exists(self.model_path):
            print(f"[MiniMolOracle] Loading cached ensemble for {self.task_name}...")
            state_dicts = torch.load(self.model_path, map_location=self.device)
            ensemble = []
            for sd in state_dicts:
                model = TaskHead(
                    hidden_dim=self.hparams[self.task_name]['hidden_dim'],
                    depth=self.hparams[self.task_name]['depth'],
                    combine=self.hparams[self.task_name]['combine']
                ).to(self.device)
                model.load_state_dict(sd)
                model.eval()
                ensemble.append(model)
            return ensemble

        print(f"[MiniMolOracle] Training new Ensemble (5 models) for {self.task_name}...")
        return self._train_ensemble()

    def _train_ensemble(self):
        # Cantor pairing function used by Graphcore for seeds
        def cantor_pairing(a, b):
            return (a + b) * (a + b + 1) // 2 + b

        print("[MiniMolOracle] Downloading TDC ADMET benchmark group data...")
        group = admet_group(path='admet_data/')
        dataset_name = 'bbb_martins' if self.task_name == 'bbb' else 'herg'
        benchmark = group.get(dataset_name)
        name = benchmark['name']

        ensemble_state_dicts = []
        ensemble_models = []
        EPOCHS = 25
        ENSEMBLE_SIZE = 5
        REPETITIONS = 1
        seed1 = 1  # Replicating the first repetition of Graphcore's loop

        # Loop exactly matches Graphcore's ensemble fold logic
        for fold_i, seed2 in enumerate(range(REPETITIONS + 1, REPETITIONS + ENSEMBLE_SIZE + 1)):
            seed = cantor_pairing(seed1, seed2)
            print(f"Training Ensemble Member {fold_i + 1}/5 (Seed: {seed})...")

            # Extract exact train/val split for this seed, isolating the test set
            mols_train, mols_valid = group.get_train_valid_split(benchmark=name, split_type='default', seed=seed)

            # Generate Embeddings
            train_embs = self.featuriser(mols_train['Drug'].tolist())
            val_embs = self.featuriser(mols_valid['Drug'].tolist())

            train_embs = [emb if emb is not None else np.zeros(512) for emb in train_embs]
            val_embs = [emb if emb is not None else np.zeros(512) for emb in val_embs]

            train_targets = mols_train['Y'].values
            val_targets = mols_valid['Y'].values

            train_loader = DataLoader(OracleDataset(train_embs, train_targets), batch_size=32, shuffle=True)
            val_loader = DataLoader(OracleDataset(val_embs, val_targets), batch_size=128, shuffle=False)

            model, optimiser, lr_scheduler, loss_fn = model_factory(
                hidden_dim=self.hparams[self.task_name]['hidden_dim'],
                depth=self.hparams[self.task_name]['depth'],
                combine=self.hparams[self.task_name]['combine'],
                lr=self.hparams[self.task_name]['lr'],
                device=self.device
            )

            best_val_loss = float('inf')
            best_model_state = None

            for epoch in range(EPOCHS):
                model.train()
                lr_scheduler.step(epoch)

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimiser.zero_grad()
                    logits = model(inputs).squeeze()
                    # Reverted back to exact Graphcore BCE implementation
                    loss = loss_fn(torch.sigmoid(logits), labels)
                    loss.backward()
                    optimiser.step()

                # Validation
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        logits = model(inputs).squeeze()
                        val_loss += loss_fn(torch.sigmoid(logits), labels).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(model.state_dict())

            ensemble_state_dicts.append(best_model_state)

            best_model = TaskHead(
                hidden_dim=self.hparams[self.task_name]['hidden_dim'],
                depth=self.hparams[self.task_name]['depth'],
                combine=self.hparams[self.task_name]['combine']
            ).to(self.device)
            best_model.load_state_dict(best_model_state)
            best_model.eval()
            ensemble_models.append(best_model)

        torch.save(ensemble_state_dicts, self.model_path)
        print(f"[MiniMolOracle] Saved ensemble weights to {self.model_path}")
        return ensemble_models

    def __call__(self, smiles):
        if isinstance(smiles, str):
            smiles = [smiles]

        try:
            raw_embeddings = self.featuriser(smiles)
            valid_embeddings = [emb if emb is not None else np.zeros(512) for emb in raw_embeddings]
            inputs = torch.tensor(np.array(valid_embeddings), dtype=torch.float32).to(self.device)

            ensemble_logits = []
            with torch.no_grad():
                for model in self.ensemble:
                    logits = model(inputs).squeeze(-1)
                    ensemble_logits.append(logits)

            averaged_logits = torch.mean(torch.stack(ensemble_logits), dim=0)
            predictions = torch.sigmoid(averaged_logits).cpu().numpy()

            return predictions[0] if len(smiles) == 1 else predictions

        except Exception as e:
            print(f"[MiniMolOracle] Evaluation Error: {e}")
            return 0.0 if len(smiles) == 1 else np.zeros(len(smiles))