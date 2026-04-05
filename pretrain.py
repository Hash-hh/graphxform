import argparse
import copy
import importlib
import os

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from molecule_dataset import RandomMoleculeDataset

import torch
import numpy as np
import wandb
from config import MoleculeConfig
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, config: MoleculeConfig, network: MoleculeTransformer,
                        optimizer: torch.optim.Optimizer, datasets: list, is_validation=False):
    """
    Accepts a list of RandomMoleculeDataset instances and implements a Round-Robin
    batching strategy to sample evenly across all provided tasks.
    """
    # Create a dataloader for EACH dataset passed in
    dataloaders = [
        DataLoader(ds, batch_size=1, shuffle=not is_validation, num_workers=config.num_dataloader_workers,
                   pin_memory=True, persistent_workers=False)
        for ds in datasets
    ]

    metrics = dict()
    # Train for one epoch
    network.train() if not is_validation else network.eval()

    accumulated_loss = 0
    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0

    # Calculate total batches across all dataloaders
    num_batches = sum(len(dl) for dl in dataloaders)
    progress_bar = tqdm(total=num_batches)

    # Setup the Round-Robin queue
    data_iters = [iter(dl) for dl in dataloaders]

    for _ in range(num_batches):
        # --- Round Robin Logic ---
        data = None
        for _ in range(len(data_iters)):
            if not data_iters:
                break
            current_iter = data_iters.pop(0)  # Pop from the front
            try:
                data = next(current_iter)
                data_iters.append(current_iter)  # If successful, put it at the back of the queue
                break
            except StopIteration:
                pass  # If this dataloader is exhausted, we drop it from the queue

        if data is None:
            break  # All iterators exhausted
        # -------------------------

        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

        # Teacher Forcing Override
        # If the expert dataset says an action is correct, we must guarantee it is unmasked.
        # This immunizes the loss function against environment edge-cases (like Kekulization shifts).
        for i in range(target_zero.size(0)):
            if target_zero[i] != -1:
                input_data["feasibility_mask_level_zero"][i, target_zero[i]] = False
        for i in range(target_one.size(0)):
            if target_one[i] != -1:
                input_data["feasibility_mask_level_one"][i, target_one[i]] = False
        for i in range(target_two.size(0)):
            if target_two[i] != -1:
                input_data["feasibility_mask_level_two"][i, target_two[i]] = False
        # ------------------------------------------------

        # We mask the output according to feasibility
        logits_zero[input_data["feasibility_mask_level_zero"]] = float("-inf")
        logits_one[input_data["feasibility_mask_level_one"]] = float("-inf")
        logits_two[input_data["feasibility_mask_level_two"]] = float("-inf")

        criterion = CrossEntropyLoss(reduction="mean", ignore_index=-1)
        loss_zero = criterion(logits_zero, target_zero)
        loss_zero = torch.tensor(0.) if torch.isnan(loss_zero) else loss_zero
        loss_one = criterion(logits_one, target_one)
        loss_one = torch.tensor(0.) if torch.isnan(loss_one) else loss_one
        loss_two = criterion(logits_two, target_two)
        loss_two = torch.tensor(0.) if torch.isnan(loss_two) else loss_two
        loss = loss_zero + config.scale_factor_level_one * loss_one + config.scale_factor_level_two * loss_two

        if not is_validation:  # backward pass training
            # Optimization step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if config.optimizer["gradient_clipping"] > 0:
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

            optimizer.step()

        batch_loss = loss.item()
        accumulated_loss += batch_loss
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.update(1)
        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    progress_bar.close()

    metric_prefix = "" if not is_validation else "val_"

    # Protect against division by zero if a dataset is empty
    safe_num_batches = num_batches if num_batches > 0 else 1

    metrics[f"{metric_prefix}full_loss"] = accumulated_loss / safe_num_batches
    metrics[f"{metric_prefix}loss_level_zero"] = accumulated_loss_lvl_zero / safe_num_batches
    metrics[f"{metric_prefix}loss_level_one"] = accumulated_loss_lvl_one / safe_num_batches
    metrics[f"{metric_prefix}loss_level_two"] = accumulated_loss_lvl_two / safe_num_batches

    return metrics


if __name__ == '__main__':
    # 6 Separated Pickle Files
    train_files = [
        "./data/chembl/pretrain_sequences/chembl_train_additive.pickle",
        "./data/chembl/pretrain_sequences/chembl_train_removal.pickle",
        "./data/chembl/pretrain_sequences/chembl_train_replacement.pickle"
    ]
    val_files = [
        "./data/chembl/pretrain_sequences/chembl_valid_additive.pickle",
        "./data/chembl/pretrain_sequences/chembl_valid_removal.pickle",
        "./data/chembl/pretrain_sequences/chembl_valid_replacement.pickle"
    ]

    pretrain_num_epochs = 1000
    batch_size = 32
    num_batches_per_epoch = 3000
    batch_size_validation = 512
    training_device = "cuda:0"  # Device on which to train.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None

    print(">> Pretraining Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    args = parser.parse_args()

    if args.config is not None:
        # Load config from given path
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig

    config = MoleculeConfig()
    print(f"Results path: {config.results_path}")
    config.max_num_atoms = None
    config.training_device = training_device
    config.num_dataloader_workers = num_dataloader_workers

    logger = Logger(args, config.results_path, config.log_to_file, config=config)
    logger.log_hyperparams(config)

    # --- Initialize WandB ---
    if config.use_wandb:
        print(f"Initializing WandB (Project: {config.wandb_project}, Run: {config.wandb_run_name})...")
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity if config.wandb_entity else None,
            name=config.wandb_run_name,
            config={k: v for k, v in vars(config).items() if not k.startswith("__")}
        )

    # Fix random number generator seed for better reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup the neural network for training
    network = MoleculeTransformer(config, config.training_device)

    # Load checkpoint if needed
    if load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {load_checkpoint_from_path}")
        checkpoint = torch.load(load_checkpoint_from_path)
        print(f"{checkpoint['pretrain_epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "pretrain_epochs_trained": 0,
            "pretrain_best_validation_loss": float("inf"),
            "epochs_trained": 0,
            "validation_metric": float("-inf"),  # objective of the best molecule designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    if pretrain_num_epochs > 0:
        # Training loop
        print(f"Starting pre-training for {pretrain_num_epochs} epochs.")

        print("Setting up optimizer.")
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.optimizer["lr"],
            weight_decay=config.optimizer["weight_decay"]
        )
        if checkpoint["optimizer_state"] is not None and config.load_optimizer_state:
            print("Loading optimizer state from checkpoint.")
            optimizer.load_state_dict(
                checkpoint["optimizer_state"]
            )
        print("Setting up LR scheduler")
        _lambda = lambda epoch: config.optimizer["schedule"]["decay_factor"] ** (
                checkpoint["pretrain_epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        # Divide the total desired training batches evenly among the 3 task types
        batches_per_task = num_batches_per_epoch // len(train_files)

        train_datasets = [
            RandomMoleculeDataset(config, path, batch_size=batch_size, custom_num_batches=batches_per_task)
            for path in train_files
        ]

        # Cap Validation Batches to prevent massive epoch delays
        val_batches_per_task = int((num_batches_per_epoch // len(val_files)) * (batch_size / batch_size_validation))

        val_datasets = [
            RandomMoleculeDataset(config, path, batch_size=batch_size_validation,
                                  custom_num_batches=val_batches_per_task, no_random=True)
            for path in val_files
        ]
        task_names = ["Additive", "Removal", "Replacement"]

        for epoch in range(pretrain_num_epochs):
            # Accumulator dictionary for WandB to keep steps perfectly synced
            epoch_wandb_metrics = {}

            print(f"\n--- Epoch {checkpoint['pretrain_epochs_trained']} ---")
            print("Training (Round-Robin)...")
            generated_loggable_dict = train_for_one_epoch(
                epoch, config, network, optimizer, train_datasets
            )
            checkpoint["pretrain_epochs_trained"] += 1
            scheduler.step()

            print(f">> Train | Avg loss level 0: {generated_loggable_dict['loss_level_zero']:.4f}, "
                  f"Avg loss level 1: {generated_loggable_dict['loss_level_one']:.4f}, "
                  f"Avg loss level 2: {generated_loggable_dict['loss_level_two']:.4f}")
            logger.log_metrics(generated_loggable_dict, step=epoch)

            # Queue train metrics for WandB
            epoch_wandb_metrics.update({f"Train/{k}": v for k, v in generated_loggable_dict.items()})

            # Evaluate tasks separately
            print("Validating...")
            torch.cuda.empty_cache()

            blended_val_loss = 0.0

            with torch.no_grad():
                for val_ds, task_name in zip(val_datasets, task_names):
                    # Pass a list of length 1 to evaluate this specific task
                    val_metrics = train_for_one_epoch(
                        None, config, network, None, datasets=[val_ds], is_validation=True
                    )

                    print(f">> Val ({task_name}) | "
                          f"L0: {val_metrics['val_loss_level_zero']:.4f}, "
                          f"L1: {val_metrics['val_loss_level_one']:.4f}, "
                          f"L2: {val_metrics['val_loss_level_two']:.4f}")

                    # Log specific task metrics locally
                    task_specific_metrics = {f"{task_name}_{k}": v for k, v in val_metrics.items()}
                    logger.log_metrics(task_specific_metrics, step=epoch)

                    # Queue validation metrics for WandB
                    epoch_wandb_metrics.update({f"Val_{task_name}/{k}": v for k, v in val_metrics.items()})

                    blended_val_loss += val_metrics["val_full_loss"]

            blended_val_loss /= len(val_datasets)
            epoch_wandb_metrics["Val_Overall/blended_val_loss"] = blended_val_loss

            # Push accumulated metrics to WandB for this epoch
            if config.use_wandb:
                wandb.log(epoch_wandb_metrics, step=epoch)

            # Save model
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(
                dict_to_cpu(optimizer.state_dict())
            )

            save_checkpoint(checkpoint, "last_model.pt", config)

            # Check against the blended validation loss
            if blended_val_loss < checkpoint["pretrain_best_validation_loss"]:
                print(">> Got new best model.")
                checkpoint["pretrain_best_validation_loss"] = blended_val_loss
                save_checkpoint(checkpoint, "best_model.pt", config)

    logger.finish()
    if config.use_wandb:
        wandb.finish()