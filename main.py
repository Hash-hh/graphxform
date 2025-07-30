import argparse
import copy
import importlib
import os
import time

from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from logger import Logger
from molecule_dataset import RandomMoleculeDataset

os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"
import ray
import torch
import numpy as np
from config import MoleculeConfig
from core.gumbeldore_dataset import GumbeldoreDataset
from model.molecule_transformer import MoleculeTransformer, dict_to_cpu
from molecule_evaluator import MoleculeObjectiveEvaluator


def save_checkpoint(checkpoint: dict, filename: str, config: MoleculeConfig):
    os.makedirs(config.results_path, exist_ok=True)
    path = os.path.join(config.results_path, filename)
    torch.save(checkpoint, path)


def train_for_one_epoch(epoch: int, config: MoleculeConfig, network: MoleculeTransformer, network_weights: dict,
                        optimizer: torch.optim.Optimizer, objective_evaluator: MoleculeObjectiveEvaluator, best_objective: float):

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    metrics = gumbeldore_dataset.generate_dataset(
        network_weights,
        best_objective=best_objective,
        memory_aggressive=False
    )
    print("Generated molecules")
    print(f"Mean obj. over fresh best mols: {metrics['mean_best_gen_obj']:.3f}")
    print(f"Best / worst obj. over fresh best mols: {metrics['best_gen_obj']:.3f}, {metrics['worst_gen_obj']:.3f}")
    print(f"Mean obj. over all time top 20 mols: {metrics['mean_top_20_obj']:.3f}")
    print(f"All time best mol: {list(metrics['top_20_molecules'][0].values())[0]:.3f}")
    torch.cuda.empty_cache()
    time.sleep(1)
    print("---- Loading dataset")
    dataset = RandomMoleculeDataset(config, config.gumbeldore_config["destination_path"], batch_size=config.batch_size_training,
                                    custom_num_batches=config.num_batches_per_epoch)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=config.num_dataloader_workers,
                            pin_memory=True, persistent_workers=True)

    # Train for one epoch
    network.train()

    # freeze layers except the last
    for parameter in network.parameters():
        parameter.requires_grad = False
    network.virtual_atom_linear.weight.requires_grad = True
    network.virtual_atom_linear.bias.requires_grad = True
    network.bond_atom_linear.weight.requires_grad = True
    network.bond_atom_linear.bias.requires_grad = True

    accumulated_loss_lvl_zero = 0
    accumulated_loss_lvl_one = 0
    accumulated_loss_lvl_two = 0
    num_batches = len(dataloader)
    progress_bar = tqdm(range(num_batches))
    data_iter = iter(dataloader)
    for _ in progress_bar:
        data = next(data_iter)
        input_data = {k: v[0].to(network.device) for k, v in data["input"].items()}
        # targets for the logit levels
        target_zero = data["target_zero"][0].to(network.device)
        target_one = data["target_one"][0].to(network.device)
        target_two = data["target_two"][0].to(network.device)

        logits_zero, logits_one, logits_two = network(input_data)

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

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if config.optimizer["gradient_clipping"] > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=config.optimizer["gradient_clipping"])

        optimizer.step()

        batch_loss = loss.item()
        accumulated_loss_lvl_zero += loss_zero.item()
        accumulated_loss_lvl_one += loss_one.item()
        accumulated_loss_lvl_two += loss_two.item()

        progress_bar.set_postfix({"batch_loss": batch_loss})

        del data

    metrics["loss_level_zero"] = accumulated_loss_lvl_zero / num_batches
    metrics["loss_level_one"] = accumulated_loss_lvl_one / num_batches
    metrics["loss_level_two"] = accumulated_loss_lvl_two / num_batches

    top_20_molecules = metrics["top_20_molecules"]
    del metrics["top_20_molecules"]
    return metrics, top_20_molecules


def evaluate(eval_type: str, config: MoleculeConfig, network: MoleculeTransformer, objective_evaluator: MoleculeObjectiveEvaluator):
    config = copy.deepcopy(config)
    config.gumbeldore_config["destination_path"] = None

    gumbeldore_dataset = GumbeldoreDataset(
        config=config, objective_evaluator=objective_evaluator
    )
    metrics = gumbeldore_dataset.generate_dataset(copy.deepcopy(network.get_weights()), memory_aggressive=False)
    top_20_mols = metrics["top_20_molecules"]
    metrics = {
        f"{eval_type}_mean_top_20_obj": metrics["mean_top_20_obj"],
        f"{eval_type}_mean_top_20_sa_score": metrics["mean_top_20_sa_score"],
        f"{eval_type}_best_obj": metrics['best_gen_obj'],
        f"{eval_type}_best_mol_sa_score": metrics['best_gen_sa_score'],
    }
    print("Evaluation done")
    print(f"Eval ({eval_type}) best obj: {metrics[f'{eval_type}_best_obj']:.3f}")
    print(f"Eval ({eval_type}) mean top 20 obj: {metrics[f'{eval_type}_mean_top_20_obj']:.3f}")

    return metrics, top_20_mols


if __name__ == '__main__':
    print(">> Molecule Design")

    parser = argparse.ArgumentParser(description='Experiment')
    parser.add_argument('--config', help="Path to optional config relative to main.py")
    args = parser.parse_args()

    if args.config is not None:
        # Load config from given path
        MoleculeConfig = importlib.import_module(args.config).MoleculeConfig
    config = MoleculeConfig()

    num_gpus = len(config.CUDA_VISIBLE_DEVICES.split(","))
    ray.init(num_gpus=num_gpus, logging_level="info")
    print(ray.available_resources())

    logger = Logger(args, config.results_path, config.log_to_file)
    logger.log_hyperparams(config)
    # Fix random number generator seed for better reproducibility
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Setup the neural network for training
    network = MoleculeTransformer(config, config.training_device)
    # Setup the GNN-based evaluator which predicts the objective function evaluation from a designed molecule.
    objective_evaluator = MoleculeObjectiveEvaluator(config, device=config.objective_gnn_device)

    # Load checkpoint if needed
    if config.load_checkpoint_from_path is not None:
        print(f"Loading checkpoint from path {config.load_checkpoint_from_path}")
        checkpoint = torch.load(config.load_checkpoint_from_path)
        print(f"{checkpoint['epochs_trained']} episodes have been trained in the loaded checkpoint.")
    else:
        checkpoint = {
            "model_weights": None,
            "best_model_weights": None,
            "optimizer_state": None,
            "epochs_trained": 0,
            "validation_metric": float("-inf"),   # objective of the best molecule designed during validation.
            "best_validation_metric": float("-inf")  # corresponding to best model weights
        }
    if checkpoint["model_weights"] is not None:
        network.load_state_dict(checkpoint["model_weights"])

    print(f"Policy network is on device {config.training_device}")
    network.to(network.device)
    network.eval()

    if config.num_epochs > 0:
        # Training loop
        print(f"Starting training for {config.num_epochs} epochs.")

        best_model_weights = checkpoint["best_model_weights"]  # can be None
        best_validation_metric = checkpoint["best_validation_metric"]

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
                    checkpoint["epochs_trained"] // config.optimizer["schedule"]["decay_lr_every_epochs"])
        scheduler = LambdaLR(optimizer, lr_lambda=_lambda)

        start_time_counter = None
        if config.wall_clock_limit is not None:
            print(f"Wall clock limit of training set to {config.wall_clock_limit / 3600} hours")
            start_time_counter = time.perf_counter()

        for epoch in range(config.num_epochs):
            print("------")
            print(f"Generating dataset.")
            network_weights = copy.deepcopy(network.get_weights())

            generated_loggable_dict, generated_text_to_save = train_for_one_epoch(
                epoch, config, network, network_weights, optimizer, objective_evaluator, best_validation_metric
            )

            checkpoint["epochs_trained"] += 1
            scheduler.step()
            print(f">> Epoch {checkpoint['epochs_trained']}. Avg loss level 0: {generated_loggable_dict['loss_level_zero']},"
                  f" Avg loss level 1: {generated_loggable_dict['loss_level_one']},"
                  f" Avg loss level 2: {generated_loggable_dict['loss_level_two']}")
            logger.log_metrics(generated_loggable_dict, step=epoch)
            # Save the top 20 molecules
            logger.text_artifact(os.path.join(config.results_path, f"epoch_{epoch + 1}_train_top_20_molecules.txt"), generated_text_to_save)

            # Save model
            checkpoint["model_weights"] = copy.deepcopy(network.get_weights())
            checkpoint["optimizer_state"] = copy.deepcopy(
                dict_to_cpu(optimizer.state_dict())
            )
            val_metric = generated_loggable_dict["best_gen_obj"]   # measure by best objective found during sampling
            checkpoint["validation_metric"] = val_metric
            save_checkpoint(checkpoint, "last_model.pt", config)

            if val_metric > best_validation_metric:
                print(">> Got new best model.")
                checkpoint["best_model_weights"] = copy.deepcopy(checkpoint["model_weights"])
                checkpoint["best_validation_metric"] = val_metric
                best_model_weights = checkpoint["best_model_weights"]
                best_validation_metric = val_metric
                save_checkpoint(checkpoint, "best_model.pt", config)

            if start_time_counter is not None and time.perf_counter() - start_time_counter > config.wall_clock_limit:
                print("Time exceeded. Stopping training.")
                break

    if config.num_epochs == 0:
        print(f"Testing with loaded model.")
    else:
        print(f"Testing with best model.")
        checkpoint = torch.load(os.path.join(config.results_path, "best_model.pt"))
        network.load_state_dict(checkpoint["model_weights"])

    if checkpoint["model_weights"] is None and config.num_epochs == 0:
        print("WARNING! No training was performed, but also no checkpoint to load was given. "
              "Evaluating with random model.")

    torch.cuda.empty_cache()
    with torch.no_grad():
        test_loggable_dict, test_text_to_save = evaluate('test', config, network, objective_evaluator)
    print(">> TEST")
    print(test_loggable_dict)
    logger.log_metrics(test_loggable_dict, step=0, step_desc="test")
    print(test_text_to_save)
    logger.text_artifact(os.path.join(config.results_path, "test_top_20_molecules.txt"),
                         test_text_to_save)

    print("Finished. Shutting down ray.")
    ray.shutdown()