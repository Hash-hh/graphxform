# **AMORTIX: Amortized Molecular Optimization via Group Relative Policy Optimization**

AMORTIX is a framework for amortized molecular optimization. It adapts a pre-trained Graph Transformer model to optimize molecules via sequential atom-and-bond additions. To address the high variance arising from the heterogeneous difficulty of distinct starting structures, AMORTIX employs Group Relative Policy Optimization (GRPO) for goal-directed fine-tuning.

This approach normalizes rewards relative to the starting structure, stabilizing the learning process. The resulting policy generalizes to out-of-distribution molecular scaffolds and generates optimized molecules in a single forward pass without requiring inference-time oracle calls or iterative refinement.

## **Installation**

Install the dependencies found in requirements.txt:

`pip install -r requirements.txt`

Additionally, install **[torch_scatter](https://github.com/rusty1s/pytorch_scatter)** with options corresponding to your hardware.

Create the required directories:

`mkdir -p results data/chembl/pretrain_sequences`

## Pretraining

You can pretrain youself or use the provided pretrained checkpoint.

### Pretrained checkpoint download

Download the pretrained checkpoint [here](https://drive.google.com/file/d/14HFad1ZQHlbU33J6D87uVWeNXcQRWAzl/view?usp=sharing).

You can use this checkpoint for fine-tuning on downstream tasks by specifying the path to the checkpoint in `config.py` (in variable `load_checkpoint_from_path`).


### Pretrain yourself

To pretrain with the settings in the paper, do the following: 

1. Download the file `chembl_35_chemreps.txt.gz` from this this link: [Download File](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_35/chembl_35_chemreps.txt.gz)
and put the extracted `chembl_35_chemreps.txt` under the `./data/chembl` directory.
2. Run `$ python filter_molecules.py`. This will perform a train/val split with 100k validation molecules, which will be
put under `data/chembl/chembl_{train/valid}.smiles`. Then, depending on the allowed vocabulary (see `filter_molecules.py`),
it filters the smiles for all strings containing the allowed sets of characters and save them under `data/chembl/chembl_{train/valid}_filtered.smiles`.
3. Run `$ python create_pretrain_dataset.py`. This will convert the filtered SMILES to instances of `MoleculeDesign` (the class in `molecule_design.py`, which takes the role of the molecular graph environment).
The pickled molecules are saved in `./data/chembl/pretrain_sequences/chembl_{train/valid}.pickle`.
4. Run `$ python pretrain.py` to perform pretraining of the model. The general config to use (e.g., architecture) is under `config.py`.
In `pretrain.py`, you can specify pretrain-specific options directly at the entrypoint to adjust to your hardware, i.e.:
```python
if __name__ == '__main__':
    pretrain_train_dataset = "./data/chembl/pretrain_sequences/chembl_train.pickle"
    pretrain_val_dataset = "./data/chembl/pretrain_sequences/chembl_valid.pickle"
    pretrain_num_epochs = 1000   # For how many epochs to train
    batch_size = 512  # Minibatch size
    num_batches_per_epoch = 3000   # Number of minibatches per epoch.
    batch_size_validation = 512  # Batch size during validation
    training_device = "cuda:0"  # Device on which to train. Set to "cpu" if no CUDA available.
    num_dataloader_workers = 10  # Number of dataloader workers for creating batches for training
    load_checkpoint_from_path = None   # Path to checkpoint if training should be continued from existing weights.
```
4. The terminal output will show under which subdirectory (named after timestamp) in `./results` the script will save the model checkpoints.

5. After pretraining, the best model checkpoint (lowest validation loss) will be saved as `best_model.pt` in the corresponding subdirectory in `./results`. You can use this checkpoint for fine-tuning on downstream tasks by specifying the path to the checkpoint in `config.py` (in variable `load_checkpoint_from_path`).

## **Data Preparation**

Before running scaffold-constrained tasks, generate the cluster-based scaffold splits from the ZINC dataset to evaluate generalization to out-of-distribution scaffolds.

`python scaffold_splitting/zinc_splits/smart_splitting.py`

## **Usage and Configuration**

The framework is controlled via configuration files and command-line arguments. The primary configuration is located in config.py, where you can define the objective task, model hyperparameters, and enable reinforcement learning.

Key variables in `config.py`:

* `objective_type`: Sets the target objective (e.g., `"kinase_mpo"`, `"prodrug_bbb"`).
* `use_dr_grpo`: Set to True to enable GRPO reinforcement learning. Set to False to run supervised fine-tuning.  
* `use_fragment_library`: Master switch for structural conditioning (scaffold-based generation).

### **Running an Experiment**

Run the training script using the default configuration:

`python main.py`

To run a specific experiment configuration module from an Experiments/ directory, use the \--config argument:

`python main.py --config Experiments.case2_scaffold_kinase`

You can also override specific RL hyperparameters directly via the command line:

`python main.py --learning_rate 1e-4 --rl_entropy_beta 0.001 --ppo_epochs 1 --rl_ppo_clip_epsilon 0.2`

## **Supported Evaluation Tasks**

The codebase natively supports the objective functions discussed in the paper:

1. **Kinase Scaffold Decoration (Kinase MPO):** Multi-objective optimization targeting GSK3B and JNK3 activity, QED, and Synthetic Accessibility (SA).  
2. **Prodrug Transfer:** Structural transformation targeting blood-brain barrier permeability (optimizing for Delta LogP, HBD masking, cleavability, and QED).  
3. **PMO Benchmark:** De-novo generation tasks utilizing the standard PMO evaluation suite. **Note: To reproduce the PMO benchmark results, please use the `pmo` branch of this repository.**

## **Output**

Results, model checkpoints, and logs are saved to the `results` directory by default:

* `best_model.pt` and `last_model.pt` weights.  
* CSV logs generated during testing, containing starting SMILES, generated SMILES, and objective scores.
* Text files logging the top generated molecules per epoch.
