import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer architecture for GRXForm.
    - L0 logits (Terminate, Select Anchor Atom) are dynamic based on batch atom count.
    - L1 logits (Add Atom, Select Existing Atom, Replace Atom, Remove Atom) are dynamic.
    - L2 logits (Set Bond Type) are fixed size.
    """

    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        # --- Vocabulary and Dimension Info ---
        try:
            bond_types_keys = MoleculeDesign.bond_types.keys()
        except AttributeError as err:
            raise err
        else:
            max_bond_actions = len(bond_types_keys) + 1  # +1 for "remove bond"

        self.vocab_size = len(self.config.atom_vocabulary)

        valid_valences = []
        for atom_data in self.config.atom_vocabulary.values():
            if "valence" in atom_data and atom_data["valence"] is not None and atom_data["valence"] >= 0:
                valid_valences.append(atom_data["valence"])

        max_possible_valence = max([0] + valid_valences) if valid_valences else 0
        degree_padding_idx = max_possible_valence + 1

        num_atom_embeddings = self.vocab_size + 2
        atom_padding_idx = self.vocab_size + 1

        virtual_bond_idx = MoleculeDesign.virtual_bond_idx
        bond_padding_idx = virtual_bond_idx + 1
        num_bond_embeddings = virtual_bond_idx + 2

        # --- Input Embeddings ---
        self.virtual_atom_level_embedding = nn.Embedding(3, self.latent_dim)  # Levels 0, 1, 2
        self.atom_learnable_embedding = nn.Embedding(num_atom_embeddings, self.latent_dim, padding_idx=atom_padding_idx)
        self.degree_learnable_embedding = nn.Embedding(max_possible_valence + 2, self.latent_dim,
                                                       padding_idx=degree_padding_idx)

        self.bond_learnable_embedding = nn.Embedding(num_bond_embeddings, self.num_blocks * self.num_heads,
                                                     padding_idx=bond_padding_idx)
        self.picked_atom_embedding = nn.Embedding(3, self.latent_dim)  # 0:not picked, 1:L0 anchor, 2:L1 target

        # --- Output Linear Layers ---
        # L0: Dynamic Size (B, N_batch)
        self.linear_l0_terminate = nn.Linear(self.latent_dim, 1)
        self.linear_l0_select_atom = nn.Linear(self.latent_dim, 1)

        # L1: Dynamic Size (2*V + N_batch)
        l1_virtual_output_size = self.vocab_size + self.vocab_size + 1  # Add(V), Replace(V), Remove(1)
        self.linear_l1_virtual_add_replace_remove = nn.Linear(self.latent_dim, l1_virtual_output_size)
        self.linear_l1_select_existing = nn.Linear(self.latent_dim, 1)  # Select(N_batch - 1)

        # L2: Fixed Size (Set/Remove Bond)
        self.output_linear_level_two = nn.Linear(self.latent_dim, max_bond_actions)

        # --- Transformer Encoder ---
        self.encoder = nn.ModuleList([])
        for _ in range(config.num_transformer_blocks):
            if not config.use_rezero_transformer:
                block = TransformerEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True
                )
            else:
                block = RZTXEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim, dropout=config.dropout,
                    activation="gelu", batch_first=True
                )
            self.encoder.append(block)

    def forward(self, x: dict):
        batch_size, num_atoms_in_batch = x["atoms"].shape

        # --- 1. Construct Initial Atom Features ---
        atom_sequence = self.atom_learnable_embedding(x["atoms"])
        if num_atoms_in_batch > 1:
            degree_embeddings = self.degree_learnable_embedding(x["atoms_degree"][:, 1:])
            atom_sequence[:, 1:] = atom_sequence[:, 1:] + degree_embeddings

        level_embedding = self.virtual_atom_level_embedding(x["level_idx"])
        atom_sequence[:, 0] = atom_sequence[:, 0] + level_embedding

        picked_embedding = self.picked_atom_embedding(x["picked_atom_mhe"])
        atom_sequence = atom_sequence + picked_embedding

        # --- 2. Prepare Attention Masks ---
        attn_mask_bias = self.bond_learnable_embedding(x["bonds"])
        attn_mask_bias = torch.permute(attn_mask_bias, (0, 3, 1, 2))
        attn_mask_bias = attn_mask_bias.view(batch_size, self.num_blocks, self.num_heads, num_atoms_in_batch,
                                             num_atoms_in_batch)

        padding_attn_mask = x["additive_padding_attn_mask"].unsqueeze(1).unsqueeze(2)

        # --- 3. Process through Transformer Encoder ---
        current_src = atom_sequence
        for i, trf_block in enumerate(self.encoder):
            block_attn_bias = attn_mask_bias[:, i, :, :, :]
            current_block_mask = block_attn_bias + padding_attn_mask.squeeze(1)
            mask_for_block_folded = current_block_mask.reshape(batch_size * self.num_heads, num_atoms_in_batch,
                                                               num_atoms_in_batch)
            current_src = trf_block(current_src, src_mask=mask_for_block_folded)
        atom_sequence = current_src

        # --- 4. Generate Logits ---
        virtual_atom_state = atom_sequence[:, 0, :]

        if num_atoms_in_batch > 1:
            real_atom_states = atom_sequence[:, 1:, :]
        else:
            real_atom_states = torch.empty((batch_size, 0, self.latent_dim), dtype=virtual_atom_state.dtype,
                                           device=virtual_atom_state.device)

        # L0 Logits
        logits_l0_terminate = self.linear_l0_terminate(virtual_atom_state)
        if num_atoms_in_batch > 1:
            logits_l0_select = self.linear_l0_select_atom(real_atom_states).squeeze(-1)
            logits_zero = torch.cat((logits_l0_terminate, logits_l0_select), dim=1)
        else:
            logits_zero = logits_l0_terminate

            # L1 Logits
        virtual_l1_logits_combined = self.linear_l1_virtual_add_replace_remove(virtual_atom_state)
        logits_l1_add = virtual_l1_logits_combined[:, :self.vocab_size]
        logits_l1_replace = virtual_l1_logits_combined[:, self.vocab_size: 2 * self.vocab_size]
        logit_l1_remove = virtual_l1_logits_combined[:, 2 * self.vocab_size:]

        if num_atoms_in_batch > 1:
            logits_l1_select_existing = self.linear_l1_select_existing(real_atom_states).squeeze(-1)
        else:
            logits_l1_select_existing = torch.empty((batch_size, 0), dtype=virtual_atom_state.dtype,
                                                    device=virtual_atom_state.device)

        # Output Order Must Perfectly Match Mask Order: [Add(V) | SelectExisting(N-1) | Replace(V) | Remove(1)]
        logits_one = torch.cat(
            (logits_l1_add, logits_l1_select_existing, logits_l1_replace, logit_l1_remove),
            dim=1
        )

        # L2 Logits
        logits_two = self.output_linear_level_two(virtual_atom_state)

        return logits_zero, logits_one, logits_two

    def get_weights(self):
        """Returns the model's state dict with tensors moved to CPU."""
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary: dict) -> dict:
    """Recursively moves all tensors in a dictionary (and its sub-dictionaries) to CPU."""
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict