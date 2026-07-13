import torch
from torch import nn
from torch.nn.modules import TransformerEncoderLayer
from model.rztx import RZTXEncoderLayer
from config import MoleculeConfig
from molecule_design import MoleculeDesign


class MoleculeTransformer(nn.Module):
    """
    Molecular Transformer for molecular design.

    Supports two action-space modes:

    **Atomic mode (legacy):**
        - L0: terminate + atom types (virtual) | select existing atom (per-atom)
        - L1: select second atom (per-atom)
        - L2: bond order (virtual)

    **Fragment mode (AMORTIX 2.0):**
        - L0: terminate + fragments + open sites (all from virtual atom)
        - L1: fragment site or second scaffold site (all from virtual atom)
        - L2: scaffold site or bond order (all from virtual atom)

    In fragment mode, ALL logits are produced by the virtual atom because
    actions don't map 1:1 to atoms (an atom can have multiple open sites,
    and fragment sites are properties of the fragment, not the scaffold).
    The virtual atom attends to all atoms via MHSA and aggregates the
    necessary context.
    """

    def __init__(self, config: MoleculeConfig, device: torch.device = None):
        super().__init__()
        self.config = config
        self.device = torch.device("cpu") if device is None else device
        self.latent_dim = self.config.latent_dimension
        self.num_heads = self.config.num_heads
        self.num_blocks = self.config.num_transformer_blocks

        # ── Mode flag ──────────────────────────────────────────────
        self._is_fragment_mode = getattr(
            config, 'use_fragment_action_space', True
        )

        # ── Vocabulary sizes ───────────────────────────────────────
        max_possible_valence = max([
            self.config.atom_vocabulary[x]["valence"]
            for x in self.config.atom_vocabulary
        ])
        self.num_possible_atom_types = len(self.config.atom_vocabulary) + 1
        self.num_possible_bonds = MoleculeDesign.maximum_bond_order

        # ── Embeddings (shared between modes) ──────────────────────
        self.virtual_atom_level_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.latent_dim
        )
        self.atom_learnable_embedding = nn.Embedding(
            num_embeddings=self.num_possible_atom_types + 1,
            embedding_dim=self.latent_dim,
            padding_idx=self.num_possible_atom_types
        )
        self.degree_learnable_embedding = nn.Embedding(
            num_embeddings=max_possible_valence + 2,
            embedding_dim=self.latent_dim,
            padding_idx=max_possible_valence + 1
        )

        # FIX 1: Bond embedding now accommodates aromatic bonds (idx 8)
        # and padding (idx 9).  Previously only had indices 0–8, but
        # padded bonds use index 9 (= aromatic_bond_idx + 1).
        self.bond_learnable_embedding = nn.Embedding(
            num_embeddings=MoleculeDesign.aromatic_bond_idx + 2,  # = 10
            embedding_dim=self.num_blocks * self.num_heads,
            padding_idx=MoleculeDesign.aromatic_bond_idx + 1       # = 9
        )

        # picked_atom_embedding: 0=nothing, 1=picked at L0, 2=picked at L1
        # Same encoding in both modes.
        self.picked_atom_embedding = nn.Embedding(
            num_embeddings=3, embedding_dim=self.latent_dim, padding_idx=0
        )

        # FIX 3: open_site_embedding — tells the model which atoms have
        # open attachment sites (dummy neighbors).  Only applied in
        # fragment mode, but always created for checkpoint compatibility.
        # padding_idx=0 → atoms without open sites get zero embedding.
        self.open_site_embedding = nn.Embedding(
            num_embeddings=2, embedding_dim=self.latent_dim, padding_idx=0
        )

        # ── Output heads (mode-dependent) ──────────────────────────
        if self._is_fragment_mode:
            # ── Fragment mode ──────────────────────────────────────
            # ALL logits come from the virtual atom because actions
            # don't map 1:1 to atoms.
            K = len(config.fragment_vocabulary) \
                if config.fragment_vocabulary else 0
            S_max = config.max_open_attachment_sites
            D_max = (
                max(f.num_attachment_sites for f in config.fragment_vocabulary)
                if config.fragment_vocabulary else 2
            )

            # L0: terminate(1) + fragments(K) + open sites(S_max)
            self.l0_size = 1 + K + S_max
            # L1: max(D_max, S_max) — fragment sites or scaffold sites
            self.l1_size = max(D_max, S_max)
            # L2: max(S_max, 3) — scaffold sites or bond orders
            self.l2_size = max(S_max, 3)

            self.virtual_atom_linear = nn.Linear(
                self.latent_dim, self.l0_size + self.l1_size + self.l2_size
            )

            # bond_atom_linear is not used in fragment mode, but created
            # for checkpoint loading compatibility.
            self.bond_atom_linear = nn.Linear(self.latent_dim, 2)
        else:
            # ── Atomic mode (unchanged) ───────────────────────────
            self.virtual_atom_linear = nn.Linear(
                self.latent_dim,
                self.num_possible_atom_types + self.num_possible_bonds
            )
            self.bond_atom_linear = nn.Linear(self.latent_dim, 2)

        # ── Transformer encoder (unchanged) ───────────────────────
        self.encoder = nn.ModuleList([])
        for _ in range(config.num_transformer_blocks):
            if not config.use_rezero_transformer:
                block = TransformerEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim,
                    dropout=config.dropout,
                    activation="gelu", batch_first=True, norm_first=True
                )
            else:
                block = RZTXEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads,
                    dim_feedforward=4 * self.latent_dim,
                    dropout=config.dropout,
                    activation="gelu", batch_first=True
                )
            self.encoder.append(block)

    def forward(self, x: dict):
        """
        Forward pass producing logits for all 3 action levels.

        In fragment mode, all logits come from the virtual atom.
        In atomic mode, L0/L1 use per-atom logits (original behavior).

        The caller (``log_probability_fn``) trims logits to the current
        mask size, so producing S_max logits when only S are needed is
        safe.
        """
        batch_size, num_atoms = x["atoms"].shape

        # ════════════════════════════════════════════════════════════
        # 1. Input embeddings
        # ════════════════════════════════════════════════════════════
        atom_sequence = self.atom_learnable_embedding(x["atoms"])
            # (B, num_atoms, latent_dim)

        # Add degree embedding to real atoms (not virtual)
        atom_sequence[:, 1:] = atom_sequence[:, 1:] + \
            self.degree_learnable_embedding(x["atoms_degree"][:, 1:])

        # Add level index embedding to virtual atom
        atom_sequence[:, 0] = atom_sequence[:, 0] + \
            self.virtual_atom_level_embedding(x["level_idx"])

        # Add picked-atom indicator (0=nothing, 1=L0 pick, 2=L1 pick)
        atom_sequence = atom_sequence + \
            self.picked_atom_embedding(x["picked_atom_mhe"])

        # FIX 3: In fragment mode, add open-site indicator
        # This tells the model which atoms have open attachment sites
        # (dummy neighbors).  Atoms without open sites get zero embedding
        # (padding_idx=0).
        if self._is_fragment_mode:
            atom_sequence = atom_sequence + \
                self.open_site_embedding(x["open_sites_mask"])

        # ════════════════════════════════════════════════════════════
        # 2. Attention mask preparation (unchanged)
        # ════════════════════════════════════════════════════════════
        attn_mask = self.bond_learnable_embedding(x["bonds"])
            # (B, num_atoms, num_atoms, num_blocks * num_heads)
        attn_mask = torch.permute(
            attn_mask, (0, 3, 1, 2)
        ).view(
            batch_size, self.num_blocks, self.num_heads, num_atoms, num_atoms
        )

        padding_attn_mask = x["additive_padding_attn_mask"][:, None, :, :] \
            .repeat((1, self.num_blocks * self.num_heads, 1, 1))
        padding_attn_mask = padding_attn_mask.view(
            batch_size, self.num_blocks, self.num_heads, num_atoms, num_atoms
        )
        attn_mask = attn_mask + padding_attn_mask

        # ════════════════════════════════════════════════════════════
        # 3. Transformer encoder (unchanged)
        # ════════════════════════════════════════════════════════════
        for i, trf_block in enumerate(self.encoder):
            mask_block_folded = attn_mask[:, i, :, :, :] \
                .reshape(batch_size * self.num_heads, num_atoms, num_atoms)
            atom_sequence = trf_block(
                atom_sequence, src_mask=mask_block_folded
            )

        # ════════════════════════════════════════════════════════════
        # 4. Output logits (mode-dependent)
        # ════════════════════════════════════════════════════════════
        virtual_atom = atom_sequence[:, 0, :]  # (B, latent_dim)

        if self._is_fragment_mode:
            # ── Fragment mode: all logits from virtual atom ───────
            #
            # The virtual atom attends to all real atoms via MHSA and
            # aggregates the necessary context:
            #   - open_sites_mask → which atoms have open sites
            #   - picked_atom_mhe → which atoms/sites were picked
            #   - bonds → molecular topology
            #
            # L0 logits (1+K+S_max):
            #   [0]         = terminate
            #   [1..K]      = add fragment 0..K-1
            #   [K+1..K+S]  = pick open site 0..S-1
            #   [K+S+1..K+S_max] = padding (masked out by caller)
            #
            # L1 logits (max(D_max, S_max)):
            #   Case 1A (fragment at L0): [0..D-1] = fragment sites
            #   Case 1B (site at L0):     [0..S-1] = scaffold sites
            #
            # L2 logits (max(S_max, 3)):
            #   Case 2A (fragment): [0..S_before-1] = scaffold sites
            #   Case 2B (site-site): [0..2] = bond orders (1/2/3)
            #     or [0] = deterministic bond order

            all_logits = self.virtual_atom_linear(virtual_atom)
                # (B, l0_size + l1_size + l2_size)

            level_zero_logits = all_logits[:, :self.l0_size]
            level_one_logits = all_logits[:, self.l0_size:
                                          self.l0_size + self.l1_size]
            level_two_logits = all_logits[:, self.l0_size + self.l1_size:]

        else:
            # ── Atomic mode: original behavior ────────────────────
            #
            # L0 = [virtual L0 (terminate + atom types)]
            #    + [per-atom L0 (1 logit per atom: "select existing")]
            #
            # L1 = [per-atom L1 (1 logit per atom: "select second")]
            #
            # L2 = [virtual L2 (bond orders 1–6)]

            virtual_level_zero_and_two_logits = \
                self.virtual_atom_linear(virtual_atom)
                # (B, num_possible_atom_types + num_possible_bonds)

            virtual_level_zero_logits = \
                virtual_level_zero_and_two_logits[:, :-self.num_possible_bonds]
            level_two_logits = \
                virtual_level_zero_and_two_logits[:, -self.num_possible_bonds:]

            atom_level_zero_and_one_logits = \
                self.bond_atom_linear(atom_sequence[:, 1:, :])
                # (B, num_atoms - 1, 2)

            level_zero_logits = torch.concatenate(
                (virtual_level_zero_logits,
                 atom_level_zero_and_one_logits[:, :, 0]),
                dim=1
            )
            level_one_logits = atom_level_zero_and_one_logits[:, :, 1]

        return level_zero_logits, level_one_logits, level_two_logits

    def get_weights(self):
        return dict_to_cpu(self.state_dict())


def dict_to_cpu(dictionary):
    cpu_dict = {}
    for key, value in dictionary.items():
        if isinstance(value, torch.Tensor):
            cpu_dict[key] = value.cpu()
        elif isinstance(value, dict):
            cpu_dict[key] = dict_to_cpu(value)
        else:
            cpu_dict[key] = value
    return cpu_dict