import torch
from torch import nn


class FiLMGenerator(nn.Module):
    """
    Produces per-block (gamma, beta) feature-wise linear modulation parameters
    from a conditioning vector c. Zero-init of the final projection makes gamma=1,
    beta=0 at init, so the FiLM-augmented model is numerically identical to the
    baseline at initialization (safe warm-start from an existing checkpoint).

    Output shapes (per forward): gamma, beta each (B, num_blocks, d_model).
    """

    def __init__(self, cond_dim: int, d_model: int, num_blocks: int, hidden: int = 256):
        super().__init__()
        self.cond_dim = cond_dim
        self.d_model = d_model
        self.num_blocks = num_blocks

        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.head = nn.Linear(hidden, num_blocks * 2 * d_model)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, c: torch.Tensor):
        h = self.head(self.trunk(c))                                # (B, num_blocks*2*d_model)
        h = h.view(-1, self.num_blocks, 2, self.d_model)            # (B, num_blocks, 2, d_model)
        gamma = 1.0 + h[:, :, 0, :]
        beta = h[:, :, 1, :]
        return gamma, beta
