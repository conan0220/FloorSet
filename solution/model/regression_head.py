"""
Regression Head.

Maps Decoder hidden states [B, k, d_model] to block placements [B, k, 4].

Architecture:
    Linear(d_model, 64) → ReLU → Linear(64, 4)

Output is raw regression values (no activation); the training loss and
post-processing handle any clamping or constraint enforcement.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import D_MODEL


class RegressionHead(nn.Module):
    """
    Two-layer MLP that maps decoder embeddings to (x, y, w, h).

    Args:
        d_model  (int): Input dimension. Default: D_MODEL (256).
        hidden   (int): Hidden layer dimension. Default: 64.
        out_dim  (int): Output dimension. Default: 4  [(x, y, w, h)].
    """

    def __init__(self, d_model: int = D_MODEL, hidden: int = 64, out_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, k, d_model]  decoder hidden states

        Returns:
            [B, k, 4]  predicted (x, y, w, h), normalised scale
        """
        return self.net(x)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 45)
    print("regression_head.py  —  smoke test")
    print("=" * 45)

    B, k = 2, 55
    x = torch.randn(B, k, D_MODEL)

    head = RegressionHead()
    head.eval()

    with torch.no_grad():
        out = head(x)

    print(f"\nInput  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    assert out.shape == (B, k, 4), \
        f"Expected ({B}, {k}, 4), got {tuple(out.shape)}"

    total = sum(p.numel() for p in head.parameters())
    print(f"Parameters   : {total:,}  (Linear {D_MODEL}×64 + Linear 64×4 + biases)")

    print("\nSmoke test passed.")
    print("=" * 45)
