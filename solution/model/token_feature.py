"""
Token Feature Projection.

Projects the 18-dim raw block feature vector into d_model-dim embeddings
that serve as Encoder input tokens.

Architecture:
    raw_feature [k, 18]
        → Linear(18, d_model)
        → LayerNorm(d_model)
        → Dropout
    output [k, d_model]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import D_MODEL, RAW_FEATURE_DIM, DROPOUT


class TokenFeatureProjection(nn.Module):
    """
    Linear projection from raw block features to Encoder token embeddings.

    Args:
        raw_dim  (int): Input feature dimension. Default: RAW_FEATURE_DIM (18).
        d_model  (int): Output embedding dimension. Default: D_MODEL (256).
        dropout  (float): Dropout probability applied after projection. Default: DROPOUT.
    """

    def __init__(
        self,
        raw_dim: int = RAW_FEATURE_DIM,
        d_model: int = D_MODEL,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.projection = nn.Linear(raw_dim, d_model)
        self.norm       = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, k, raw_dim]  raw token features

        Returns:
            [batch, k, d_model]  projected token embeddings
        """
        return self.dropout(self.norm(self.projection(x)))


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("token_feature.py  —  smoke test")
    print("=" * 50)

    batch, k = 4, 55
    x = torch.randn(batch, k, RAW_FEATURE_DIM)

    model = TokenFeatureProjection()
    model.eval()

    with torch.no_grad():
        out = model(x)

    print(f"\nInput  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    assert out.shape == (batch, k, D_MODEL), \
        f"Expected ({batch}, {k}, {D_MODEL}), got {tuple(out.shape)}"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {total_params:,}  "
          f"(Linear {RAW_FEATURE_DIM}×{D_MODEL} + bias + LayerNorm)")

    print("\nSmoke test passed.")
    print("=" * 50)
