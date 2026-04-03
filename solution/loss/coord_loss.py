"""
L_coord — grid-based coordinate cross-entropy loss.

Converts GT (x, y) positions to grid indices (cell = floor(pos * G)),
then computes cross-entropy against the predicted logits over G×G cells.

Excluded from the loss:
  - Preplaced blocks  (position is hard-constrained, no need to learn)
  - Padding tokens    (key_padding_mask == True)
"""

import sys
from pathlib import Path

import math

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import GRID_SIZE


def coord_loss(
    logits:           torch.Tensor,   # [B, k, G*G]
    gt_positions_norm: torch.Tensor,  # [B, k, 4]  (x, y, w, h), normalised
    constraints:      torch.Tensor,   # [B, k, 5]
    key_padding_mask: torch.Tensor,   # [B, k] bool  (True = padding)
) -> torch.Tensor:
    """
    Args:
        logits            [B, k, G*G]  raw logits from DiscreteRegressionHead
        gt_positions_norm [B, k, 4]   ground-truth (x, y, w, h), in [0, 1]
        constraints       [B, k, 5]   col 1 = is_preplaced
        key_padding_mask  [B, k]      True where the token is padding

    Returns:
        scalar cross-entropy loss (mean over valid tokens)
    """
    B, k, G2 = logits.shape
    G = int(round(G2 ** 0.5))

    # GT position → grid index  (row-major: index = gy * G + gx)
    gt_x = gt_positions_norm[:, :, 0].clamp(0.0, 1.0 - 1e-6)   # [B, k]
    gt_y = gt_positions_norm[:, :, 1].clamp(0.0, 1.0 - 1e-6)   # [B, k]
    gx   = (gt_x * G).long().clamp(0, G - 1)                    # [B, k]
    gy   = (gt_y * G).long().clamp(0, G - 1)                    # [B, k]
    gt_index = (gy * G + gx)                                     # [B, k]

    # Build validity mask: exclude preplaced and padding tokens
    is_preplaced = constraints[:, :, 1] > 0   # [B, k]
    valid        = ~(is_preplaced | key_padding_mask)             # [B, k]

    if valid.sum() == 0:
        return logits.sum() * 0.0   # differentiable zero

    valid_flat  = valid.view(B * k)               # [B*k]
    logits_flat = logits.view(B * k, G2)          # [B*k, G*G]
    gt_flat     = gt_index.view(B * k)            # [B*k]

    # Normalise by maximum possible cross-entropy (uniform distribution over G*G cells)
    # so the loss is in [0, 1] regardless of grid size.
    ce = F.cross_entropy(logits_flat[valid_flat], gt_flat[valid_flat])
    return ce / math.log(G * G)


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("coord_loss.py  —  smoke test (cross-entropy)")
    print("=" * 55)

    G = GRID_SIZE
    B, k = 2, 10

    logits = torch.randn(B, k, G * G, requires_grad=True)
    gt     = torch.rand(B, k, 4)   # normalised (x, y, w, h)

    constraints = torch.zeros(B, k, 5)
    constraints[:, :2, 1] = 1.0   # blocks 0-1 preplaced (excluded)

    kpm = torch.zeros(B, k, dtype=torch.bool)
    kpm[:, -1] = True              # last token is padding (excluded)

    loss = coord_loss(logits, gt, constraints, kpm)

    print(f"\nlogits shape       : {tuple(logits.shape)}")
    print(f"gt shape           : {tuple(gt.shape)}")
    print(f"Loss value         : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"
    assert loss.item() >= 0, "Cross-entropy must be non-negative"

    loss.backward()
    assert logits.grad is not None, "No gradient to logits"
    print(f"Gradient norm      : {logits.grad.norm().item():.6f}")

    # All-preplaced + padding → dummy zero loss
    kpm_all = torch.ones(B, k, dtype=torch.bool)
    zero_loss = coord_loss(logits.detach(), gt, constraints, kpm_all)
    print(f"All-excluded loss  : {zero_loss.item():.6f}  (expected 0)")
    assert zero_loss.item() == 0.0

    print("\nSmoke test passed.")
    print("=" * 55)
