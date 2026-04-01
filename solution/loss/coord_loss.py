"""
L_coord — per-block coordinate regression loss.

For each block i:
    loss_i = (x̂-x)² + (ŷ-y)² + (ŵ-w)² + (ĥ-h)²

Preplaced blocks are up-weighted by `preplaced_weight` (default 5.0).
Returns the weighted average across all blocks and all batch samples.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import PREPLACED_COORD_WEIGHT


def coord_loss(
    pred_positions:   torch.Tensor,                    # [B, k, 4]
    gt_positions:     torch.Tensor,                    # [B, k, 4]
    constraints:      torch.Tensor,                    # [B, k, 5]
    preplaced_weight: float = PREPLACED_COORD_WEIGHT,  # default 5.0
) -> torch.Tensor:
    """
    Weighted coordinate MSE loss.

    Args:
        pred_positions   [B, k, 4]  predicted (x, y, w, h), normalised
        gt_positions     [B, k, 4]  ground truth (x, y, w, h), normalised
        constraints      [B, k, 5]  placement constraints; col 1 = is_preplaced
        preplaced_weight float      extra weight for preplaced blocks

    Returns:
        scalar loss (weighted mean over all blocks × batch)
    """
    # Per-block squared error: [B, k]
    per_block = ((pred_positions - gt_positions) ** 2).sum(dim=-1)

    # Build per-block weight tensor
    is_preplaced = constraints[:, :, 1].float()           # [B, k]
    weights = 1.0 + (preplaced_weight - 1.0) * is_preplaced   # [B, k]

    # Weighted average
    return (weights * per_block).sum() / weights.sum()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 45)
    print("coord_loss.py  —  smoke test")
    print("=" * 45)

    B, k = 2, 10
    pred = torch.rand(B, k, 4, requires_grad=True)
    gt   = torch.rand(B, k, 4)

    # Mark first 2 blocks of each sample as preplaced
    constraints = torch.zeros(B, k, 5)
    constraints[:, :2, 1] = 1.0

    loss = coord_loss(pred, gt, constraints)

    print(f"\nInput  pred shape  : {tuple(pred.shape)}")
    print(f"Input  gt   shape  : {tuple(gt.shape)}")
    print(f"Loss value         : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar output"
    assert loss.item() >= 0, "Loss must be non-negative"

    loss.backward()
    assert pred.grad is not None, "No gradient to pred_positions"
    print(f"Gradient norm      : {pred.grad.norm().item():.6f}")

    # Perfect prediction → loss should be 0
    zero_loss = coord_loss(gt, gt, constraints)
    assert zero_loss.item() < 1e-6, f"Expected ~0, got {zero_loss.item()}"
    print(f"Perfect pred loss  : {zero_loss.item():.2e}  (expected ~0)")

    print("\nSmoke test passed.")
    print("=" * 45)
