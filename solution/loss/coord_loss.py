"""
L_coord — continuous coordinate L1 loss.

Penalises the mean absolute error between predicted and GT (x, y) positions.

Excluded from the loss:
  - Preplaced blocks  (position is hard-constrained, no need to learn)
  - Padding tokens    (key_padding_mask == True)
"""

import torch
import torch.nn.functional as F


def coord_loss(
    pred_positions_norm: torch.Tensor,  # [B, k, 4]  (x, y, w, h), normalised
    gt_positions_norm:   torch.Tensor,  # [B, k, 4]
    constraints:         torch.Tensor,  # [B, k, 5]
    key_padding_mask:    torch.Tensor,  # [B, k] bool  (True = padding)
) -> torch.Tensor:
    """
    Returns scalar: mean L1 error on (x, y) over valid (non-preplaced, non-padding) blocks.
    """
    pred_xy = pred_positions_norm[..., :2]   # [B, k, 2]
    gt_xy   = gt_positions_norm[..., :2]     # [B, k, 2]

    is_preplaced = constraints[..., 1] > 0   # [B, k]
    valid        = ~(is_preplaced | key_padding_mask)   # [B, k]

    if valid.sum() == 0:
        return pred_positions_norm.sum() * 0.0   # differentiable zero

    return F.l1_loss(pred_xy[valid], gt_xy[valid])


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("coord_loss.py  —  smoke test (L1)")
    print("=" * 55)

    B, k = 2, 10

    pred_norm = torch.rand(B, k, 4, requires_grad=True)
    gt_norm   = torch.rand(B, k, 4)

    constraints = torch.zeros(B, k, 5)
    constraints[:, :2, 1] = 1.0   # blocks 0-1: preplaced (excluded)

    kpm = torch.zeros(B, k, dtype=torch.bool)
    kpm[:, -1] = True              # last token is padding (excluded)

    loss = coord_loss(pred_norm, gt_norm, constraints, kpm)

    print(f"\npred_norm shape  : {tuple(pred_norm.shape)}")
    print(f"gt_norm shape    : {tuple(gt_norm.shape)}")
    print(f"Loss value       : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"
    assert loss.item() >= 0, "L1 must be non-negative"

    loss.backward()
    assert pred_norm.grad is not None, "No gradient to pred_norm"
    print(f"Gradient norm    : {pred_norm.grad.norm().item():.6f}")

    # Perfect prediction → loss = 0
    zero_loss = coord_loss(
        gt_norm.detach().requires_grad_(True), gt_norm, constraints, kpm
    )
    print(f"Perfect pred loss: {zero_loss.item():.2e}  (expected 0)")
    assert zero_loss.item() < 1e-6

    # All excluded → differentiable zero
    kpm_all   = torch.ones(B, k, dtype=torch.bool)
    zero_loss2 = coord_loss(pred_norm.detach(), gt_norm, constraints, kpm_all)
    print(f"All-excluded loss: {zero_loss2.item():.6f}  (expected 0)")
    assert zero_loss2.item() == 0.0

    print("\nSmoke test passed.")
    print("=" * 55)
