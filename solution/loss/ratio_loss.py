"""
L_ratio — aspect ratio supervision loss.

Penalises the squared error between predicted and GT log aspect ratios:

    L_ratio = mean over valid blocks of ( log(w_pred/h_pred) - log(w_gt/h_gt) )^2

Using log space ensures the loss is symmetric for reciprocal ratios
(e.g. 2:1 and 1:2 are penalised equally relative to 1:1).

Excluded from the loss:
  - Fixed-shape and preplaced blocks  (w, h are hard-constrained by architecture)
  - Padding tokens
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def ratio_loss(
    pred_positions_norm: torch.Tensor,  # [B, k, 4]  (x, y, w, h), normalised
    gt_positions_norm:   torch.Tensor,  # [B, k, 4]
    constraints:         torch.Tensor,  # [B, k, 5]
    key_padding_mask:    torch.Tensor,  # [B, k] bool  (True = padding)
) -> torch.Tensor:
    """
    Returns scalar: mean squared log-ratio error over valid soft blocks.
    """
    pred_w = pred_positions_norm[:, :, 2]   # [B, k]
    pred_h = pred_positions_norm[:, :, 3]   # [B, k]
    gt_w   = gt_positions_norm[:, :, 2]     # [B, k]
    gt_h   = gt_positions_norm[:, :, 3]     # [B, k]

    pred_log_ratio = torch.log(pred_w.clamp(min=1e-8)) - torch.log(pred_h.clamp(min=1e-8))
    gt_log_ratio   = torch.log(gt_w.clamp(min=1e-8))   - torch.log(gt_h.clamp(min=1e-8))

    # Exclude fixed-shape, preplaced, and padding tokens
    is_fixed     = constraints[:, :, 0] > 0   # [B, k]
    is_preplaced = constraints[:, :, 1] > 0   # [B, k]
    valid = ~(is_fixed | is_preplaced | key_padding_mask)   # [B, k]

    if valid.sum() == 0:
        return pred_positions_norm.sum() * 0.0   # differentiable zero

    err = (pred_log_ratio - gt_log_ratio).pow(2)   # [B, k]
    return err[valid].mean()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("ratio_loss.py  —  smoke test")
    print("=" * 55)

    torch.manual_seed(0)
    B, k = 2, 10

    pred_norm = torch.rand(B, k, 4).clamp(min=0.01).detach().requires_grad_(True)
    gt_norm   = torch.rand(B, k, 4).clamp(min=0.01)

    constraints = torch.zeros(B, k, 5)
    constraints[:, :2, 0] = 1.0   # blocks 0-1: fixed shape (excluded)
    constraints[:, 2:4, 1] = 1.0  # blocks 2-3: preplaced (excluded)

    kpm = torch.zeros(B, k, dtype=torch.bool)
    kpm[:, -1] = True              # last token is padding (excluded)

    loss = ratio_loss(pred_norm, gt_norm, constraints, kpm)

    print(f"\npred_norm shape  : {tuple(pred_norm.shape)}")
    print(f"gt_norm shape    : {tuple(gt_norm.shape)}")
    print(f"Loss value       : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    loss.backward()
    assert pred_norm.grad is not None, "No gradient to pred_norm"
    print(f"Gradient norm    : {pred_norm.grad.norm().item():.6f}")

    # Perfect prediction → loss = 0
    zero_loss = ratio_loss(gt_norm.detach().requires_grad_(True), gt_norm, constraints, kpm)
    print(f"Perfect pred loss: {zero_loss.item():.2e}  (expected ~0)")
    assert zero_loss.item() < 1e-6

    print("\nSmoke test passed.")
    print("=" * 55)
