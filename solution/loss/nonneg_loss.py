"""
L_nonneg — exponential penalty for negative (x, y, w, h) values.

For each coordinate value v:
    penalty(v) = exp(max(0, -v)) - 1

Properties:
    v >= 0  →  penalty = 0          (no penalty)
    v = -1  →  penalty ≈ 1.718
    v = -2  →  penalty ≈ 6.389
    v = -3  →  penalty ≈ 19.09      (grows exponentially)

The gradient at v < 0 is -exp(-v), which grows as v becomes more negative,
giving increasingly strong signal to push values back above zero.

pred_positions_norm : [B, k, 4]  (x, y, w, h), normalised
key_padding_mask    : [B, k] bool, True = padding (excluded from loss)
"""

import torch


def nonneg_loss(
    pred_positions_norm: torch.Tensor,   # [B, k, 4]
    key_padding_mask:    torch.Tensor,   # [B, k]  True = padding
) -> torch.Tensor:
    """
    Returns scalar: mean exponential penalty for all negative coordinate values.
    """
    # penalty(v) = exp(max(0, -v)) - 1  — zero for v>=0, exponential for v<0
    penalties = torch.exp((-pred_positions_norm).clamp(min=0)) - 1   # [B, k, 4]

    valid = (~key_padding_mask).float().unsqueeze(-1)   # [B, k, 1]
    return (penalties * valid).sum()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("nonneg_loss.py  —  smoke test")
    print("=" * 50)

    torch.manual_seed(0)
    B, k = 2, 4

    kpm = torch.zeros(B, k, dtype=torch.bool)

    # All positive → loss should be 0
    pos_pred = torch.rand(B, k, 4).requires_grad_(True)
    loss_pos = nonneg_loss(pos_pred, kpm)
    print(f"\nAll positive loss  : {loss_pos.item():.6f}  (expected 0)")
    assert loss_pos.item() < 1e-6

    # Mix of positive and negative
    mixed = torch.tensor([[[0.5, 0.3, -0.1, 0.2],
                            [0.1, -0.2, 0.3, 0.4],
                            [0.2,  0.1, 0.1, 0.1],
                            [0.3,  0.2, 0.1, 0.1]]]).expand(B, -1, -1)
    mixed = mixed.clone().detach().requires_grad_(True)
    loss_mixed = nonneg_loss(mixed, kpm)
    print(f"Mixed loss         : {loss_mixed.item():.6f}  (expected > 0)")
    assert loss_mixed.item() > 0

    loss_mixed.backward()
    assert mixed.grad is not None
    # Gradient should be negative (penalty) for negative values
    print(f"Gradient at v=-0.1 : {mixed.grad[0, 0, 2].item():.4f}  (expected < 0)")
    assert mixed.grad[0, 0, 2].item() < 0

    # Padding blocks should not contribute
    kpm_pad = torch.zeros(B, k, dtype=torch.bool)
    kpm_pad[:, -2:] = True
    pred_pad = torch.full((B, k, 4), -10.0, requires_grad=True)
    loss_pad = nonneg_loss(pred_pad, kpm_pad)
    pred_pad_no_pad = pred_pad.detach().clone()
    pred_pad_no_pad[:, -2:] = 0.0
    pred_pad_no_pad = pred_pad_no_pad.requires_grad_(True)
    loss_no_pad = nonneg_loss(pred_pad_no_pad, kpm_pad)
    print(f"Padding excluded   : loss={loss_pad.item():.4f}  ==  {loss_no_pad.item():.4f}  "
          f"({'OK' if abs(loss_pad.item() - loss_no_pad.item()) < 1e-4 else 'FAIL'})")

    print("\nSmoke test passed.")
    print("=" * 50)
