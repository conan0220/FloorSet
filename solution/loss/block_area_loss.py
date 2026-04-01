"""
L_block_area — per-block area constraint loss.

Forces each block's predicted area (w × h) to match its area_target.

    L_block_area = (1/k) * sum_i ( ŵ_i * ĥ_i - area_target_i / canvas_ref² )²

pred_positions_norm : [B, k, 4]  (x, y, w, h), normalised by canvas_ref
token_features      : [B, k, 18] token feature matrix; area_target_norm is
                      stored at index 0 (already divided by canvas_ref²).
key_padding_mask    : [B, k] bool, True = padding (excluded from loss).
"""

import torch


def block_area_loss(
    pred_positions_norm: torch.Tensor,   # [B, k, 4]
    token_features:      torch.Tensor,   # [B, k, 18]
    key_padding_mask:    torch.Tensor,   # [B, k]  True = padding
) -> torch.Tensor:
    """
    Returns scalar: mean per-block area error across all valid blocks and batch.
    """
    pred_w = pred_positions_norm[:, :, 2]   # [B, k]
    pred_h = pred_positions_norm[:, :, 3]   # [B, k]
    pred_area = pred_w * pred_h             # [B, k]

    # area_target_norm is feature index 0: area_target / canvas_ref²
    area_target_norm = token_features[:, :, 0]   # [B, k]

    valid = (~key_padding_mask).float()          # [B, k]  1 = valid, 0 = padding

    error = (pred_area - area_target_norm) ** 2  # [B, k]

    n_valid = valid.sum().clamp(min=1)
    return (error * valid).sum() / n_valid


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("block_area_loss.py  —  smoke test")
    print("=" * 50)

    torch.manual_seed(0)
    B, k = 2, 6

    # area_target_norm stored at token_features[:, :, 0]
    area_target_norm = torch.rand(B, k) * 0.1 + 0.01   # [B, k]
    token_features = torch.zeros(B, k, 18)
    token_features[:, :, 0] = area_target_norm

    # padding: last 2 blocks are padding
    key_padding_mask = torch.zeros(B, k, dtype=torch.bool)
    key_padding_mask[:, -2:] = True

    # Perfect prediction: pred w*h == area_target_norm
    # Use w = h = sqrt(area_target_norm)
    sqrt_area = area_target_norm.sqrt()
    pred_perfect = torch.zeros(B, k, 4)
    pred_perfect[:, :, 2] = sqrt_area
    pred_perfect[:, :, 3] = sqrt_area
    pred_perfect = pred_perfect.detach().requires_grad_(True)

    loss_zero = block_area_loss(pred_perfect, token_features, key_padding_mask)
    print(f"\nPerfect pred loss  : {loss_zero.item():.2e}  (expected ~0)")
    assert loss_zero.item() < 1e-6

    # Random prediction
    pred_rand = torch.rand(B, k, 4, requires_grad=True)
    loss_rand = block_area_loss(pred_rand, token_features, key_padding_mask)
    print(f"Random pred loss   : {loss_rand.item():.6f}")
    assert loss_rand.ndim == 0
    assert loss_rand.item() >= 0

    loss_rand.backward()
    assert pred_rand.grad is not None
    print(f"Gradient norm      : {pred_rand.grad.norm().item():.6f}")

    # Padding blocks must not contribute
    pred_pad = torch.zeros(B, k, 4, requires_grad=True)
    # Make non-padded blocks perfect, padded blocks wildly wrong
    with torch.no_grad():
        pred_pad[:, :k-2, 2] = sqrt_area[:, :k-2]
        pred_pad[:, :k-2, 3] = sqrt_area[:, :k-2]
        pred_pad[:, -2:, 2]  = 999.0
        pred_pad[:, -2:, 3]  = 999.0
    pred_pad = pred_pad.detach().requires_grad_(True)
    loss_pad = block_area_loss(pred_pad, token_features, key_padding_mask)
    print(f"Padding-ignored loss: {loss_pad.item():.2e}  (expected ~0)")
    assert loss_pad.item() < 1e-6

    print("\nSmoke test passed.")
    print("=" * 50)
