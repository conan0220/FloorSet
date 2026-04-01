"""
L_overlap — soft pairwise block overlap penalty.

For each pair (i, j) with i < j, the overlap area is:
    overlap_x = max(0, min(x_i+w_i, x_j+w_j) - max(x_i, x_j))
    overlap_y = max(0, min(y_i+h_i, y_j+h_j) - max(y_i, y_j))
    overlap   = overlap_x * overlap_y

L_overlap = mean over batch of:
    sum_{i<j} overlap(i, j) / canvas_ref²

pred_positions_norm should be normalised (divided by canvas_ref) so the
penalty is scale-invariant across different floorplan sizes.
"""

import torch


def overlap_loss(
    pred_positions_norm: torch.Tensor,   # [B, k, 4]  (x, y, w, h), normalised
) -> torch.Tensor:
    """
    Returns scalar: mean total pairwise overlap area across the batch.
    Fully differentiable via clamp.
    """
    x = pred_positions_norm[:, :, 0]   # [B, k]
    y = pred_positions_norm[:, :, 1]   # [B, k]
    w = pred_positions_norm[:, :, 2]   # [B, k]
    h = pred_positions_norm[:, :, 3]   # [B, k]

    x_end = x + w   # [B, k]
    y_end = y + h   # [B, k]

    # Broadcast all pairs: [B, k, k]
    # overlap_x[b, i, j] = max(0, min(x_end_i, x_end_j) - max(x_i, x_j))
    overlap_x = (
        torch.minimum(x_end.unsqueeze(2), x_end.unsqueeze(1))
        - torch.maximum(x.unsqueeze(2), x.unsqueeze(1))
    ).clamp(min=0)   # [B, k, k]

    overlap_y = (
        torch.minimum(y_end.unsqueeze(2), y_end.unsqueeze(1))
        - torch.maximum(y.unsqueeze(2), y.unsqueeze(1))
    ).clamp(min=0)   # [B, k, k]

    overlap_area = overlap_x * overlap_y   # [B, k, k]

    # Zero out diagonal (self-overlap) and take upper triangle to avoid double-counting
    B, k, _ = overlap_area.shape
    mask = torch.triu(torch.ones(k, k, device=overlap_area.device), diagonal=1)
    overlap_area = overlap_area * mask.unsqueeze(0)   # [B, k, k]

    return overlap_area.sum()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("overlap_loss.py  —  smoke test")
    print("=" * 50)

    torch.manual_seed(0)
    B, k = 2, 4

    # No overlap: place blocks in a 2×2 grid, each 0.4×0.4
    no_overlap = torch.tensor([[[0.0, 0.0, 0.4, 0.4],
                                 [0.5, 0.0, 0.4, 0.4],
                                 [0.0, 0.5, 0.4, 0.4],
                                 [0.5, 0.5, 0.4, 0.4]]]).expand(B, -1, -1)
    loss_zero = overlap_loss(no_overlap)
    print(f"\nNo-overlap loss    : {loss_zero.item():.6f}  (expected 0)")
    assert loss_zero.item() < 1e-6

    # Full overlap: all blocks at the same position 0.0,0.0,0.4,0.4
    full_overlap = torch.zeros(B, k, 4)
    full_overlap[:, :, 2] = 0.4
    full_overlap[:, :, 3] = 0.4
    full_overlap_req = full_overlap.detach().requires_grad_(True)
    loss_full = overlap_loss(full_overlap_req)
    # B * k*(k-1)/2 pairs × 0.4*0.4 = 2 * 6 × 0.16 = 1.92
    print(f"Full-overlap loss  : {loss_full.item():.6f}  (expected {B * k*(k-1)//2 * 0.16:.4f})")
    assert abs(loss_full.item() - B * k * (k - 1) / 2 * 0.16) < 1e-4

    loss_full.backward()
    assert full_overlap_req.grad is not None
    print(f"Gradient norm      : {full_overlap_req.grad.norm().item():.6f}")

    print("\nSmoke test passed.")
    print("=" * 50)
