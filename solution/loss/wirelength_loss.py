"""
L_wirelength — HPWL gap loss.

HPWL_pred = Inter-module HPWL  +  External HPWL

  Inter-module HPWL:
    For each block pair (i, j) with w_int[b,i,j] > 0:
        contribution = w_int[b,i,j] * (|cx_i - cx_j| + |cy_i - cy_j|)
    where cx = x + w/2, cy = y + h/2  (block centre).
    The symmetric w_int is halved to avoid double-counting.

  External HPWL:
    For each valid edge in p2b_conn = (pin_idx, block_idx, edge_weight):
        contribution = edge_weight * (|pin_x - cx| + |pin_y - cy|)

L_wirelength = mean over batch of:
    (HPWL_pred - hpwl_baseline) / hpwl_baseline

Note: pred_positions_raw and hpwl_baseline must be in the same (raw pixel)
scale.  w_int passed here should be the *unnormalised* connectivity matrix.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/


def wirelength_loss(
    pred_positions_raw: torch.Tensor,   # [B, k, 4]  (x, y, w, h) raw scale
    w_int:              torch.Tensor,   # [B, k, k]  unnormalised b2b weights
    pins_pos:           torch.Tensor,   # [B, r, 2]  pin positions, raw scale
    p2b_conn:           torch.Tensor,   # [B, e, 3]  (pin_idx, block_idx, weight)
    hpwl_baseline:      torch.Tensor,   # [B]
) -> torch.Tensor:
    """
    Returns scalar: mean HPWL gap across the batch.
    """
    pred = pred_positions_raw
    B, k, _ = pred.shape
    device   = pred.device

    # ── Block centres ──────────────────────────────────────────────
    cx = pred[:, :, 0] + pred[:, :, 2] / 2   # [B, k]
    cy = pred[:, :, 1] + pred[:, :, 3] / 2   # [B, k]

    # ── Inter-module HPWL ──────────────────────────────────────────
    # Pairwise Manhattan between all block centres: [B, k, k]
    dx = (cx.unsqueeze(2) - cx.unsqueeze(1)).abs()
    dy = (cy.unsqueeze(2) - cy.unsqueeze(1)).abs()
    manhattan_b2b = dx + dy                   # [B, k, k]

    # 0.5 because w_int is symmetric → each pair counted twice
    hpwl_b2b = (w_int * manhattan_b2b).sum(dim=(-2, -1)) * 0.5   # [B]

    # ── External HPWL ──────────────────────────────────────────────
    e = p2b_conn.shape[1]
    r = pins_pos.shape[1]

    valid = p2b_conn[:, :, 0] >= 0                                # [B, e]

    # Safe indices (padding entries clamped to 0, then masked out)
    pin_idx_s   = p2b_conn[:, :, 0].long().clamp(0, r - 1)       # [B, e]
    block_idx_s = p2b_conn[:, :, 1].long().clamp(0, k - 1)       # [B, e]
    edge_w      = p2b_conn[:, :, 2]                               # [B, e]

    # Gather pin positions:   [B, e, 2]
    pin_xy = torch.gather(
        pins_pos, 1,
        pin_idx_s.unsqueeze(-1).expand(-1, -1, 2),
    )

    # Gather block centres:   [B, e, 2]
    block_centres = torch.stack([cx, cy], dim=-1)                  # [B, k, 2]
    block_xy = torch.gather(
        block_centres, 1,
        block_idx_s.unsqueeze(-1).expand(-1, -1, 2),
    )

    manhattan_p2b = (pin_xy - block_xy).abs().sum(dim=-1)          # [B, e]
    hpwl_p2b = (edge_w * manhattan_p2b * valid.float()).sum(dim=1)  # [B]

    # ── HPWL gap ───────────────────────────────────────────────────
    hpwl_pred = hpwl_b2b + hpwl_p2b                               # [B]
    baseline  = hpwl_baseline.to(device).clamp(min=1e-8)
    gap       = (hpwl_pred - baseline) / baseline                  # [B]

    return gap.mean()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("wirelength_loss.py  —  smoke test")
    print("=" * 50)

    torch.manual_seed(0)
    B, k, r, e = 2, 10, 5, 15

    # Place blocks in a grid-like spread (raw scale ~0..200)
    pred_raw = (torch.rand(B, k, 4) * 200).detach().requires_grad_(True)
    # w_int: symmetric, unnormalised
    _w = torch.rand(B, k, k) * 5
    w_int = (_w + _w.transpose(1, 2)) / 2

    pins_pos = torch.rand(B, r, 2) * 200

    # Construct valid p2b edges + some padding rows
    p2b = torch.full((B, e, 3), -1.0)
    n_valid = 10
    p2b[:, :n_valid, 0] = torch.randint(0, r, (B, n_valid)).float()
    p2b[:, :n_valid, 1] = torch.randint(0, k, (B, n_valid)).float()
    p2b[:, :n_valid, 2] = torch.rand(B, n_valid) * 3

    hpwl_baseline = torch.tensor([500.0, 600.0])

    loss = wirelength_loss(pred_raw, w_int, pins_pos, p2b, hpwl_baseline)

    print(f"\npred_positions_raw : {tuple(pred_raw.shape)}")
    print(f"w_int              : {tuple(w_int.shape)}")
    print(f"pins_pos           : {tuple(pins_pos.shape)}")
    print(f"p2b_conn           : {tuple(p2b.shape)}  ({n_valid} valid / {e} total)")
    print(f"hpwl_baseline      : {hpwl_baseline.tolist()}")
    print(f"Loss value         : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"

    loss.backward()
    assert pred_raw.grad is not None, "No gradient to pred_positions_raw"
    print(f"Gradient norm      : {pred_raw.grad.norm().item():.6f}")

    # GT positions → gap should be exactly 0 when baseline = HPWL(gt)
    gt_raw = pred_raw.detach().clone()
    # Compute full HPWL(gt): b2b + p2b
    cx_gt  = gt_raw[:, :, 0] + gt_raw[:, :, 2] / 2
    cy_gt  = gt_raw[:, :, 1] + gt_raw[:, :, 3] / 2
    dx_    = (cx_gt.unsqueeze(2) - cx_gt.unsqueeze(1)).abs()
    dy_    = (cy_gt.unsqueeze(2) - cy_gt.unsqueeze(1)).abs()
    hpwl_b2b_gt = (w_int * (dx_ + dy_)).sum(dim=(-2, -1)) * 0.5
    pin_idx_s_  = p2b[:, :, 0].long().clamp(0, r - 1)
    blk_idx_s_  = p2b[:, :, 1].long().clamp(0, k - 1)
    valid_      = p2b[:, :, 0] >= 0
    pin_xy_     = torch.gather(pins_pos, 1, pin_idx_s_.unsqueeze(-1).expand(-1,-1,2))
    ctrs_gt     = torch.stack([cx_gt, cy_gt], dim=-1)
    blk_xy_     = torch.gather(ctrs_gt, 1, blk_idx_s_.unsqueeze(-1).expand(-1,-1,2))
    hpwl_p2b_gt = (p2b[:,:,2] * (pin_xy_-blk_xy_).abs().sum(-1) * valid_.float()).sum(1)
    baseline_full = hpwl_b2b_gt + hpwl_p2b_gt
    zero_loss = wirelength_loss(gt_raw, w_int, pins_pos, p2b, baseline_full)
    print(f"Same-as-baseline loss : {zero_loss.item():.2e}  (expected ~0)")
    assert abs(zero_loss.item()) < 1e-4, f"Expected ~0, got {zero_loss.item()}"

    print("\nSmoke test passed.")
    print("=" * 50)
