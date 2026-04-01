"""
L_violation — soft constraint violation penalty.

L_violation = V_grouping + V_mib + V_boundary + V_fixed + V_preplaced

  V_grouping : all is_cluster==1 blocks treated as one group;
               penalty = sum of centroid distances to the group mean centroid.

  V_mib      : all is_mib==1 blocks treated as one group;
               penalty = sum of L2 distances from each block's (w, h) to
               the group mean (w, h).

  V_boundary : for each block with a boundary_code constraint, penalise
               the gap between the block's relevant edge and the floorplan
               bounding-box edge.  boundary_code bits:
                   bit 0 (1)  → block left  edge must touch bbox x_min
                   bit 1 (2)  → block right edge must touch bbox x_max
                   bit 2 (4)  → block bottom edge must touch bbox y_min
                   bit 3 (8)  → block top   edge must touch bbox y_max

  V_fixed    : is_fixed_shape==1 blocks; penalty = L2( pred(w,h) - gt(w,h) ).

  V_preplaced: is_preplaced==1 blocks; penalty = L2( pred(x,y,w,h) - gt(x,y,w,h) ).

All violations are summed then divided by batch size B.
pred_positions_norm and gt_positions_norm must be in the same (normalised) scale.
constraints[:, :, 5] layout: [is_fixed_shape, is_preplaced, is_mib, is_cluster, boundary_code].
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/


def violation_loss(
    pred_positions_norm: torch.Tensor,   # [B, k, 4]  (x, y, w, h), normalised
    gt_positions_norm:   torch.Tensor,   # [B, k, 4]  GT in same scale
    constraints:         torch.Tensor,   # [B, k, 5]
) -> torch.Tensor:
    """
    Returns scalar: total soft-constraint violation averaged over the batch.
    """
    pred = pred_positions_norm
    gt   = gt_positions_norm
    B, k, _ = pred.shape
    device   = pred.device

    # ── Block geometry ─────────────────────────────────────────────
    cx      = pred[:, :, 0] + pred[:, :, 2] / 2   # [B, k]  centroid x
    cy      = pred[:, :, 1] + pred[:, :, 3] / 2   # [B, k]  centroid y
    x_left  = pred[:, :, 0]                        # [B, k]
    x_right = pred[:, :, 0] + pred[:, :, 2]        # [B, k]
    y_bot   = pred[:, :, 1]                        # [B, k]
    y_top   = pred[:, :, 1] + pred[:, :, 3]        # [B, k]

    # ── Floorplan bounding box ──────────────────────────────────────
    bbox_xmin = x_left.min(dim=1).values            # [B]
    bbox_xmax = x_right.max(dim=1).values           # [B]
    bbox_ymin = y_bot.min(dim=1).values             # [B]
    bbox_ymax = y_top.max(dim=1).values             # [B]

    # ── Constraint masks ───────────────────────────────────────────
    is_fixed     = constraints[:, :, 0].float()     # [B, k]
    is_preplaced = constraints[:, :, 1].float()     # [B, k]
    is_mib       = constraints[:, :, 2].float()     # [B, k]
    is_cluster   = constraints[:, :, 3].float()     # [B, k]
    bcode        = constraints[:, :, 4].long()      # [B, k]  bitmask int

    # ── V_grouping ─────────────────────────────────────────────────
    n_clus   = is_cluster.sum(dim=1, keepdim=True).clamp(min=1)          # [B, 1]
    mean_cx  = (cx * is_cluster).sum(dim=1, keepdim=True) / n_clus       # [B, 1]
    mean_cy  = (cy * is_cluster).sum(dim=1, keepdim=True) / n_clus       # [B, 1]
    dist_clus = ((cx - mean_cx).pow(2) + (cy - mean_cy).pow(2) + 1e-8).sqrt()   # [B, k]
    v_grouping = (dist_clus * is_cluster).sum()

    # ── V_mib ──────────────────────────────────────────────────────
    pw       = pred[:, :, 2]                                              # [B, k]
    ph       = pred[:, :, 3]                                              # [B, k]
    n_mib    = is_mib.sum(dim=1, keepdim=True).clamp(min=1)              # [B, 1]
    mean_w   = (pw * is_mib).sum(dim=1, keepdim=True) / n_mib            # [B, 1]
    mean_h   = (ph * is_mib).sum(dim=1, keepdim=True) / n_mib            # [B, 1]
    dist_mib = ((pw - mean_w).pow(2) + (ph - mean_h).pow(2) + 1e-8).sqrt()      # [B, k]
    v_mib    = (dist_mib * is_mib).sum()

    # ── V_boundary ─────────────────────────────────────────────────
    has_left  = (bcode & 1).float()                                       # [B, k]
    has_right = ((bcode >> 1) & 1).float()
    has_bot   = ((bcode >> 2) & 1).float()
    has_top   = ((bcode >> 3) & 1).float()

    pen_left  = (x_left  - bbox_xmin.unsqueeze(1)).clamp(min=0) * has_left
    pen_right = (bbox_xmax.unsqueeze(1) - x_right).clamp(min=0) * has_right
    pen_bot   = (y_bot   - bbox_ymin.unsqueeze(1)).clamp(min=0) * has_bot
    pen_top   = (bbox_ymax.unsqueeze(1) - y_top  ).clamp(min=0) * has_top

    v_boundary = (pen_left + pen_right + pen_bot + pen_top).sum()

    # ── V_fixed ────────────────────────────────────────────────────
    diff_wh  = (pred[:, :, 2:4] - gt[:, :, 2:4]).pow(2).sum(-1).add(1e-8).sqrt()   # [B, k]
    v_fixed  = (diff_wh * is_fixed).sum()

    # ── V_preplaced ────────────────────────────────────────────────
    diff_all    = (pred - gt).pow(2).sum(-1).add(1e-8).sqrt()                        # [B, k]
    v_preplaced = (diff_all * is_preplaced).sum()

    # ── Total, averaged over batch ─────────────────────────────────
    total = v_grouping + v_mib + v_boundary + v_fixed + v_preplaced
    return total / B


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("violation_loss.py  —  smoke test")
    print("=" * 55)

    torch.manual_seed(0)
    B, k = 2, 10

    pred_norm = (torch.rand(B, k, 4) * 0.8).detach().requires_grad_(True)
    gt_norm   = torch.rand(B, k, 4) * 0.8

    # Constraints: [is_fixed_shape, is_preplaced, is_mib, is_cluster, boundary_code]
    constraints = torch.zeros(B, k, 5)
    constraints[:, 0, 1] = 1.0                 # block 0: preplaced
    constraints[:, 1, 0] = 1.0                 # block 1: fixed-shape
    constraints[:, 2:4, 2] = 1.0               # blocks 2,3: mib
    constraints[:, 4:6, 3] = 1.0               # blocks 4,5: cluster
    constraints[:, 6, 4] = 1.0                 # block 6: left boundary (code=1)
    constraints[:, 7, 4] = 6.0                 # block 7: right+bottom (code=6)

    loss = violation_loss(pred_norm, gt_norm, constraints)

    print(f"\npred_positions_norm : {tuple(pred_norm.shape)}")
    print(f"gt_positions_norm   : {tuple(gt_norm.shape)}")
    print(f"constraints         : {tuple(constraints.shape)}")
    print(f"Loss value          : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    loss.backward()
    assert pred_norm.grad is not None, "No gradient to pred_positions_norm"
    print(f"Gradient norm       : {pred_norm.grad.norm().item():.6f}")

    # Perfect prediction — test without boundary constraints so that
    # V_fixed=0, V_preplaced=0, V_grouping=0 (identical positions), V_mib=0
    constraints_no_bdry = constraints.clone()
    constraints_no_bdry[:, :, 4] = 0.0     # clear boundary codes

    gt_perf = torch.rand(B, k, 4) * 0.5
    # Make mib blocks identical in (w, h)
    gt_perf[:, 2:4, 2:4] = gt_perf[:, 2:3, 2:4].expand(-1, 2, -1)
    # Make cluster blocks identical (same centroid)
    gt_perf[:, 4:6, :] = gt_perf[:, 4:5, :].expand(-1, 2, -1)

    zero_loss = violation_loss(gt_perf, gt_perf, constraints_no_bdry)
    print(f"Perfect pred loss   : {zero_loss.item():.2e}  (expected ~0)")
    assert abs(zero_loss.item()) < 1e-4, f"Expected ~0, got {zero_loss.item()}"

    # Boundary penalty: a block at x_min should give 0 left-penalty
    pred_bdry = torch.zeros(B, k, 4)
    pred_bdry[:, :, 2] = 0.1   # w
    pred_bdry[:, :, 3] = 0.1   # h
    # Spread x positions so block 6 IS at x_min
    for i in range(k):
        pred_bdry[:, i, 0] = i * 0.15
        pred_bdry[:, i, 1] = i * 0.15
    # block 6 (left boundary, code=1): x_left[6]=0.9; x_min over all=0 → penalty=0.9
    # Let's instead put block 6 at x_min
    pred_bdry[:, 6, 0] = 0.0   # left edge at x_min=0
    cons_bdry = torch.zeros(B, k, 5)
    cons_bdry[:, 6, 4] = 1.0   # left boundary
    bdry_loss = violation_loss(pred_bdry.detach(), torch.zeros(B, k, 4), cons_bdry)
    print(f"Boundary (block at edge) loss : {bdry_loss.item():.2e}  (expected ~0)")

    print("\nSmoke test passed.")
    print("=" * 55)
