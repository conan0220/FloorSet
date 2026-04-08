"""
L_violation — soft constraint violation penalty.

L_violation = V_grouping + V_mib + V_boundary + V_overlap

  V_grouping : blocks are grouped by cluster_group_id (1–4);
               within each group, penalise centroid distances to that group's mean centroid;
               groups are independent (group 1 blocks pull toward group-1 mean only).

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

  V_overlap  : all block pair (i, j) pairwise overlap area sum:
                   overlap_x = max(0, min(x̂ᵢ+ŵᵢ, x̂ⱼ+ŵⱼ) - max(x̂ᵢ, x̂ⱼ))
                   overlap_y = max(0, min(ŷᵢ+ĥᵢ, ŷⱼ+ĥⱼ) - max(ŷᵢ, ŷⱼ))
                   V_overlap = sum of (overlap_x * overlap_y) for all i < j

All components are computed in normalised space.
V_fixed and V_preplaced are handled by the Regression Head architecture and
are therefore not included here.

constraints layout: [is_fixed_shape, is_preplaced, is_mib, is_cluster, boundary_code].
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/


def violation_loss(
    pred_positions_norm: torch.Tensor,   # [B, k, 4]  (x, y, w, h), normalised
    constraints:         torch.Tensor,   # [B, k, 5]
) -> tuple:
    """
    Returns (v_grouping, v_mib, v_boundary, v_overlap) — four normalised scalars.

    Each term is normalised to "average penalty per constrained block/pair" so
    that all four are on a comparable scale regardless of batch size or k.
    """
    pred = pred_positions_norm
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
    is_mib      = constraints[:, :, 2].float()      # [B, k]
    clust_ids   = constraints[:, :, 3].long()       # [B, k]  group IDs 0–4
    bcode       = constraints[:, :, 4].long()       # [B, k]  bitmask int

    # ── V_grouping ─────────────────────────────────────────────────
    # Average centroid distance to group mean, per cluster block.
    v_grouping_sum  = pred.new_zeros(1).squeeze()
    n_cluster_total = pred.new_zeros(1).squeeze()
    for g in range(1, 5):
        mask = (clust_ids == g).float()             # [B, k]
        n_g  = mask.sum(dim=1, keepdim=True).clamp(min=1)
        has_any = (mask.sum(dim=1) > 0)
        if not has_any.any():
            continue
        mean_cx_g = (cx * mask).sum(dim=1, keepdim=True) / n_g
        mean_cy_g = (cy * mask).sum(dim=1, keepdim=True) / n_g
        dist_g = ((cx - mean_cx_g).pow(2) + (cy - mean_cy_g).pow(2) + 1e-8).sqrt()
        v_grouping_sum  = v_grouping_sum  + (dist_g * mask).sum()
        n_cluster_total = n_cluster_total + mask.sum()
    v_grouping = v_grouping_sum / n_cluster_total.clamp(min=1)

    # ── V_mib ──────────────────────────────────────────────────────
    # Average (w, h) deviation from group mean, per mib block.
    pw    = pred[:, :, 2]                                                 # [B, k]
    ph    = pred[:, :, 3]                                                 # [B, k]
    n_mib = is_mib.sum(dim=1, keepdim=True).clamp(min=1)                 # [B, 1]
    mean_w = (pw * is_mib).sum(dim=1, keepdim=True) / n_mib
    mean_h = (ph * is_mib).sum(dim=1, keepdim=True) / n_mib
    dist_mib = ((pw - mean_w).pow(2) + (ph - mean_h).pow(2) + 1e-8).sqrt()
    v_mib = (dist_mib * is_mib).sum() / is_mib.sum().clamp(min=1)

    # ── V_boundary ─────────────────────────────────────────────────
    # Penalise the gap between a boundary block's relevant edge and the
    # canvas edge (0 or 1 in normalised space).  Using fixed canvas edges
    # gives a stable training signal regardless of the predicted bounding
    # box, which would otherwise be very small when boundary blocks are
    # placed last in the autoregressive sequence.
    #
    # Bit layout (matches violation_loss convention):
    #   bit 0 (1): left  → x_left  must touch x = 0
    #   bit 1 (2): right → x_right must touch x = 1
    #   bit 2 (4): bot   → y_bot   must touch y = 0
    #   bit 3 (8): top   → y_top   must touch y = 1
    has_left  = (bcode & 1).float()                                       # [B, k]
    has_right = ((bcode >> 1) & 1).float()
    has_bot   = ((bcode >> 2) & 1).float()
    has_top   = ((bcode >> 3) & 1).float()

    pen_left  = x_left.clamp(min=0)            * has_left   # gap from x = 0
    pen_right = (1.0 - x_right).clamp(min=0)  * has_right  # gap from x = 1
    pen_bot   = y_bot.clamp(min=0)             * has_bot    # gap from y = 0
    pen_top   = (1.0 - y_top).clamp(min=0)    * has_top    # gap from y = 1

    n_boundary = (bcode > 0).float().sum().clamp(min=1)
    v_boundary = (pen_left + pen_right + pen_bot + pen_top).sum() / n_boundary

    # ── V_overlap ──────────────────────────────────────────────────
    # Average overlap area per block pair.
    x_end = x_right                                                       # [B, k]
    y_end = y_top                                                         # [B, k]

    ov_x = (
        torch.minimum(x_end.unsqueeze(2), x_end.unsqueeze(1))
        - torch.maximum(x_left.unsqueeze(2), x_left.unsqueeze(1))
    ).clamp(min=0)                                                        # [B, k, k]

    ov_y = (
        torch.minimum(y_end.unsqueeze(2), y_end.unsqueeze(1))
        - torch.maximum(y_bot.unsqueeze(2), y_bot.unsqueeze(1))
    ).clamp(min=0)                                                        # [B, k, k]

    ov_area = ov_x * ov_y                                                 # [B, k, k]
    triu_mask = torch.triu(torch.ones(k, k, device=device), diagonal=1)
    # Normalise per block (not per pair) so the scale is comparable to other
    # violation terms and is not diluted by the many non-overlapping pairs.
    n_blocks = B * k
    v_overlap = (ov_area * triu_mask.unsqueeze(0)).sum() / n_blocks

    return v_grouping, v_mib, v_boundary, v_overlap


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

    # Constraints: [is_fixed_shape, is_preplaced, is_mib, is_cluster, boundary_code]
    constraints = torch.zeros(B, k, 5)
    constraints[:, 2:4, 2] = 1.0               # blocks 2,3: mib
    constraints[:, 4:6, 3] = 1.0               # blocks 4,5: cluster
    constraints[:, 6, 4] = 1.0                 # block 6: left boundary (code=1)
    constraints[:, 7, 4] = 6.0                 # block 7: right+bottom (code=6)

    v_grouping, v_mib, v_boundary, v_overlap = violation_loss(pred_norm, constraints)

    print(f"\npred_positions_norm : {tuple(pred_norm.shape)}")
    print(f"constraints         : {tuple(constraints.shape)}")
    print(f"v_grouping          : {v_grouping.item():.6f}")
    print(f"v_mib               : {v_mib.item():.6f}")
    print(f"v_boundary          : {v_boundary.item():.6f}")
    print(f"v_overlap           : {v_overlap.item():.6f}")
    for v in (v_grouping, v_mib, v_boundary, v_overlap):
        assert v.ndim == 0, "Expected scalar"
        assert v.item() >= 0, "Loss must be non-negative"

    v_overlap.backward()
    assert pred_norm.grad is not None, "No gradient to pred_positions_norm"
    print(f"Gradient norm (overlap): {pred_norm.grad.norm().item():.6f}")

    # No-overlap test: blocks arranged in non-overlapping grid
    no_ov = torch.zeros(B, k, 4)
    for i in range(k):
        no_ov[:, i, 0] = i * 0.1   # x
        no_ov[:, i, 1] = 0.0       # y
        no_ov[:, i, 2] = 0.09      # w
        no_ov[:, i, 3] = 0.09      # h
    cons_ov = torch.zeros(B, k, 5)
    _, _, _, ov_loss = violation_loss(no_ov, cons_ov)
    print(f"No-overlap V_overlap: {ov_loss.item():.2e}  (expected ~0)")
    assert ov_loss.item() < 1e-6

    print("\nSmoke test passed.")
    print("=" * 55)
