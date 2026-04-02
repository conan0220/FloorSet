"""
L_violation — soft constraint violation penalty.

L_violation = V_grouping + V_mib + V_boundary + V_overlap

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
) -> torch.Tensor:
    """
    Returns scalar: total soft-constraint violation summed over the batch.
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
    is_mib     = constraints[:, :, 2].float()       # [B, k]
    is_cluster = constraints[:, :, 3].float()       # [B, k]
    bcode      = constraints[:, :, 4].long()        # [B, k]  bitmask int

    # ── V_grouping ─────────────────────────────────────────────────
    n_clus   = is_cluster.sum(dim=1, keepdim=True).clamp(min=1)          # [B, 1]
    mean_cx  = (cx * is_cluster).sum(dim=1, keepdim=True) / n_clus       # [B, 1]
    mean_cy  = (cy * is_cluster).sum(dim=1, keepdim=True) / n_clus       # [B, 1]
    dist_clus = ((cx - mean_cx).pow(2) + (cy - mean_cy).pow(2) + 1e-8).sqrt()
    v_grouping = (dist_clus * is_cluster).sum()

    # ── V_mib ──────────────────────────────────────────────────────
    pw    = pred[:, :, 2]                                                 # [B, k]
    ph    = pred[:, :, 3]                                                 # [B, k]
    n_mib = is_mib.sum(dim=1, keepdim=True).clamp(min=1)                 # [B, 1]
    mean_w = (pw * is_mib).sum(dim=1, keepdim=True) / n_mib              # [B, 1]
    mean_h = (ph * is_mib).sum(dim=1, keepdim=True) / n_mib              # [B, 1]
    dist_mib = ((pw - mean_w).pow(2) + (ph - mean_h).pow(2) + 1e-8).sqrt()
    v_mib = (dist_mib * is_mib).sum()

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

    # ── V_overlap ──────────────────────────────────────────────────
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
    n_pairs = max(k * (k - 1) / 2, 1)
    v_overlap = (ov_area * triu_mask.unsqueeze(0)).sum() / n_pairs

    return v_grouping + v_mib + v_boundary + v_overlap


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

    loss = violation_loss(pred_norm, constraints)

    print(f"\npred_positions_norm : {tuple(pred_norm.shape)}")
    print(f"constraints         : {tuple(constraints.shape)}")
    print(f"Loss value          : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"
    assert loss.item() >= 0, "Loss must be non-negative"

    loss.backward()
    assert pred_norm.grad is not None, "No gradient to pred_positions_norm"
    print(f"Gradient norm       : {pred_norm.grad.norm().item():.6f}")

    # No-overlap test: blocks arranged in non-overlapping grid
    no_ov = torch.zeros(B, k, 4)
    for i in range(k):
        no_ov[:, i, 0] = i * 0.1   # x
        no_ov[:, i, 1] = 0.0       # y
        no_ov[:, i, 2] = 0.09      # w
        no_ov[:, i, 3] = 0.09      # h
    cons_ov = torch.zeros(B, k, 5)
    ov_loss = violation_loss(no_ov, cons_ov)
    print(f"No-overlap V_overlap: {ov_loss.item():.2e}  (expected ~0)")
    assert ov_loss.item() < 1e-6

    print("\nSmoke test passed.")
    print("=" * 55)
