"""
L_area — bounding-box area gap loss.

Areabbox_pred = (x_max - x_min) * (y_max - y_min)

where:
    x_min = min block left edge  = min(x)
    x_max = max block right edge = max(x + w)
    y_min = min block bottom     = min(y)
    y_max = max block top        = max(y + h)

L_area = mean over batch of:
    (Areabbox_pred - area_baseline) / area_baseline

Note: pred_positions_raw and area_baseline must be in the same (raw pixel) scale.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/


def area_loss(
    pred_positions_raw: torch.Tensor,   # [B, k, 4]  (x, y, w, h) raw scale
    area_baseline:      torch.Tensor,   # [B]
) -> torch.Tensor:
    """
    Returns scalar: mean bounding-box area gap across the batch.
    """
    pred = pred_positions_raw                         # [B, k, 4]

    x      = pred[:, :, 0]                           # [B, k]
    y      = pred[:, :, 1]                           # [B, k]
    x_end  = pred[:, :, 0] + pred[:, :, 2]           # [B, k]  x + w
    y_end  = pred[:, :, 1] + pred[:, :, 3]           # [B, k]  y + h

    x_min  = x.min(dim=1).values                     # [B]
    x_max  = x_end.max(dim=1).values                 # [B]
    y_min  = y.min(dim=1).values                     # [B]
    y_max  = y_end.max(dim=1).values                 # [B]

    bbox_area = (x_max - x_min).clamp(min=0) * (y_max - y_min).clamp(min=0)  # [B]

    baseline  = area_baseline.to(pred.device).clamp(min=1e-8)
    gap       = (bbox_area - baseline) / baseline     # [B]

    # Clamp to non-negative: doing better than baseline is free,
    # but should not reduce total loss and mask other penalties.
    return gap.clamp(min=0).mean()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 50)
    print("area_loss.py  —  smoke test")
    print("=" * 50)

    torch.manual_seed(0)
    B, k = 2, 10

    pred_raw = (torch.rand(B, k, 4) * 200).detach().requires_grad_(True)
    # Fake baseline that's smaller than bounding box → positive gap expected
    area_baseline = torch.tensor([5000.0, 6000.0])

    loss = area_loss(pred_raw, area_baseline)

    print(f"\npred_positions_raw : {tuple(pred_raw.shape)}")
    print(f"area_baseline      : {area_baseline.tolist()}")
    print(f"Loss value         : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"

    loss.backward()
    assert pred_raw.grad is not None, "No gradient to pred_positions_raw"
    print(f"Gradient norm      : {pred_raw.grad.norm().item():.6f}")

    # GT positions → gap should be exactly 0 when baseline = bbox(gt)
    gt_raw = pred_raw.detach().clone()
    x_gt     = gt_raw[:, :, 0]
    y_gt     = gt_raw[:, :, 1]
    x_end_gt = gt_raw[:, :, 0] + gt_raw[:, :, 2]
    y_end_gt = gt_raw[:, :, 1] + gt_raw[:, :, 3]
    gt_bbox  = (
        (x_end_gt.max(1).values - x_gt.min(1).values).clamp(min=0)
        * (y_end_gt.max(1).values - y_gt.min(1).values).clamp(min=0)
    )
    zero_loss = area_loss(gt_raw, gt_bbox)
    print(f"Same-as-baseline loss : {zero_loss.item():.2e}  (expected ~0)")
    assert abs(zero_loss.item()) < 1e-4, f"Expected ~0, got {zero_loss.item()}"

    print("\nSmoke test passed.")
    print("=" * 50)
