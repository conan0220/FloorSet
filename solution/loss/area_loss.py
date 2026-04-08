"""
L_area — bounding-box area gap loss (soft bbox version).

Uses log-sum-exp to approximate max/min so that ALL blocks receive
gradients, not just the single boundary block.

  soft_max(v, τ) = τ · log( Σ exp(vᵢ / τ) )      ≈  max(vᵢ)  as τ → 0
  soft_min(v, τ) = -soft_max(-v, τ)

With τ → 0 the result is identical to hard max/min.
With moderate τ (e.g. 0.05 in normalised space) all blocks near the
boundary contribute gradient proportional to exp(edge / τ).

L_area = max(0, (soft_bbox_area - area_baseline) / area_baseline)
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/


def _soft_max(x: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    """Differentiable smooth maximum along `dim`."""
    return temperature * torch.logsumexp(x / temperature, dim=dim)


def _soft_min(x: torch.Tensor, temperature: float, dim: int) -> torch.Tensor:
    """Differentiable smooth minimum along `dim`."""
    return -_soft_max(-x, temperature, dim)


def area_loss(
    pred_positions_raw: torch.Tensor,   # [B, k, 4]  (x, y, w, h) raw scale
    area_baseline:      torch.Tensor,   # [B]
    temperature:        float = 20.0,   # soft-max temperature (raw pixel scale)
) -> torch.Tensor:
    """
    Returns scalar: mean soft bounding-box area gap across the batch.

    All blocks receive gradients via the log-sum-exp approximation of
    max/min, unlike the hard-max version where only the single boundary
    block gets a non-zero gradient.

    temperature: controls smoothness.  Larger → more blocks get gradient
    but approximation is less tight.  Default 5.0 (raw pixel units).
    """
    pred = pred_positions_raw                         # [B, k, 4]

    x_left  = pred[:, :, 0]                          # [B, k]
    y_bot   = pred[:, :, 1]                          # [B, k]
    x_right = pred[:, :, 0] + pred[:, :, 2]          # [B, k]  x + w
    y_top   = pred[:, :, 1] + pred[:, :, 3]          # [B, k]  y + h

    # Soft bbox edges — every block contributes gradient
    soft_xmax = _soft_max(x_right, temperature, dim=1)   # [B]
    soft_xmin = _soft_min(x_left,  temperature, dim=1)   # [B]
    soft_ymax = _soft_max(y_top,   temperature, dim=1)   # [B]
    soft_ymin = _soft_min(y_bot,   temperature, dim=1)   # [B]

    bbox_area = (soft_xmax - soft_xmin).clamp(min=0) * \
                (soft_ymax - soft_ymin).clamp(min=0)     # [B]

    baseline = area_baseline.to(pred.device).clamp(min=1e-8)
    gap      = (bbox_area - baseline) / baseline          # [B]

    return gap.clamp(min=0).mean()


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("area_loss.py  —  soft bbox smoke test")
    print("=" * 55)

    torch.manual_seed(0)
    B, k = 2, 10

    pred_raw = (torch.rand(B, k, 4) * 200).detach().requires_grad_(True)
    area_baseline = torch.tensor([5000.0, 6000.0])

    loss = area_loss(pred_raw, area_baseline, temperature=5.0)

    print(f"\npred_positions_raw : {tuple(pred_raw.shape)}")
    print(f"area_baseline      : {area_baseline.tolist()}")
    print(f"Loss value         : {loss.item():.6f}")
    assert loss.ndim == 0, "Expected scalar"

    loss.backward()
    assert pred_raw.grad is not None, "No gradient to pred_positions_norm"

    # Check how many blocks have non-zero gradient (should be ALL with soft bbox)
    grad_norm_per_block = pred_raw.grad[0].norm(dim=-1)   # [k]
    n_nonzero = (grad_norm_per_block > 1e-8).sum().item()
    print(f"Blocks with non-zero gradient: {n_nonzero} / {k}  (hard max would be <= 4)")
    print(f"  (with hard max this would be <= 4 out of {k})")
    assert n_nonzero > k // 2, f"Expected majority of blocks to have gradient, got {n_nonzero}/{k}"

    # Compare with hard-max version
    x_right = pred_raw[:, :, 0] + pred_raw[:, :, 2]
    x_left  = pred_raw[:, :, 0]
    y_top   = pred_raw[:, :, 1] + pred_raw[:, :, 3]
    y_bot   = pred_raw[:, :, 1]
    hard_bbox = (x_right.max(1).values - x_left.min(1).values).clamp(0) * \
                (y_top.max(1).values  - y_bot.min(1).values).clamp(0)
    soft_bbox = (
        (_soft_max(x_right, 5.0, 1) - _soft_min(x_left, 5.0, 1)).clamp(0) *
        (_soft_max(y_top,   5.0, 1) - _soft_min(y_bot,  5.0, 1)).clamp(0)
    )
    print(f"\nHard bbox area [0]: {hard_bbox[0].item():.2f}")
    print(f"Soft bbox area [0]: {soft_bbox[0].item():.2f}  (slightly larger due to smoothing)")

    print("\nSmoke test passed.")
    print("=" * 55)
