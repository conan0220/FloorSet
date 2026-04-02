"""
Regression Head.

Maps Decoder hidden states [B, k, d_model] to block placements [B, k, 4].

Architecture:
    MLP: Linear(d_model, 64) → ReLU → Linear(64, 3)
    Outputs (x_raw, y_raw, log_ratio).

Post-processing:
    x = Softplus(x_raw)                              # strictly > 0
    y = Softplus(y_raw)                              # strictly > 0
    ratio = exp(log_ratio)                           # w / h, strictly > 0
    w = sqrt(area_target_norm * ratio)               # ensures w * h == area_target_norm
    h = sqrt(area_target_norm / ratio)

    For fixed-shape or preplaced blocks:
        w, h overridden with target_w, target_h from token_features.
    For preplaced blocks:
        x, y overridden with target_x, target_y from token_features.

Token feature layout (18 dims):
    [0]    area_target_norm   (area_target / canvas_ref²)
    [1]    is_soft
    [2]    is_fixed_shape
    [3]    is_preplaced
    [4-5]  target_w, target_h (normalised)
    [6-7]  target_x, target_y (normalised)
    [8-15] boundary_type one-hot
    [16]   is_mib
    [17]   is_cluster
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import D_MODEL


class RegressionHead(nn.Module):
    """
    Two-layer MLP that maps decoder embeddings to (x, y, w, h).

    Args:
        d_model  (int): Input dimension. Default: D_MODEL (256).
        hidden   (int): Hidden layer dimension. Default: 64.
    """

    def __init__(self, d_model: int = D_MODEL, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),   # (x_raw, y_raw, log_ratio)
        )

    def forward(
        self,
        x:              torch.Tensor,   # [B, k, d_model]  decoder hidden states
        token_features: torch.Tensor,   # [B, k, 18]
    ) -> torch.Tensor:
        """
        Args:
            x:              [B, k, d_model]
            token_features: [B, k, 18]

        Returns:
            [B, k, 4]  predicted (x, y, w, h), normalised scale
        """
        raw       = self.net(x)                                      # [B, k, 3]
        x_raw     = raw[..., 0]                                      # [B, k]
        y_raw     = raw[..., 1]                                      # [B, k]
        log_ratio = raw[..., 2]                                      # [B, k]

        # x, y strictly positive via Softplus
        pred_x = F.softplus(x_raw)                                   # [B, k]
        pred_y = F.softplus(y_raw)                                   # [B, k]

        # w, h from area constraint: w * h == area_target_norm exactly
        area_norm = token_features[..., 0].clamp(min=1e-12)          # [B, k]
        ratio     = torch.exp(log_ratio)                             # [B, k]
        pred_w    = (area_norm * ratio).clamp(min=1e-8).sqrt()       # [B, k]
        pred_h    = (area_norm / ratio.clamp(min=1e-8)).clamp(min=1e-8).sqrt()  # [B, k]

        # Fixed / preplaced: override w, h with target size
        is_fixed     = token_features[..., 2]                        # [B, k]
        is_preplaced = token_features[..., 3]                        # [B, k]
        has_fixed_size = (is_fixed + is_preplaced) > 0               # [B, k] bool

        target_w = token_features[..., 4]                            # [B, k]
        target_h = token_features[..., 5]                            # [B, k]
        target_x = token_features[..., 6]                            # [B, k]
        target_y = token_features[..., 7]                            # [B, k]

        pred_w = torch.where(has_fixed_size, target_w, pred_w)
        pred_h = torch.where(has_fixed_size, target_h, pred_h)

        # Preplaced: override x, y too
        is_pre_bool = is_preplaced > 0                               # [B, k] bool
        pred_x = torch.where(is_pre_bool, target_x, pred_x)
        pred_y = torch.where(is_pre_bool, target_y, pred_y)

        return torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1) # [B, k, 4]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 45)
    print("regression_head.py  —  smoke test")
    print("=" * 45)

    B, k = 2, 55
    x = torch.randn(B, k, D_MODEL)

    # Build dummy token_features [B, k, 18]
    tf = torch.zeros(B, k, 18)
    tf[:, :, 0] = 0.05        # area_target_norm
    tf[:, :, 1] = 1.0         # is_soft by default

    head = RegressionHead()
    head.eval()

    with torch.no_grad():
        out = head(x, tf)

    print(f"\nInput  shape : {tuple(x.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    assert out.shape == (B, k, 4), \
        f"Expected ({B}, {k}, 4), got {tuple(out.shape)}"

    # x, y must be strictly > 0 (Softplus)
    assert (out[..., 0] > 0).all(), "x must be > 0"
    assert (out[..., 1] > 0).all(), "y must be > 0"
    # w, h must be > 0
    assert (out[..., 2] > 0).all(), "w must be > 0"
    assert (out[..., 3] > 0).all(), "h must be > 0"
    # w * h should equal area_target_norm for soft blocks
    area_pred = out[..., 2] * out[..., 3]
    area_tgt  = tf[..., 0]
    err = (area_pred - area_tgt).abs().max().item()
    print(f"Max area constraint error : {err:.2e}  (expected ~0)")
    assert err < 1e-5, f"Area constraint violated: max error = {err}"

    # Preplaced block: x, y, w, h should match targets
    tf_pre = tf.clone()
    tf_pre[:, 0, 3] = 1.0   # is_preplaced
    tf_pre[:, 0, 2] = 0.0   # not is_fixed_shape
    tf_pre[:, 0, 4] = 0.3   # target_w
    tf_pre[:, 0, 5] = 0.2   # target_h
    tf_pre[:, 0, 6] = 0.1   # target_x
    tf_pre[:, 0, 7] = 0.05  # target_y
    with torch.no_grad():
        out_pre = head(x, tf_pre)
    assert abs(out_pre[0, 0, 0].item() - 0.1) < 1e-5, "preplaced x mismatch"
    assert abs(out_pre[0, 0, 1].item() - 0.05) < 1e-5, "preplaced y mismatch"
    assert abs(out_pre[0, 0, 2].item() - 0.3) < 1e-5, "preplaced w mismatch"
    assert abs(out_pre[0, 0, 3].item() - 0.2) < 1e-5, "preplaced h mismatch"
    print("Preplaced override       : OK")

    total = sum(p.numel() for p in head.parameters())
    print(f"Parameters   : {total:,}")

    print("\nSmoke test passed.")
    print("=" * 45)
