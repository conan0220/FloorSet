"""
Discrete Regression Head.

Maps Decoder hidden states to block placements using a G×G grid.

Architecture:
    shared: Linear(d_model, hidden) → ReLU
    xy_head:    Linear(hidden, G*G)   → logits over grid positions
    ratio_head: Linear(hidden, 1)     → log_ratio (w/h)

Post-processing:
    grid cell (gx, gy) → x = gx/G, y = gy/G  (top-left corner, normalised)
    ratio = exp(log_ratio)
    w = sqrt(area_target_norm * ratio)
    h = sqrt(area_target_norm / ratio)

    For fixed-shape or preplaced blocks:
        w, h overridden with target_w, target_h from token_features.
    For preplaced blocks:
        x, y overridden with target_x, target_y (grid prediction ignored).

forward() modes
───────────────
occupancy=None  (training / teacher-forcing):
    x:              [B, k, d_model]
    token_features: [B, k, 18]
    Returns: (pred_positions [B,k,4],  logits [B,k,G*G])

occupancy=[B,G,G]  (autoregressive inference):
    x:              [B, 1, d_model]
    token_features: [B, 1, 18]
    Returns: pred_positions [B,1,4]
    Side-effect: updates occupancy in-place with the newly placed block.

Token feature layout (18 dims):
    [0]    area_target_norm
    [1]    is_soft
    [2]    is_fixed_shape
    [3]    is_preplaced
    [4-5]  target_w, target_h (normalised)
    [6-7]  target_x, target_y (normalised)
    [8-15] boundary_type one-hot
    [16]   is_mib
    [17]   is_cluster
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import D_MODEL, GRID_SIZE


class DiscreteRegressionHead(nn.Module):
    """
    Grid-based regression head for floorplan placement.

    Args:
        d_model   (int): Input embedding dimension. Default: D_MODEL.
        hidden    (int): Shared MLP hidden size.    Default: 64.
        grid_size (int): Grid resolution (G).       Default: GRID_SIZE.
    """

    def __init__(
        self,
        d_model:   int = D_MODEL,
        hidden:    int = 64,
        grid_size: int = GRID_SIZE,
    ):
        super().__init__()
        self.G = grid_size
        G = grid_size
        self.shared     = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU())
        self.xy_head    = nn.Linear(hidden, G * G)
        self.ratio_head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_size(
        self,
        token_features: torch.Tensor,  # [..., 18]
        ratio_raw:      torch.Tensor,  # [..., 1]
    ):
        """Compute (w, h) tensors with fixed-shape overrides."""
        area_norm = token_features[..., 0].clamp(min=1e-12)
        ratio     = torch.exp(ratio_raw.squeeze(-1))
        pred_w    = (area_norm * ratio).clamp(min=1e-8).sqrt()
        pred_h    = (area_norm / ratio.clamp(min=1e-8)).clamp(min=1e-8).sqrt()

        is_fixed     = token_features[..., 2]
        is_preplaced = token_features[..., 3]
        has_fixed    = (is_fixed + is_preplaced) > 0
        target_w     = token_features[..., 4]
        target_h     = token_features[..., 5]

        pred_w = torch.where(has_fixed, target_w, pred_w)
        pred_h = torch.where(has_fixed, target_h, pred_h)
        return pred_w, pred_h

    def _build_positions(
        self,
        flat_idx:       torch.Tensor,  # [B, k] or [B, 1]
        token_features: torch.Tensor,  # [B, k, 18] or [B, 1, 18]
        ratio_raw:      torch.Tensor,  # [B, k, 1] or [B, 1, 1]
    ) -> torch.Tensor:
        """Convert flat grid indices to (x, y, w, h) with overrides."""
        G  = self.G
        gy = flat_idx // G
        gx = flat_idx % G
        x  = gx.float() / G
        y  = gy.float() / G

        pred_w, pred_h = self._decode_size(token_features, ratio_raw)

        is_preplaced = token_features[..., 3] > 0
        target_x     = token_features[..., 6]
        target_y     = token_features[..., 7]
        x = torch.where(is_preplaced, target_x, x)
        y = torch.where(is_preplaced, target_y, y)

        return torch.stack([x, y, pred_w, pred_h], dim=-1)

    def apply_occupancy_mask(
        self,
        logits:         torch.Tensor,  # [B, G*G]
        token_features: torch.Tensor,  # [B, 18]
        occupancy:      torch.Tensor,  # [B, G, G]
    ) -> torch.Tensor:
        """
        Set logits to -inf at positions where the block would overlap
        already-placed blocks.  Uses F.conv2d on the occupancy map.
        Returns: [B, G*G]
        """
        G      = self.G
        B      = logits.shape[0]
        device = logits.device

        # Estimate block footprint for masking (use ratio=1 approximation)
        area_norm = token_features[:, 0].clamp(min=1e-12)
        w_approx  = area_norm.sqrt()
        h_approx  = w_approx.clone()
        is_fixed     = token_features[:, 2]
        is_preplaced = token_features[:, 3]
        has_fixed    = (is_fixed + is_preplaced) > 0
        w_approx = torch.where(has_fixed, token_features[:, 4], w_approx)
        h_approx = torch.where(has_fixed, token_features[:, 5], h_approx)

        masked = logits.clone()
        for b in range(B):
            w_cells = max(1, min(G, int(math.ceil(w_approx[b].item() * G))))
            h_cells = max(1, min(G, int(math.ceil(h_approx[b].item() * G))))

            occ    = occupancy[b].unsqueeze(0).unsqueeze(0)              # [1,1,G,G]
            kernel = torch.ones(1, 1, h_cells, w_cells, device=device)
            conv   = F.conv2d(occ, kernel, padding=0)                    # [1,1,gh,gw]

            gh = G - h_cells + 1
            gw = G - w_cells + 1

            valid_mask = torch.zeros(G, G, dtype=torch.bool, device=device)
            if gh > 0 and gw > 0:
                valid_mask[:gh, :gw] = conv[0, 0] == 0

            masked[b] = masked[b].masked_fill(~valid_mask.view(G * G), float('-inf'))

        return masked

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,          # [B, k, d_model] or [B, 1, d_model]
        token_features: torch.Tensor,          # [B, k, 18]      or [B, 1, 18]
        occupancy:      torch.Tensor = None,   # [B, G, G]  — inference only
    ):
        """
        Training (occupancy=None):
            Returns (pred_positions [B,k,4], logits [B,k,G*G])

        Inference (occupancy=[B,G,G]):
            Returns pred_positions [B,1,4]
            Updates occupancy in-place.
        """
        h         = self.shared(x)          # [..., hidden]
        logits    = self.xy_head(h)          # [..., G*G]
        ratio_raw = self.ratio_head(h)       # [..., 1]

        if occupancy is None:
            # ── Training mode ──────────────────────────────────────
            flat_idx      = logits.argmax(dim=-1)                    # [B, k]
            pred_positions = self._build_positions(flat_idx, token_features, ratio_raw)
            return pred_positions, logits

        else:
            # ── Inference mode (one step at a time) ────────────────
            G = self.G
            logits_step = logits[:, 0, :]          # [B, G*G]
            tf_step     = token_features[:, 0, :]  # [B, 18]

            logits_masked = self.apply_occupancy_mask(logits_step, tf_step, occupancy)

            # Fallback if entire grid is occupied
            if not torch.isfinite(logits_masked).any():
                logits_masked = logits_step

            flat_idx = logits_masked.argmax(dim=-1)  # [B]

            flat_idx_2d    = flat_idx.unsqueeze(1)   # [B, 1]
            pred_positions = self._build_positions(flat_idx_2d, token_features, ratio_raw)

            # Update occupancy in-place
            B = x.shape[0]
            for b in range(B):
                idx    = flat_idx[b].item()
                gy     = idx // G
                gx     = idx % G
                pw     = pred_positions[b, 0, 2].item()
                ph     = pred_positions[b, 0, 3].item()
                w_cells = max(1, min(G - gx, int(math.ceil(pw * G))))
                h_cells = max(1, min(G - gy, int(math.ceil(ph * G))))
                occupancy[b, gy:gy + h_cells, gx:gx + w_cells] = 1.0

            return pred_positions  # [B, 1, 4]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("regression_head.py  —  smoke test (DiscreteRegressionHead)")
    print("=" * 55)

    G = GRID_SIZE
    B, k = 2, 55
    x  = torch.randn(B, k, D_MODEL)
    tf = torch.zeros(B, k, 18)
    tf[:, :, 0] = 0.05   # area_target_norm
    tf[:, :, 1] = 1.0    # is_soft

    head = DiscreteRegressionHead()
    head.eval()

    # ── Training mode ─────────────────────────────────────────────
    print("\n[1] Training mode (occupancy=None)")
    with torch.no_grad():
        pred_pos, logits = head(x, tf)
    assert pred_pos.shape == (B, k, 4), f"Expected ({B},{k},4), got {pred_pos.shape}"
    assert logits.shape == (B, k, G * G), f"Expected ({B},{k},{G*G}), got {logits.shape}"
    print(f"  pred_positions : {tuple(pred_pos.shape)}")
    print(f"  logits         : {tuple(logits.shape)}")
    assert (pred_pos[..., 2] > 0).all(), "w must be > 0"
    assert (pred_pos[..., 3] > 0).all(), "h must be > 0"
    print("  w, h > 0       : OK")

    # ── Backward pass ─────────────────────────────────────────────
    print("\n[2] Backward pass")
    x_train = torch.randn(B, k, D_MODEL, requires_grad=False)
    pred_pos_t, logits_t = head(x_train, tf)
    logits_t.sum().backward()
    print("  Backward        : OK")
    head.zero_grad()

    # ── Preplaced override ────────────────────────────────────────
    print("\n[3] Preplaced override")
    tf_pre = tf.clone()
    tf_pre[:, 0, 3] = 1.0   # is_preplaced
    tf_pre[:, 0, 4] = 0.3   # target_w
    tf_pre[:, 0, 5] = 0.2   # target_h
    tf_pre[:, 0, 6] = 0.1   # target_x
    tf_pre[:, 0, 7] = 0.05  # target_y
    with torch.no_grad():
        pred_pre, _ = head(x, tf_pre)
    assert abs(pred_pre[0, 0, 0].item() - 0.1) < 1e-5,  "preplaced x mismatch"
    assert abs(pred_pre[0, 0, 1].item() - 0.05) < 1e-5, "preplaced y mismatch"
    assert abs(pred_pre[0, 0, 2].item() - 0.3) < 1e-5,  "preplaced w mismatch"
    assert abs(pred_pre[0, 0, 3].item() - 0.2) < 1e-5,  "preplaced h mismatch"
    print("  Override        : OK")

    # ── Inference mode with occupancy ────────────────────────────
    print("\n[4] Inference mode (occupancy=[B,G,G])")
    x_step  = torch.randn(B, 1, D_MODEL)
    tf_step = tf[:, :1, :]
    occupancy = torch.zeros(B, G, G)
    with torch.no_grad():
        pred_step = head(x_step, tf_step, occupancy)
    assert pred_step.shape == (B, 1, 4), f"Expected ({B},1,4), got {pred_step.shape}"
    print(f"  pred_step      : {tuple(pred_step.shape)}")
    print(f"  occupancy sum  : {occupancy.sum().item():.0f}  (should be > 0)")
    assert occupancy.sum() > 0, "Occupancy was not updated"
    print("  Occupancy updated : OK")

    total = sum(p.numel() for p in head.parameters())
    print(f"\nParameters: {total:,}")

    print("\nSmoke test passed.")
    print("=" * 55)
