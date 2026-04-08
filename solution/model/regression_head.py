"""
Continuous Regression Head.

Maps Decoder hidden states directly to continuous block placements (x, y, w, h).

Architecture:
    shared:     Linear(d_model, hidden) → ReLU
    xy_head:    Linear(hidden, 2)       → sigmoid → (x, y) ∈ [0, 1]
    ratio_head: Linear(hidden + 2, 1)   → clamp log_ratio → ratio = w/h

Size decoding (guarantees w * h == area_target):
    ratio = exp(clamp(ratio_raw, log(MIN_RATIO), log(MAX_RATIO)))
    w = sqrt(area_norm * ratio)
    h = sqrt(area_norm / ratio)

Overrides for fixed / preplaced blocks:
    is_fixed or is_preplaced  → w, h replaced with target_w, target_h
    is_preplaced              → x, y replaced with target_x, target_y

forward() modes
───────────────
placed_blocks=None  (training / teacher-forcing):
    x:              [B, k, d_model]
    token_features: [B, k, 21]
    Returns: pred_positions [B, k, 4]

placed_blocks=list  (autoregressive inference):
    x:              [B, 1, d_model]
    token_features: [B, 1, 21]
    Returns: pred_positions [B, 1, 4]
    Side-effect: appends newly placed block to each placed_blocks[b].

Token feature layout (21 dims):
    [0]    area_target_norm
    [1]    is_soft
    [2]    is_fixed_shape
    [3]    is_preplaced
    [4-5]  target_w, target_h (normalised)
    [6-7]  target_x, target_y (normalised)
    [8-15] boundary_type one-hot
    [16]   is_mib
    [17-20] cluster_group one-hot (groups 1-4; all-zero = no cluster)
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import D_MODEL, MIN_RATIO, MAX_RATIO


# =============================================================================
# Non-overlap placement search
# =============================================================================

def _has_overlap(x: float, y: float, w: float, h: float, placed_blocks: list) -> bool:
    # Use a tight threshold: touching blocks (overlap == 0) must not be flagged.
    # 1e-9 filters genuine gaps while ignoring IEEE-754 rounding noise.
    _TOL = 1e-9
    for (px, py, pw, ph) in placed_blocks:
        if (min(x + w, px + pw) - max(x, px) > _TOL and
                min(y + h, py + ph) - max(y, py) > _TOL):
            return True
    return False


def find_nonoverlap_position(
    x_pred:       float,
    y_pred:       float,
    w:            float,
    h:            float,
    placed_blocks: list,   # list of (px, py, pw, ph) already placed
):
    """
    Find the position closest to (x_pred, y_pred) that does not overlap
    any already-placed block.  Canvas boundary is not enforced — the model
    is free to place blocks outside [0, 1] if needed.

    Uses boundary-candidate search: any feasible placement can be compacted
    until x touches {0} ∪ {px_i + pw_i} and y touches {0} ∪ {py_i + ph_i},
    so this set of candidates is guaranteed to contain a valid position
    whenever any exists.

    Returns: (x, y, is_fallback)
        is_fallback=True only when no non-overlapping position was found
        among all candidates (should not occur with infinite space).
    """
    # Candidate x-coordinates: origin + right edges of placed blocks.
    # A small epsilon gap avoids IEEE-754 rounding artifacts that cause
    # nominally-touching blocks to appear overlapping after de-normalisation
    # (canvas_ref * 100-200 amplifies ~3.8e-6 normalised errors above the
    # contest threshold of 1e-6 raw pixels).
    _EPS = 1e-6
    x_cands = [0.0] + [px + pw + _EPS for (px, py, pw, ph) in placed_blocks]
    # Candidate y-coordinates: origin + top edges of placed blocks
    y_cands = [0.0] + [py + ph + _EPS for (px, py, pw, ph) in placed_blocks]

    # Also include the predicted position itself
    x_cands.append(x_pred)
    y_cands.append(y_pred)

    # Sort all (x, y) pairs by Euclidean distance to predicted position
    candidates = sorted(
        [(x, y) for x in x_cands for y in y_cands],
        key=lambda p: (p[0] - x_pred) ** 2 + (p[1] - y_pred) ** 2,
    )

    for (x, y) in candidates:
        if not _has_overlap(x, y, w, h, placed_blocks):
            return x, y, False

    # Should not reach here — return predicted position as fallback
    return x_pred, y_pred, True


# =============================================================================
# ContinuousRegressionHead
# =============================================================================

class ContinuousRegressionHead(nn.Module):
    """
    Continuous regression head for floorplan placement.

    Directly predicts (x, y) via sigmoid and ratio via a clamped linear
    layer, then derives (w, h) from ratio and target area.

    Args:
        d_model   (int):   Input embedding dimension.  Default: D_MODEL.
        hidden    (int):   Shared MLP hidden size.     Default: 64.
        min_ratio (float): Minimum w/h ratio.          Default: MIN_RATIO.
        max_ratio (float): Maximum w/h ratio.          Default: MAX_RATIO.
    """

    def __init__(
        self,
        d_model:   int   = D_MODEL,
        hidden:    int   = 64,
        min_ratio: float = MIN_RATIO,
        max_ratio: float = MAX_RATIO,
    ):
        super().__init__()
        self._log_min = math.log(min_ratio)
        self._log_max = math.log(max_ratio)

        self.shared     = nn.Sequential(nn.Linear(d_model, hidden), nn.ReLU())
        self.xy_head    = nn.Linear(hidden, 2)         # → (x, y) before sigmoid
        self.ratio_head = nn.Linear(hidden + 2, 1)     # +2: conditioned on (x, y)
        self._fallback_count = 0

        # Initialise ratio_head near-zero → ratio ≈ 1 (square) at init
        nn.init.xavier_uniform_(self.ratio_head.weight, gain=0.01)
        nn.init.zeros_(self.ratio_head.bias)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _decode_ratio(
        self,
        h:  torch.Tensor,  # [..., hidden]
        x:  torch.Tensor,  # [...] normalised x
        y:  torch.Tensor,  # [...] normalised y
    ) -> torch.Tensor:
        """Predict w/h ratio clamped to [MIN_RATIO, MAX_RATIO]."""
        xy        = torch.stack([x, y], dim=-1)                      # [..., 2]
        ratio_raw = self.ratio_head(torch.cat([h, xy], dim=-1))      # [..., 1]
        log_ratio = ratio_raw.squeeze(-1).clamp(self._log_min, self._log_max)
        return torch.exp(log_ratio)                                   # [...]

    def _decode_size(
        self,
        token_features: torch.Tensor,  # [..., 21]
        ratio:          torch.Tensor,  # [...]
    ):
        """Compute (w, h) from ratio + area, with fixed-shape overrides."""
        area_norm = token_features[..., 0].clamp(min=1e-12)
        pred_w    = (area_norm * ratio).clamp(min=1e-8).sqrt()
        pred_h    = (area_norm / ratio.clamp(min=1e-8)).clamp(min=1e-8).sqrt()

        is_fixed     = token_features[..., 2]
        is_preplaced = token_features[..., 3]
        has_fixed    = (is_fixed + is_preplaced) > 0
        pred_w = torch.where(has_fixed, token_features[..., 4], pred_w)
        pred_h = torch.where(has_fixed, token_features[..., 5], pred_h)
        return pred_w, pred_h

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x:              torch.Tensor,        # [B, k, d_model] or [B, 1, d_model]
        token_features: torch.Tensor,        # [B, k, 21]      or [B, 1, 21]
        placed_blocks:  list = None,         # list[list[tuple]] — inference only
    ):
        """
        Training (placed_blocks=None):
            Returns pred_positions [B, k, 4]

        Inference (placed_blocks=list):
            Returns pred_positions [B, 1, 4]
            Updates placed_blocks in-place.
        """
        h = self.shared(x)                           # [..., hidden]

        # Step 1: predict (x, y)
        xy_raw = self.xy_head(h)                     # [..., 2]
        pred_x = torch.sigmoid(xy_raw[..., 0])       # [...] ∈ (0, 1)
        pred_y = torch.sigmoid(xy_raw[..., 1])       # [...] ∈ (0, 1)

        # Preplaced blocks: override predicted (x, y) with hard-constrained coords
        is_preplaced = token_features[..., 3] > 0
        pred_x = torch.where(is_preplaced, token_features[..., 6], pred_x)
        pred_y = torch.where(is_preplaced, token_features[..., 7], pred_y)

        # Step 2: predict ratio conditioned on (x, y), then derive (w, h)
        ratio          = self._decode_ratio(h, pred_x, pred_y)
        pred_w, pred_h = self._decode_size(token_features, ratio)

        if placed_blocks is None:
            # ── Training mode ──────────────────────────────────────────
            return torch.stack([pred_x, pred_y, pred_w, pred_h], dim=-1)

        else:
            # ── Inference mode: non-overlap search ────────────────────
            B      = x.shape[0]
            device = x.device

            final_x = pred_x[:, 0].clone()   # [B]
            final_y = pred_y[:, 0].clone()   # [B]
            w       = pred_w[:, 0]            # [B]
            h_      = pred_h[:, 0]            # [B]

            for b in range(B):
                is_pre = token_features[b, 0, 3].item() > 0
                if is_pre:
                    # Preplaced blocks have already been inserted into
                    # placed_blocks by TransformerFloorplan before the
                    # autoregressive loop — do NOT append again.
                    pass
                else:
                    xn, yn, is_fb = find_nonoverlap_position(
                        final_x[b].item(), final_y[b].item(),
                        w[b].item(), h_[b].item(),
                        placed_blocks[b],
                    )
                    if is_fb:
                        self._fallback_count += 1
                    final_x[b] = xn
                    final_y[b] = yn

                    placed_blocks[b].append((
                        final_x[b].item(), final_y[b].item(),
                        w[b].item(), h_[b].item(),
                    ))

            pred_positions = torch.stack(
                [final_x, final_y, w, h_], dim=-1
            ).unsqueeze(1)                    # [B, 1, 4]
            return pred_positions


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("regression_head.py  —  smoke test (ContinuousRegressionHead)")
    print("=" * 60)

    B, k = 2, 55
    x  = torch.randn(B, k, D_MODEL)
    tf = torch.zeros(B, k, 21)
    tf[:, :, 0] = 0.05   # area_target_norm
    tf[:, :, 1] = 1.0    # is_soft

    head = ContinuousRegressionHead()
    head.eval()

    # ── Training mode ─────────────────────────────────────────────
    print("\n[1] Training mode (placed_blocks=None)")
    with torch.no_grad():
        pred_pos = head(x, tf)
    assert pred_pos.shape == (B, k, 4), f"Expected ({B},{k},4), got {pred_pos.shape}"
    print(f"  pred_positions : {tuple(pred_pos.shape)}")
    assert (pred_pos[..., 0] > 0).all() and (pred_pos[..., 0] < 1).all(), "x out of (0,1)"
    assert (pred_pos[..., 1] > 0).all() and (pred_pos[..., 1] < 1).all(), "y out of (0,1)"
    assert (pred_pos[..., 2] > 0).all(), "w must be > 0"
    assert (pred_pos[..., 3] > 0).all(), "h must be > 0"
    print("  x, y in (0,1)  : OK")
    print("  w, h > 0        : OK")

    # ── Ratio clamp ───────────────────────────────────────────────
    print("\n[2] Ratio clamp check")
    ratio = pred_pos[..., 2] / pred_pos[..., 3]
    assert (ratio >= MIN_RATIO - 1e-5).all(), f"ratio below MIN_RATIO={MIN_RATIO}"
    assert (ratio <= MAX_RATIO + 1e-5).all(), f"ratio above MAX_RATIO={MAX_RATIO}"
    print(f"  ratio ∈ [{MIN_RATIO}, {MAX_RATIO}]: OK  (actual min={ratio.min():.3f}, max={ratio.max():.3f})")

    # ── Area conservation ─────────────────────────────────────────
    print("\n[3] Area conservation (w * h == area_target_norm)")
    area_pred = pred_pos[..., 2] * pred_pos[..., 3]
    area_gt   = tf[..., 0].clamp(min=1e-12)
    err = (area_pred - area_gt).abs().max().item()
    assert err < 1e-5, f"Area mismatch: max err={err}"
    print(f"  Max area error  : {err:.2e}  OK")

    # ── Backward pass ─────────────────────────────────────────────
    print("\n[4] Backward pass")
    x_train  = torch.randn(B, k, D_MODEL, requires_grad=False)
    pred_t   = head(x_train, tf)
    pred_t.sum().backward()
    print("  Backward        : OK")
    head.zero_grad()

    # ── Preplaced override ────────────────────────────────────────
    print("\n[5] Preplaced override")
    tf_pre = tf.clone()
    tf_pre[:, 0, 3] = 1.0   # is_preplaced
    tf_pre[:, 0, 4] = 0.3   # target_w
    tf_pre[:, 0, 5] = 0.2   # target_h
    tf_pre[:, 0, 6] = 0.1   # target_x
    tf_pre[:, 0, 7] = 0.05  # target_y
    with torch.no_grad():
        pred_pre = head(x, tf_pre)
    assert abs(pred_pre[0, 0, 0].item() - 0.1)  < 1e-5, "preplaced x mismatch"
    assert abs(pred_pre[0, 0, 1].item() - 0.05) < 1e-5, "preplaced y mismatch"
    assert abs(pred_pre[0, 0, 2].item() - 0.3)  < 1e-5, "preplaced w mismatch"
    assert abs(pred_pre[0, 0, 3].item() - 0.2)  < 1e-5, "preplaced h mismatch"
    print("  Override        : OK")

    # ── Inference mode with non-overlap search ────────────────────
    print("\n[6] Inference mode (placed_blocks=[])")
    x_step      = torch.randn(B, 1, D_MODEL)
    tf_step     = tf[:, :1, :]
    placed_blocks = [[] for _ in range(B)]
    with torch.no_grad():
        pred_step = head(x_step, tf_step, placed_blocks)
    assert pred_step.shape == (B, 1, 4), f"Expected ({B},1,4), got {pred_step.shape}"
    assert all(len(pb) == 1 for pb in placed_blocks), "placed_blocks not updated"
    print(f"  pred_step       : {tuple(pred_step.shape)}")
    print(f"  placed_blocks   : {len(placed_blocks[0])} block(s) registered  OK")

    # ── find_nonoverlap_position: guarantee no overlap ────────────
    print("\n[7] find_nonoverlap_position exhaustive test")
    placed = [(0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 0.5, 0.5), (0.0, 0.5, 0.5, 0.5)]
    x_n, y_n, is_fb = find_nonoverlap_position(0.0, 0.0, 0.5, 0.5, placed)
    assert not is_fb, "Should find valid position (bottom-right quadrant is free)"
    assert not _has_overlap(x_n, y_n, 0.5, 0.5, placed), "Returned position still overlaps"
    print(f"  Placed position : ({x_n:.3f}, {y_n:.3f})  is_fallback={is_fb}  OK")

    total = sum(p.numel() for p in head.parameters())
    print(f"\nParameters: {total:,}")
    print("\nSmoke test passed.")
    print("=" * 60)
