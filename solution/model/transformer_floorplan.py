"""
Floorplan Transformer — full model.

Integrates all sub-modules into a single nn.Module:

    token_features [B, k, 18]
          │
          ▼
    TokenFeatureProjection      → [B, k, d_model]
          │
          ▼
    FloorplanEncoder            → [B, k, d_model]   (contextual embeddings)
          │ (cross-attention)
          ▼
    FloorplanDecoder            → [B, k, d_model]   (teacher forcing or autoregressive)
          │
          ▼
    RegressionHead              → [B, k, 4]         (x, y, w, h, normalised)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # solution/
sys.path.insert(0, str(Path(__file__).resolve().parent))           # solution/model/
from config import D_MODEL, RAW_FEATURE_DIM, DROPOUT

from token_feature   import TokenFeatureProjection
from encoder         import FloorplanEncoder
from decoder         import FloorplanDecoder
from regression_head import RegressionHead


class TransformerFloorplan(nn.Module):
    """
    End-to-end Floorplan Transformer.

    Args:
        raw_dim         (int):   Raw token feature dim.     Default: RAW_FEATURE_DIM (18).
        d_model         (int):   Embedding dimension.       Default: D_MODEL (256).
        dropout         (float): Dropout probability.       Default: DROPOUT (0.1).

    All other hyper-parameters (num_layers, num_heads, …) are inherited
    from the defaults in each sub-module, which read from config.py.
    """

    def __init__(
        self,
        raw_dim: int   = RAW_FEATURE_DIM,
        d_model: int   = D_MODEL,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.token_proj  = TokenFeatureProjection(raw_dim, d_model, dropout)
        self.encoder     = FloorplanEncoder()
        self.decoder     = FloorplanDecoder()
        self.head        = RegressionHead(d_model)

    def forward(
        self,
        token_features:   torch.Tensor,          # [B, k, 18]
        w_int:            torch.Tensor,          # [B, k, k]
        gt_positions:     torch.Tensor = None,   # [B, k, 4]  — required if teacher_forcing
        key_padding_mask: torch.Tensor = None,   # [B, k] bool  (True = padding)
        teacher_forcing:  bool         = True,
    ) -> torch.Tensor:
        """
        Returns:
            pred_positions  [B, k, 4]   predicted (x, y, w, h), normalised scale
        """
        # 1. Project raw features to d_model
        tokens = self.token_proj(token_features)                     # [B, k, d_model]

        # 2. Encode with netlist attention bias
        enc_out = self.encoder(tokens, w_int, key_padding_mask)      # [B, k, d_model]

        # 3. Decode
        dec_out = self.decoder(
            enc_out,
            gt_positions=gt_positions,
            teacher_forcing=teacher_forcing,
            regression_head=self.head if not teacher_forcing else None,
            token_features=token_features if not teacher_forcing else None,
        )                                                            # [B, k, d_model]

        # 4. Regress to (x, y, w, h)
        return self.head(dec_out, token_features)                    # [B, k, 4]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("transformer_floorplan.py  —  smoke test")
    print("=" * 60)

    B, k = 2, 55
    token_features = torch.randn(B, k, RAW_FEATURE_DIM)
    w_int          = torch.rand(B, k, k)
    w_int          = (w_int + w_int.transpose(1, 2)) / 2   # symmetrise
    gt_positions   = torch.rand(B, k, 4)

    model = TransformerFloorplan()

    # ── Parameter count ───────────────────────────────────────────
    total = sum(p.numel() for p in model.parameters())
    breakdown = {
        "token_proj" : sum(p.numel() for p in model.token_proj.parameters()),
        "encoder"    : sum(p.numel() for p in model.encoder.parameters()),
        "decoder"    : sum(p.numel() for p in model.decoder.parameters()),
        "head"       : sum(p.numel() for p in model.head.parameters()),
    }
    print(f"\nParameter breakdown:")
    for name, n in breakdown.items():
        print(f"  {name:12s}: {n:>10,}")
    print(f"  {'TOTAL':12s}: {total:>10,}")

    # ── Teacher forcing ───────────────────────────────────────────
    print("\n[1] Teacher forcing (training mode)")
    model.train()
    out_tf = model(token_features, w_int, gt_positions=gt_positions, teacher_forcing=True)
    print(f"  Input  token_features : {tuple(token_features.shape)}")
    print(f"  Input  w_int          : {tuple(w_int.shape)}")
    print(f"  Input  gt_positions   : {tuple(gt_positions.shape)}")
    print(f"  Output pred_positions : {tuple(out_tf.shape)}")
    assert out_tf.shape == (B, k, 4), f"Expected ({B},{k},4), got {tuple(out_tf.shape)}"

    # Verify backward pass
    out_tf.sum().backward()
    print(f"  Backward pass         : OK")
    model.zero_grad()

    # ── Autoregressive inference ──────────────────────────────────
    print("\n[2] Autoregressive inference mode")
    model.eval()
    with torch.no_grad():
        out_ar = model(token_features, w_int, teacher_forcing=False)
    print(f"  Input  token_features : {tuple(token_features.shape)}")
    print(f"  Input  w_int          : {tuple(w_int.shape)}")
    print(f"  Output pred_positions : {tuple(out_ar.shape)}")
    assert out_ar.shape == (B, k, 4), f"Expected ({B},{k},4), got {tuple(out_ar.shape)}"

    # ── Outputs should differ between the two modes ───────────────
    model.eval()
    with torch.no_grad():
        out_tf_eval = model(token_features, w_int, gt_positions=gt_positions, teacher_forcing=True)
    diff = (out_tf_eval - out_ar).abs().mean().item()
    print(f"\n  Mean absolute diff (TF vs AR): {diff:.6f}")
    assert diff > 0, "Teacher forcing and autoregressive outputs should differ"
    print(f"  Outputs differ: OK")

    print("\nSmoke test passed.")
    print("=" * 60)
