"""
Floorplan Transformer Encoder with Netlist Attention Bias.

Each encoder layer modifies the standard self-attention score as:
    Attention = softmax( QK^T / sqrt(d_h)  +  alpha * w_int ) * V

where alpha is a per-layer learnable scalar (nn.Parameter).

Architecture:
    FloorplanEncoder
      └─ 6 × NetlistEncoderLayer
              ├─ NetlistBiasAttention  (self-attn + additive w_int bias)
              └─ FFN  (Linear → ReLU → Dropout → Linear)

Both sub-layers use post-norm (Add & Norm), matching standard Transformer style.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    D_MODEL, NUM_HEADS, NUM_ENCODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT,
)


# =============================================================================
# 1. Multi-head self-attention with additive Netlist Attention Bias
# =============================================================================

class NetlistBiasAttention(nn.Module):
    """
    Multi-head self-attention where the attention logits are biased by
    the (learnable) scaled netlist adjacency matrix:

        scores = QK^T / sqrt(d_h)  +  alpha * w_int

    Args:
        d_model   (int):   Total embedding dimension.
        num_heads (int):   Number of attention heads.
        dropout   (float): Attention dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.d_model   = d_model
        self.dropout_p = dropout

        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable scalar coefficient for the netlist bias
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x:                torch.Tensor,          # [B, k, d_model]
        w_int:            torch.Tensor,          # [B, k, k]
        key_padding_mask: torch.Tensor = None,   # [B, k] bool, True = pad
    ) -> torch.Tensor:
        """
        Returns:
            [B, k, d_model]
        """
        B, seq, _ = x.shape
        H, Dh = self.num_heads, self.head_dim

        # Project and reshape to [B, H, seq, Dh]
        def _proj(linear, t):
            return linear(t).view(B, seq, H, Dh).transpose(1, 2)

        q = _proj(self.q_proj, x)
        k = _proj(self.k_proj, x)
        v = _proj(self.v_proj, x)

        # Additive bias: [B, k, k] → [B, 1, k, k] (broadcast over heads)
        attn_bias = self.alpha * w_int.unsqueeze(1)   # [B, 1, k, k]

        # Incorporate padding: masked positions get -inf so softmax → 0
        if key_padding_mask is not None:
            # [B, k] → [B, 1, 1, k]
            inf_mask = key_padding_mask.unsqueeze(1).unsqueeze(2).float()
            inf_mask = inf_mask * float('-inf')
            inf_mask = inf_mask.nan_to_num(nan=0.0)
            attn_bias = attn_bias + inf_mask              # [B, 1, k, k]

        dp = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=dp,
        )                                                # [B, H, seq, Dh]

        out = out.transpose(1, 2).contiguous().view(B, seq, self.d_model)
        return self.out_proj(out)


# =============================================================================
# 2. Single encoder layer
# =============================================================================

class NetlistEncoderLayer(nn.Module):
    """
    One Transformer Encoder layer with Netlist Attention Bias.

    Sub-layer layout (post-norm):
        x  →  [Self-Attn + Dropout]  →  Add  →  LayerNorm  →
           →  [FFN + Dropout]        →  Add  →  LayerNorm  →  output
    """

    def __init__(
        self,
        d_model:         int,
        num_heads:       int,
        dim_feedforward: int,
        dropout:         float,
    ):
        super().__init__()
        self.self_attn = NetlistBiasAttention(d_model, num_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x:                torch.Tensor,          # [B, k, d_model]
        w_int:            torch.Tensor,          # [B, k, k]
        key_padding_mask: torch.Tensor = None,   # [B, k] bool
    ) -> torch.Tensor:
        # Self-attention sub-layer
        x = self.norm1(x + self.dropout(
            self.self_attn(x, w_int, key_padding_mask)
        ))
        # FFN sub-layer
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x                                 # [B, k, d_model]


# =============================================================================
# 3. Full encoder (6 stacked layers)
# =============================================================================

class FloorplanEncoder(nn.Module):
    """
    6-layer Transformer Encoder with per-layer Netlist Attention Bias.

    Args:
        num_layers      (int):   Number of encoder layers. Default: NUM_ENCODER_LAYERS (6).
        d_model         (int):   Embedding dimension. Default: D_MODEL (256).
        num_heads       (int):   Attention heads. Default: NUM_HEADS (8).
        dim_feedforward (int):   FFN hidden dim. Default: DIM_FEEDFORWARD (1024).
        dropout         (float): Dropout. Default: DROPOUT (0.1).
    """

    def __init__(
        self,
        num_layers:      int   = NUM_ENCODER_LAYERS,
        d_model:         int   = D_MODEL,
        num_heads:       int   = NUM_HEADS,
        dim_feedforward: int   = DIM_FEEDFORWARD,
        dropout:         float = DROPOUT,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            NetlistEncoderLayer(d_model, num_heads, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        x:                torch.Tensor,          # [B, k, d_model]
        w_int:            torch.Tensor,          # [B, k, k]
        key_padding_mask: torch.Tensor = None,   # [B, k] bool  (True = pad)
    ) -> torch.Tensor:
        """
        Returns:
            [B, k, d_model]  contextual block embeddings
        """
        for layer in self.layers:
            x = layer(x, w_int, key_padding_mask)
        return x


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("encoder.py  —  smoke test")
    print("=" * 55)

    B, k = 2, 55
    x     = torch.randn(B, k, D_MODEL)
    w_int = torch.rand(B, k, k)
    # Symmetrise (as real data would be)
    w_int = (w_int + w_int.transpose(1, 2)) / 2

    encoder = FloorplanEncoder()
    encoder.eval()

    with torch.no_grad():
        out = encoder(x, w_int)

    print(f"\nInput  shape : {tuple(x.shape)}")
    print(f"w_int  shape : {tuple(w_int.shape)}")
    print(f"Output shape : {tuple(out.shape)}")
    assert out.shape == (B, k, D_MODEL), \
        f"Expected ({B}, {k}, {D_MODEL}), got {tuple(out.shape)}"

    # Alpha values per layer
    print("\nalpha (per layer):")
    for i, layer in enumerate(encoder.layers):
        print(f"  layer {i}: alpha = {layer.self_attn.alpha.item():.4f}")

    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Verify gradient flows through alpha
    encoder.train()
    out = encoder(x, w_int)
    out.sum().backward()
    alpha_grads = [
        layer.self_attn.alpha.grad.item()
        for layer in encoder.layers
    ]
    assert all(g is not None for g in alpha_grads), "alpha gradient is None"
    print(f"alpha gradients (all non-None): {[f'{g:.4f}' for g in alpha_grads]}")

    print("\nSmoke test passed.")
    print("=" * 55)
