"""
Floorplan Transformer Decoder.

Autoregressively generates the decoder hidden state for each block's
(x, y, w, h) placement.  A separate regression head (regression_head.py)
maps each hidden state to the final 4-dim output.

Decoder input token at time step t:
    t == 0  →  SOS token  (nn.Parameter, learnable, shape [1, 1, d_model])
    t  > 0  →  coord_proj( position[t-1] ) + block_id_embed( t-1 )
                  where position[t-1] is either GT (teacher forcing) or
                  the prediction from the previous step (inference).

Two forward modes
─────────────────
teacher_forcing=True   (training)
    Input : encoder_output [B, k, d_model]
            gt_positions   [B, k, 4]   (x,y,w,h), normalised
    Output: [B, k, d_model]
    Uses full parallel computation with a causal mask.

teacher_forcing=False  (inference)
    Input : encoder_output  [B, k, d_model]
            regression_head  nn.Module  [B,1,d_model] → [B,1,4]
    Output: [B, k, d_model]
    Iterates k steps; feeds each step's prediction as the next token.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    D_MODEL, NUM_HEADS, NUM_DECODER_LAYERS,
    DIM_FEEDFORWARD, DROPOUT, MAX_BLOCKS,
)


class FloorplanDecoder(nn.Module):
    """
    Transformer Decoder for autoregressive floorplan generation.

    Args:
        num_layers      (int):   Decoder layers.            Default: NUM_DECODER_LAYERS (6).
        d_model         (int):   Embedding dimension.       Default: D_MODEL (256).
        num_heads       (int):   Attention heads.           Default: NUM_HEADS (8).
        dim_feedforward (int):   FFN hidden dim.            Default: DIM_FEEDFORWARD (1024).
        dropout         (float): Dropout probability.       Default: DROPOUT (0.1).
        max_blocks      (int):   Max block count (for Embedding table size).
                                                            Default: MAX_BLOCKS (120).
    """

    def __init__(
        self,
        num_layers:      int   = NUM_DECODER_LAYERS,
        d_model:         int   = D_MODEL,
        num_heads:       int   = NUM_HEADS,
        dim_feedforward: int   = DIM_FEEDFORWARD,
        dropout:         float = DROPOUT,
        max_blocks:      int   = MAX_BLOCKS,
    ):
        super().__init__()
        self.d_model = d_model

        # t=0 input: learnable SOS token
        self.sos_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # t>0 input: project 4-dim position to d_model, then add block-id embedding
        self.coord_proj      = nn.Linear(4, d_model)
        self.block_id_embed  = nn.Embedding(max_blocks, d_model)

        # Transformer decoder stack
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=False,     # post-norm, matches encoder
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Additive float causal mask [seq_len, seq_len]:
            0   on and below the diagonal  (attend)
            -inf above the diagonal        (mask out)
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=device),
            diagonal=1,
        )

    def _build_tf_input(self, gt_positions: torch.Tensor) -> torch.Tensor:
        """
        Build teacher-forcing input sequence of length k:
            position 0 : SOS
            position j : coord_proj( GT[j-1] ) + block_id_embed( j-1 )

        gt_positions : [B, k, 4]
        Returns      : [B, k, d_model]
        """
        B, k, _ = gt_positions.shape
        device = gt_positions.device

        # Tokens for positions 1 .. k-1  (representing already-placed blocks 0 .. k-2)
        ids = torch.arange(k - 1, device=device)           # [k-1]
        coord_tokens = (
            self.coord_proj(gt_positions[:, :-1, :])        # [B, k-1, d_model]
            + self.block_id_embed(ids)                      # broadcast over batch
        )

        sos = self.sos_token.expand(B, 1, -1)               # [B, 1, d_model]
        return torch.cat([sos, coord_tokens], dim=1)        # [B, k, d_model]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        encoder_output:  torch.Tensor,          # [B, k, d_model]
        gt_positions:    torch.Tensor = None,   # [B, k, 4]  — teacher forcing only
        teacher_forcing: bool         = True,
        regression_head: nn.Module    = None,   # [B,1,d_model] → [B,1,4] — inference only
        token_features:  torch.Tensor = None,   # [B, k, 18]  — inference only
    ) -> torch.Tensor:
        """
        Returns:
            [B, k, d_model]  decoder hidden states (input to regression head)
        """
        B, k, _ = encoder_output.shape
        device   = encoder_output.device

        # ── Teacher forcing (training) ──────────────────────────────
        if teacher_forcing:
            assert gt_positions is not None, \
                "gt_positions is required when teacher_forcing=True"

            tgt         = self._build_tf_input(gt_positions)       # [B, k, d_model]
            causal_mask = self._causal_mask(k, device)
            return self.decoder(
                tgt, encoder_output,
                tgt_mask=causal_mask,
            )                                                       # [B, k, d_model]

        # ── Autoregressive inference ────────────────────────────────
        else:
            assert regression_head is not None, \
                "regression_head is required when teacher_forcing=False"

            sos        = self.sos_token.expand(B, 1, -1)            # [B, 1, d_model]
            tgt_tokens = [sos]
            dec_outputs = []

            for t in range(k):
                tgt         = torch.cat(tgt_tokens, dim=1)          # [B, t+1, d_model]
                causal_mask = self._causal_mask(t + 1, device)

                dec_out = self.decoder(
                    tgt, encoder_output,
                    tgt_mask=causal_mask,
                )                                                    # [B, t+1, d_model]
                last = dec_out[:, -1:, :]                            # [B, 1, d_model]
                dec_outputs.append(last)

                if t < k - 1:
                    tf_t     = token_features[:, t:t+1, :]           # [B, 1, 18]
                    pred_pos = regression_head(last, tf_t)           # [B, 1, 4]
                    block_id   = torch.tensor([t], device=device)    # [1]
                    next_token = (
                        self.coord_proj(pred_pos)
                        + self.block_id_embed(block_id)
                    )                                                # [B, 1, d_model]
                    tgt_tokens.append(next_token)

            return torch.cat(dec_outputs, dim=1)                    # [B, k, d_model]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("decoder.py  —  smoke test")
    print("=" * 60)

    B, k = 2, 55

    enc_out      = torch.randn(B, k, D_MODEL)
    gt_positions = torch.rand(B, k, 4)          # normalised (x, y, w, h)

    decoder = FloorplanDecoder()

    # ── Teacher forcing ───────────────────────────────────────────
    print("\n[1] Teacher forcing (training mode)")
    decoder.train()
    out_tf = decoder(enc_out, gt_positions=gt_positions, teacher_forcing=True)
    print(f"  encoder_output shape : {tuple(enc_out.shape)}")
    print(f"  gt_positions   shape : {tuple(gt_positions.shape)}")
    print(f"  output         shape : {tuple(out_tf.shape)}")
    assert out_tf.shape == (B, k, D_MODEL), \
        f"Expected ({B}, {k}, {D_MODEL}), got {tuple(out_tf.shape)}"

    # Verify gradients flow to SOS token and coord_proj
    out_tf.sum().backward()
    assert decoder.sos_token.grad is not None, "No gradient to sos_token"
    print(f"  sos_token grad norm  : {decoder.sos_token.grad.norm().item():.4f}")
    decoder.zero_grad()

    # ── Autoregressive inference ──────────────────────────────────
    print("\n[2] Autoregressive inference mode")

    # Minimal dummy regression head: Linear(d_model, 4)
    dummy_head = nn.Linear(D_MODEL, 4)

    decoder.eval()
    with torch.no_grad():
        out_ar = decoder(
            enc_out,
            teacher_forcing=False,
            regression_head=dummy_head,
        )
    print(f"  encoder_output shape : {tuple(enc_out.shape)}")
    print(f"  output         shape : {tuple(out_ar.shape)}")
    assert out_ar.shape == (B, k, D_MODEL), \
        f"Expected ({B}, {k}, {D_MODEL}), got {tuple(out_ar.shape)}"

    # ── Parameter summary ─────────────────────────────────────────
    total = sum(p.numel() for p in decoder.parameters())
    print(f"\n  SOS token shape      : {tuple(decoder.sos_token.shape)}")
    print(f"  block_id_embed size  : {decoder.block_id_embed.num_embeddings} × {D_MODEL}")
    print(f"  Total parameters     : {total:,}")

    print("\nSmoke test passed.")
    print("=" * 60)
