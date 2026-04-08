"""
Shared AR inference utility.

ar_inference() is the single authoritative path for autoregressive inference,
used by both train.py (_infer_and_save) and my_optimizer.py (solve()).

Design contract
───────────────
- Soft blocks must NOT see their GT target_w/h in token_features at inference
  time, regardless of how the preprocessed sample dict `s` was built.
  (Soft blocks may have non-zero dims 4-5 when `s` comes from the training
  cache, which was built with full fp_sol.)
- Fixed and preplaced blocks keep their target hints (dims 4-7), since these
  are known constraints, not GT leakage.
- Both train.py and solve() use this function so their results are identical.
"""

import torch


def ar_inference(
    model,           # TransformerFloorplan (eval mode, no_grad context expected)
    s:      dict,    # preprocessed sample dict from preprocess_sample()
    device: torch.device,
):
    """
    Run AR inference on one preprocessed sample dict.

    Soft-block hint zeroing
    ───────────────────────
    token_features dims 4-7 carry target_w/h/x/y.  For soft blocks these are
    always zeroed here, ensuring no GT leakage regardless of how `s` was built.

    Returns
    ───────
    pred_norm : Tensor [1, k, 4]  normalised (x,y,w,h) on `device`
    pred_raw  : Tensor [k, 4]     pixel-scale (x,y,w,h) in sorted block order, on CPU
    """
    k    = s["block_count"]
    tf   = s["token_features"][:k].clone()     # [k, RAW_FEATURE_DIM]
    cons = s["constraints_sorted"][:k]         # [k, 5]

    # Zero out target_w/h/x/y (dims 4-7) for soft blocks
    is_soft = (cons[:, 0] == 0) & (cons[:, 1] == 0)   # not fixed, not preplaced
    tf[is_soft, 4:8] = 0.0

    tf_ = tf.unsqueeze(0).to(device)                               # [1, k, dim]
    wi_ = s["w_int"][:k, :k].unsqueeze(0).to(device)              # [1, k, k]

    with torch.no_grad():
        pred_norm = model(tf_, wi_, teacher_forcing=False)         # [1, k, 4]

    pred_raw = (pred_norm[0] * s["canvas_ref"]).cpu()              # [k, 4]
    return pred_norm, pred_raw
