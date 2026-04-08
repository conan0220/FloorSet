"""
Diagnostic script: one batch forward pass with detailed print.
Does NOT modify any existing source file.
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn

_SOLUTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _SOLUTION_DIR.parent
sys.path.insert(0, str(_SOLUTION_DIR))
sys.path.insert(0, str(_SOLUTION_DIR / "model"))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))

from config import (
    LAMBDA_WIRELENGTH, LAMBDA_AREA,
    LAMBDA_GROUPING, LAMBDA_MIB, LAMBDA_BOUNDARY, LAMBDA_OVERLAP,
)
from data.floorset_loader import preprocess_sample, get_training_dataloader
from model.transformer_floorplan import TransformerFloorplan
from loss.wirelength_loss import wirelength_loss
from loss.area_loss       import area_loss
from loss.violation_loss  import violation_loss
from train import (
    preprocess_raw_batch,
    collate_samples,
    _build_w_int_unnorm,
)

def tensor_stats(name, t):
    t_f = t.float()
    print(f"  {name:30s}  min={t_f.min().item():+.4e}  max={t_f.max().item():+.4e}"
          f"  mean={t_f.mean().item():+.4e}"
          f"  nan={torch.isnan(t_f).any().item()}"
          f"  inf={torch.isinf(t_f).any().item()}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}\n")

    loader = get_training_dataloader(batch_size=2, num_samples=2, shuffle=False)

    raw_batch = next(iter(loader))
    samples   = preprocess_raw_batch(raw_batch)
    batch     = collate_samples(samples)

    # ── Scalars ────────────────────────────────────────────────────────────
    print("=== Scalar baselines / canvas ===")
    tensor_stats("canvas_refs",     batch["canvas_refs"])
    tensor_stats("hpwl_baselines",  batch["hpwl_baselines"])
    tensor_stats("area_baselines",  batch["area_baselines"])
    print()

    # ── Model forward ──────────────────────────────────────────────────────
    model = TransformerFloorplan().to(device)
    model.eval()

    tf   = batch["token_features"].to(device)
    wi   = batch["w_int"].to(device)
    gtn  = batch["gt_positions_norm"].to(device)
    wiu  = batch["w_int_unnorm"].to(device)
    cons = batch["constraints_sorted"].to(device)
    kpm  = batch["key_padding_mask"].to(device)
    p2b  = batch["p2b_conn"].to(device)
    pins = batch["pins_pos"].to(device)
    crefs = batch["canvas_refs"].to(device)
    hbase = batch["hpwl_baselines"].to(device)
    abase = batch["area_baselines"].to(device)

    with torch.no_grad():
        pred_norm = model(tf, wi, gt_positions=gtn,
                          key_padding_mask=kpm, teacher_forcing=True)

    valid = (~kpm).float().unsqueeze(-1)
    pred_norm = pred_norm * valid

    pred_raw = pred_norm * crefs.view(-1, 1, 1)

    print("=== pred_positions (norm, after valid mask) ===")
    tensor_stats("pred_norm", pred_norm)
    print()
    print("=== pred_positions (raw = norm * canvas_ref) ===")
    tensor_stats("pred_raw",  pred_raw)
    print()

    # ── GT / token features sanity ─────────────────────────────────────────
    print("=== GT & features ===")
    tensor_stats("gt_positions_norm",    batch["gt_positions_norm"])
    tensor_stats("token_features",       batch["token_features"])
    tensor_stats("w_int (norm)",         batch["w_int"])
    tensor_stats("w_int_unnorm",         batch["w_int_unnorm"])
    tensor_stats("p2b_conn[:,2] (wt)",   batch["p2b_conn"][:, :, 2])
    tensor_stats("pins_pos",             batch["pins_pos"])
    print()

    # ── Individual losses ──────────────────────────────────────────────────
    print("=== Loss components ===")
    with torch.no_grad():
        l_wl    = wirelength_loss(pred_raw, wiu, pins, p2b, hbase)
        l_area  = area_loss(pred_raw, abase)
        l_grouping, l_mib, l_boundary, l_overlap = violation_loss(pred_norm, cons)

    print(f"  wirelength = {l_wl.item():+.6f}")
    print(f"  area       = {l_area.item():+.6f}")
    print(f"  grouping   = {l_grouping.item():+.6f}")
    print(f"  mib        = {l_mib.item():+.6f}")
    print(f"  boundary   = {l_boundary.item():+.6f}")
    print(f"  overlap    = {l_overlap.item():+.6f}")

    total = (LAMBDA_WIRELENGTH * l_wl
             + LAMBDA_AREA       * l_area
             + LAMBDA_GROUPING   * l_grouping
             + LAMBDA_MIB        * l_mib
             + LAMBDA_BOUNDARY   * l_boundary
             + LAMBDA_OVERLAP    * l_overlap)
    print(f"  total      = {total.item():+.6f}")
    print(f"  total finite? {torch.isfinite(total).item()}")

if __name__ == "__main__":
    main()
