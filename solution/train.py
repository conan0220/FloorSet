"""
Training script for the Floorplan Transformer.

Usage:
    python solution/train.py               # full training
    python solution/train.py --smoke-test  # 2 batches × 2 epochs (quick check)
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_SOLUTION_DIR = Path(__file__).resolve().parent           # solution/
_REPO_ROOT    = _SOLUTION_DIR.parent                      # FloorSet/
sys.path.insert(0, str(_SOLUTION_DIR))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))

from config import (
    BATCH_SIZE, MAX_EPOCHS, PATIENCE, LEARNING_RATE, WARMUP_STEPS,
    GRAD_CLIP_NORM, LAMBDA_WIRELENGTH, LAMBDA_AREA,
    LAMBDA_GROUPING, LAMBDA_MIB, LAMBDA_BOUNDARY, LAMBDA_OVERLAP, LAMBDA_COORD,
    VALIDATE_EVERY, VIZ_BLOCK_SIZES, RAW_FEATURE_DIM,
    LOGS_DIR, CHECKPOINT_DIR, CONTEST_DIR,
    CACHE_DIR, CACHE_PRELOAD,
    LAMBDA_RATIO_REG, RATIO_REG_DECAY_EPOCHS,
)
from data.floorset_loader import (
    preprocess_sample,
    get_training_dataloader,
    get_cached_training_dataloader,
    build_w_int_unnorm,
    count_valid_blocks,
    compute_canvas_ref,
    sort_block_indices,
    build_sort_inv,
)
from model.transformer_floorplan import TransformerFloorplan
from loss.wirelength_loss import wirelength_loss
from loss.area_loss       import area_loss
from loss.violation_loss  import violation_loss
from loss.coord_loss      import coord_loss
from inference            import ar_inference


# =============================================================================
# Directory setup
# =============================================================================

def _make_dirs():
    for d in (LOGS_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Batch preprocessing
# =============================================================================

def preprocess_raw_batch(raw_batch: tuple) -> list:
    """
    Convert one raw batch from the official training DataLoader into
    a list of preprocessed sample dicts.  Also computes w_int_unnorm
    (unnormalised for use in wirelength_loss).
    """
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, metrics = raw_batch
    B = area_target.shape[0]

    samples = []
    for i in range(B):
        s = preprocess_sample(
            area_target[i], b2b_conn[i], p2b_conn[i],
            pins_pos[i], constraints[i], fp_sol[i], metrics[i],
        )
        s["w_int_unnorm"] = build_w_int_unnorm(
            b2b_conn[i], s["block_count"], s["sort_idx"]
        )
        samples.append(s)
    return samples


def collate_samples(samples: list) -> dict:
    """
    Pad a list of preprocessed sample dicts into tensors ready for the model.

    Padding convention:
        token_features / gt / constraints / w_int / w_int_unnorm : 0-filled
        p2b_conn                                                  : -1-filled
        key_padding_mask                                          : True = padding
    """
    B = len(samples)
    max_k    = max(s["block_count"]        for s in samples)
    max_p2b  = max(max(s["p2b_conn"].shape[0], 1)  for s in samples)
    max_pins = max(max(s["pins_pos"].shape[0], 1)  for s in samples)

    tf   = torch.zeros(B, max_k, RAW_FEATURE_DIM)
    wi   = torch.zeros(B, max_k, max_k)
    wiu  = torch.zeros(B, max_k, max_k)
    gtn  = torch.zeros(B, max_k, 4)
    gtr  = torch.zeros(B, max_k, 4)
    cons = torch.zeros(B, max_k, 5)
    kpm  = torch.ones(B, max_k, dtype=torch.bool)   # True = padding
    p2b  = torch.full((B, max_p2b, 3), -1.0)
    pins = torch.zeros(B, max_pins, 2)
    crefs  = torch.zeros(B)
    hbase  = torch.zeros(B)
    abase  = torch.zeros(B)

    for i, s in enumerate(samples):
        k  = s["block_count"]
        ep = s["p2b_conn"].shape[0]
        pp = s["pins_pos"].shape[0]

        tf[i, :k]       = s["token_features"]
        wi[i, :k, :k]   = s["w_int"]
        wiu[i, :k, :k]  = s["w_int_unnorm"]
        gtn[i, :k]      = s["gt_positions_norm"]
        gtr[i, :k]      = s["gt_positions_raw"]
        cons[i, :k]     = s["constraints_sorted"]
        kpm[i, :k]      = False
        if ep > 0:
            p2b[i, :ep] = s["p2b_conn"]
        if pp > 0:
            pins[i, :pp] = s["pins_pos"]
        crefs[i]        = s["canvas_ref"]
        hbase[i]        = s["hpwl_baseline"]
        abase[i]        = s["area_baseline"]

    return {
        "token_features":     tf,
        "w_int":              wi,
        "w_int_unnorm":       wiu,
        "gt_positions_norm":  gtn,
        "gt_positions_raw":   gtr,
        "constraints_sorted": cons,
        "key_padding_mask":   kpm,
        "p2b_conn":           p2b,
        "pins_pos":           pins,
        "canvas_refs":        crefs,
        "hpwl_baselines":     hbase,
        "area_baselines":     abase,
    }


# =============================================================================
# LR schedule: linear warmup + cosine decay
# =============================================================================

def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return max(1e-8, current_step / warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return lr_lambda


# =============================================================================
# Loss computation
# =============================================================================

def ratio_reg_weight(epoch: int) -> float:
    """Linear decay: LAMBDA_RATIO_REG at epoch 0 → 0 at RATIO_REG_DECAY_EPOCHS."""
    if epoch >= RATIO_REG_DECAY_EPOCHS:
        return 0.0
    return LAMBDA_RATIO_REG * (1.0 - epoch / RATIO_REG_DECAY_EPOCHS)


def compute_batch_loss(model, batch: dict, device: torch.device, epoch: int = 0):
    """
    Forward pass + compute all loss terms.

    Returns:
        total_loss  (scalar, differentiable)
        dict of individual scalar .item() values for logging
        pred_norm   [B, k, 4] for external use
    """
    tf   = batch["token_features"].to(device)
    wi   = batch["w_int"].to(device)
    wiu  = batch["w_int_unnorm"].to(device)
    gtn  = batch["gt_positions_norm"].to(device)
    cons = batch["constraints_sorted"].to(device)
    kpm  = batch["key_padding_mask"].to(device)
    p2b  = batch["p2b_conn"].to(device)
    pins = batch["pins_pos"].to(device)
    crefs = batch["canvas_refs"].to(device)    # [B]
    hbase = batch["hpwl_baselines"].to(device) # [B]
    abase = batch["area_baselines"].to(device) # [B]

    # Forward (teacher forcing) → pred_norm [B, k, 4]
    pred_norm = model(
        tf, wi,
        gt_positions=gtn,
        key_padding_mask=kpm,
        teacher_forcing=True,
    )

    # Zero out padded positions so they don't contribute to losses
    valid = (~kpm).float().unsqueeze(-1)       # [B, k, 1]
    pred_norm = pred_norm * valid

    # De-normalise for HPWL / area losses (raw pixel scale)
    pred_raw = pred_norm * crefs.view(-1, 1, 1)  # [B, k, 4]

    l_coord = coord_loss(pred_norm, gtn, cons, kpm)
    l_wl    = wirelength_loss(pred_raw, wiu, pins, p2b, hbase)
    l_area  = area_loss(pred_raw, abase)
    l_grouping, l_mib, l_boundary, l_overlap = violation_loss(pred_norm, cons)

    # Ratio regularisation: penalise log(w/h)² on valid (non-padded) blocks.
    # Weight decays linearly to 0 over RATIO_REG_DECAY_EPOCHS so physical losses
    # gradually take full control of block shape.
    lam_ratio = ratio_reg_weight(epoch)
    if lam_ratio > 0:
        pred_w = pred_norm[..., 2].clamp(min=1e-8)   # [B, k]
        pred_h = pred_norm[..., 3].clamp(min=1e-8)   # [B, k]
        log_ratio = torch.log(pred_w / pred_h)        # [B, k]
        valid_2d  = valid.squeeze(-1)                 # [B, k]
        l_ratio_reg = (log_ratio.pow(2) * valid_2d).sum() / valid_2d.sum().clamp(min=1)
    else:
        l_ratio_reg = pred_norm.new_zeros(1).squeeze()

    total = (LAMBDA_COORD       * l_coord
             + LAMBDA_WIRELENGTH * l_wl
             + LAMBDA_AREA       * l_area
             + LAMBDA_GROUPING   * l_grouping
             + LAMBDA_MIB        * l_mib
             + LAMBDA_BOUNDARY   * l_boundary
             + LAMBDA_OVERLAP    * l_overlap
             + lam_ratio         * l_ratio_reg)

    return total, {
        "total":      total.item(),
        "coord":      l_coord.item(),
        "wirelength": l_wl.item(),
        "area":       l_area.item(),
        "grouping":   l_grouping.item(),
        "mib":        l_mib.item(),
        "boundary":   l_boundary.item(),
        "overlap":    l_overlap.item(),
        "ratio_reg":  l_ratio_reg.item(),
        "lam_ratio":  lam_ratio,
    }, pred_norm


# =============================================================================
# Checkpointing
# =============================================================================

def save_checkpoint(model, optimizer, epoch: int, epoch_losses: list,
                    best_score: float, path: Path):
    torch.save({
        "epoch":        epoch,
        "model_state":  model.state_dict(),
        "opt_state":    optimizer.state_dict(),
        "epoch_losses": epoch_losses,
        "best_score":   best_score,
    }, path)


def load_checkpoint(path: Path, model, optimizer, device):
    """
    Load checkpoint from path.  Returns (epoch, epoch_losses, best_score).
    """
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model_state"])
    optimizer.load_state_dict(ck["opt_state"])
    return ck["epoch"], ck.get("epoch_losses", []), ck.get("best_score", float("inf"))


# =============================================================================
# Loss curve
# =============================================================================

def update_loss_plot(epoch_losses: list, save_path: Path):
    """Overwrite loss_curve.png with the latest epoch-level loss history."""
    epochs = list(range(1, len(epoch_losses) + 1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, epoch_losses, marker="o", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# =============================================================================
# Official validation (subprocess)
# =============================================================================

def validate_with_solve(
    model,
    optimizer_state,
    epoch:      int,
    epoch_losses: list,
    best_score: float,
    log_file:   Path,
    dry_run:    bool = False,
) -> float:
    """
    Run the official ICCAD evaluator as a subprocess and return Avg Cost.

    Always evaluates the CURRENT model weights by temporarily writing them to
    best.pt (which my_optimizer.py prefers over latest.pt).  If the score does
    not improve, the previous best.pt is restored from a backup so that the
    on-disk best.pt always reflects the truly best model.

    Lower score is better (feasible solutions have cost < 10.0).
    Returns float("inf") on failure.
    """
    if dry_run:
        print(f"  [smoke-test] official-val epoch={epoch} — skipped")
        return float("inf")

    import shutil
    import subprocess

    best_ck   = CHECKPOINT_DIR / "best.pt"
    backup_ck = CHECKPOINT_DIR / "_best_backup.pt"

    # --- back up existing best.pt, then write current model as best.pt ---
    if best_ck.exists():
        shutil.copy2(best_ck, backup_ck)
    save_checkpoint(model, optimizer_state, epoch, epoch_losses,
                    best_score, best_ck)

    evaluator = _REPO_ROOT / "iccad2026contest" / "iccad2026_evaluate.py"
    opt_script = _REPO_ROOT / "iccad2026contest" / "my_optimizer.py"
    cmd = [sys.executable, str(evaluator), "--evaluate", str(opt_script)]
    print(f"  [val] epoch={epoch}  running official evaluator …")

    # Stream subprocess output to terminal in real-time while also collecting
    # it for later parsing.
    lines = []
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr into stdout
            text=True,
            cwd=str(_REPO_ROOT / "iccad2026contest"),
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait(timeout=600)
        stdout = "".join(lines)
    except subprocess.TimeoutExpired:
        proc.kill()
        print("  [val] evaluator timed out — skipping")
        _restore_best_backup(best_ck, backup_ck)
        return float("inf")
    except Exception as e:
        print(f"  [val] evaluator error: {e} — skipping")
        _restore_best_backup(best_ck, backup_ck)
        return float("inf")

    # Parse "Avg Cost: X.XXXX"
    avg_cost = float("inf")
    for line in stdout.splitlines():
        if line.strip().startswith("Avg Cost:"):
            try:
                avg_cost = float(line.split(":")[1].strip())
            except ValueError:
                pass
            break

    if avg_cost == float("inf"):
        print("  [val] could not parse Avg Cost from evaluator output")
        if result.returncode != 0:
            print(f"  [val] stderr: {result.stderr[-500:]}")
        _restore_best_backup(best_ck, backup_ck)
    else:
        print(f"  [val] epoch={epoch}  Avg Cost={avg_cost:.4f}")
        if avg_cost >= best_score and backup_ck.exists():
            # Current model did not improve — restore previous best
            shutil.copy2(backup_ck, best_ck)
        if backup_ck.exists():
            backup_ck.unlink()

    with open(log_file, "a") as f:
        f.write(f"Epoch {epoch}: Avg Cost={avg_cost:.4f}\n")

    return avg_cost


def _restore_best_backup(best_ck: Path, backup_ck: Path):
    """Restore best.pt from backup if backup exists."""
    import shutil
    if backup_ck.exists():
        shutil.copy2(backup_ck, best_ck)
        backup_ck.unlink()


# =============================================================================
# Visualization
# =============================================================================

def _polygons_to_fp_sol(polygons, block_count: int) -> torch.Tensor:
    """
    Convert polygon vertex list (GT label) into fp_sol [block_count, 4]
    in the [w, h, x, y] format expected by preprocess_sample.
    """
    fp = torch.zeros(block_count, 4)
    for i in range(block_count):
        verts = polygons[i]
        if isinstance(verts, torch.Tensor):
            valid = verts[verts[:, 0] != -1]
        else:
            valid = torch.tensor(verts)
            valid = valid[valid[:, 0] != -1]
        if valid.numel() > 0:
            xy_min = valid.min(0).values
            xy_max = valid.max(0).values
            w = (xy_max[0] - xy_min[0]).item()
            h = (xy_max[1] - xy_min[1]).item()
            x = xy_min[0].item()
            y = xy_min[1].item()
            fp[i] = torch.tensor([w, h, x, y])
    return fp


def _compute_viz_loss(s: dict, pred_norm: torch.Tensor, device) -> dict:
    """
    Compute all loss terms for one sample (batch size = 1).
    pred_norm: [1, k, 4] on device.
    Returns dict of scalar .item() values.
    """
    k   = s["block_count"]
    ref = s["canvas_ref"]

    gtn  = s["gt_positions_norm"][:k].unsqueeze(0).to(device)   # [1, k, 4]
    cons = s["constraints_sorted"][:k].unsqueeze(0).to(device)  # [1, k, 5]
    p2b  = s["p2b_conn"].unsqueeze(0).to(device)                 # [1, e, 3]
    pins = s["pins_pos"].unsqueeze(0).to(device)                  # [1, r, 2]

    hbase = torch.tensor([s["hpwl_baseline"]], device=device)
    abase = torch.tensor([s["area_baseline"]], device=device)

    # w_int_unnorm: use cached value if present, else compute on the fly
    if "w_int_unnorm" in s:
        wiu = s["w_int_unnorm"][:k, :k].unsqueeze(0).to(device)
    else:
        wiu = build_w_int_unnorm(
            s["b2b_conn"], k, s["sort_idx"]
        ).unsqueeze(0).to(device)

    pred_raw = pred_norm * ref   # [1, k, 4]

    kpm_viz = torch.zeros(1, k, dtype=torch.bool, device=device)

    with torch.no_grad():
        l_wl    = wirelength_loss(pred_raw, wiu, pins, p2b, hbase).item()
        l_area  = area_loss(pred_raw, abase).item()
        l_grouping, l_mib, l_boundary, l_overlap = violation_loss(pred_norm, cons)
        l_grouping  = l_grouping.item()
        l_mib       = l_mib.item()
        l_boundary  = l_boundary.item()
        l_overlap   = l_overlap.item()
        total = (LAMBDA_WIRELENGTH * l_wl
                 + LAMBDA_AREA     * l_area
                 + LAMBDA_GROUPING * l_grouping
                 + LAMBDA_MIB      * l_mib
                 + LAMBDA_BOUNDARY * l_boundary
                 + LAMBDA_OVERLAP  * l_overlap)

    return {
        "total":      total,
        "wirelength": l_wl,
        "area":       l_area,
        "grouping":   l_grouping,
        "mib":        l_mib,
        "boundary":   l_boundary,
        "overlap":    l_overlap,
    }




# =============================================================================
# Main training loop
# =============================================================================

def train(smoke_test: bool = False, num_shards: int | None = None):
    _make_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Smoke-test overrides ───────────────────────────────────────────────
    batch_size_eff  = 2      if smoke_test else BATCH_SIZE
    max_epochs_eff  = 2      if smoke_test else MAX_EPOCHS
    n_batches_cap   = 2      if smoke_test else None   # None = unlimited
    validate_every  = 5      if smoke_test else VALIDATE_EVERY
    patience_eff    = PATIENCE

    # ── Model & optimiser ─────────────────────────────────────────────────
    model = TransformerFloorplan().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # Total steps estimate for cosine decay (rough: 16k steps/epoch × max_epochs)
    est_steps_per_epoch = 16_000 if not smoke_test else n_batches_cap
    total_steps = WARMUP_STEPS + max_epochs_eff * est_steps_per_epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(WARMUP_STEPS, total_steps)
    )

    # ── Resume from checkpoint if available ───────────────────────────────
    latest_ck = CHECKPOINT_DIR / "latest.pt"
    start_epoch = 0
    epoch_losses: list = []
    best_score = float("inf")
    no_improve_count = 0

    if latest_ck.exists() and not smoke_test:
        start_epoch, epoch_losses, best_score = load_checkpoint(
            latest_ck, model, optimizer, device
        )
        print(f"Resumed from epoch {start_epoch}")

    # ── Data loader (prefer shard cache when available) ───────────────────
    _cache_ready = (CACHE_DIR / "meta.pt").exists() and not smoke_test
    if _cache_ready:
        shard_info = f"{num_shards} shards" if num_shards else "all shards"
        print(f"Cache detected at {CACHE_DIR} — using CachedShardIterableDataset"
              + f" ({shard_info})"
              + (" (preload into RAM)" if CACHE_PRELOAD else " (streaming from disk)"))
        train_loader = get_cached_training_dataloader(
            cache_dir=CACHE_DIR,
            batch_size=batch_size_eff,
            shuffle=True,
            num_workers=2,
            preload=CACHE_PRELOAD,
            num_shards=num_shards,
        )
        _use_cache = True
    else:
        if not smoke_test:
            print("No cache found — running preprocessing on-the-fly")
            print("Tip: run  python solution/data/build_cache.py  to build the cache")
        train_loader = get_training_dataloader(
            batch_size=batch_size_eff,
            num_samples=(n_batches_cap * batch_size_eff) if n_batches_cap else None,
            shuffle=not smoke_test,
        )
        _use_cache = False

    log_file = LOGS_DIR / "validation_log.txt"
    global_step = 0

    # ══════════════════════════════════════════════════════════════════════
    # Epoch loop
    # ══════════════════════════════════════════════════════════════════════
    for epoch in range(start_epoch + 1, max_epochs_eff + 1):
        model.train()
        running_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch",
                    dynamic_ncols=True, disable=smoke_test)
        for batch_idx, raw_batch in enumerate(pbar):
            if n_batches_cap is not None and batch_idx >= n_batches_cap:
                break

            # Preprocess (skip if samples already come from cache)
            samples = raw_batch if _use_cache else preprocess_raw_batch(raw_batch)
            batch   = collate_samples(samples)

            # Forward + backward
            optimizer.zero_grad()
            loss, loss_parts, pred_norm_ = compute_batch_loss(model, batch, device, epoch=epoch)

            if not torch.isfinite(loss):
                lr_now = scheduler.get_last_lr()[0]
                p = pred_norm_.detach().float()
                print(f"  [warn] non-finite loss at epoch={epoch} step={batch_idx}")
                print(f"    wirelength = {loss_parts['wirelength']}")
                print(f"    area       = {loss_parts['area']}")
                print(f"    grouping   = {loss_parts['grouping']}")
                print(f"    mib        = {loss_parts['mib']}")
                print(f"    boundary   = {loss_parts['boundary']}")
                print(f"    overlap    = {loss_parts['overlap']}")
                print(f"    total      = {loss_parts['total']}")
                print(f"    pred min/max/mean = {p.min().item():.4e} / {p.max().item():.4e} / {p.mean().item():.4e}"
                      f"  nan={torch.isnan(p).any().item()} inf={torch.isinf(p).any().item()}")
                print(f"    lr         = {lr_now:.4e}")
                continue

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            running_loss += loss_parts["total"]
            n_steps += 1
            global_step += 1

            pbar.set_postfix(
                loss=f"{loss_parts['total']:.4f}",
                samples=f"{(batch_idx + 1) * batch_size_eff}",
            )

            if smoke_test:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step={global_step}  loss={loss_parts['total']:.4f}"
                      f"  wl={loss_parts['wirelength']:.4f}"
                      f"  area={loss_parts['area']:.4f}"
                      f"  grp={loss_parts['grouping']:.4f}"
                      f"  mib={loss_parts['mib']:.4f}"
                      f"  bnd={loss_parts['boundary']:.4f}"
                      f"  ovlp={loss_parts['overlap']:.4f}"
                      f"  ratio_reg={loss_parts['ratio_reg']:.4f}(×{loss_parts['lam_ratio']:.2f})"
                      f"  lr={lr_now:.2e}")

        # ── Epoch-end logging ─────────────────────────────────────────────
        avg_loss = running_loss / max(n_steps, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch:3d} | avg_loss={avg_loss:.4f} | steps={n_steps}")

        # Loss curve
        update_loss_plot(epoch_losses, LOGS_DIR / "loss_curve.png")

        # Checkpoint (latest)
        save_checkpoint(model, optimizer, epoch, epoch_losses,
                        best_score, CHECKPOINT_DIR / "latest.pt")
        print(f"  → saved checkpoint (latest)")

        # ── Validation every N epochs ─────────────────────────────────────
        if epoch % validate_every == 0:
            score = validate_with_solve(
                model, optimizer, epoch, epoch_losses,
                best_score, log_file, dry_run=smoke_test,
            )
            # validate_with_solve writes current model to best.pt before
            # evaluating, then restores the previous best.pt if no improvement.
            # We only need to update best_score and the early-stop counter here.
            if score < best_score:
                best_score = score
                no_improve_count = 0
                print(f"  → new best score: {best_score:.4f}, best.pt updated")
            else:
                no_improve_count += 1
                print(f"  → no improvement ({no_improve_count}/{patience_eff})")



            if no_improve_count >= patience_eff:
                print(f"Early stopping: no improvement for {patience_eff} validations.")
                break

    print("Training complete.")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"Logs:        {LOGS_DIR}")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Floorplan Transformer")
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run 2 batches × 2 epochs for quick validation"
    )
    parser.add_argument(
        "--num-shards", type=int, default=None,
        help="Use only the first N cache shards (default: all)"
    )
    args = parser.parse_args()
    train(smoke_test=args.smoke_test, num_shards=args.num_shards)
