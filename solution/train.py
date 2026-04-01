"""
Training script for the Floorplan Transformer.

Usage:
    python solution/train.py               # full training
    python solution/train.py --smoke-test  # 2 batches × 2 epochs (quick check)
"""

import argparse
import math
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    GRAD_CLIP_NORM, LAMBDA_WIRELENGTH, LAMBDA_AREA, LAMBDA_VIOLATION,
    VALIDATE_EVERY, VIZ_BLOCK_SIZES, RAW_FEATURE_DIM,
    LOGS_DIR, VIZ_DIR, CHECKPOINT_DIR, CONTEST_DIR,
    CACHE_DIR, CACHE_PRELOAD,
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
from loss.coord_loss     import coord_loss
from loss.wirelength_loss import wirelength_loss
from loss.area_loss      import area_loss
from loss.violation_loss import violation_loss


# =============================================================================
# Directory setup
# =============================================================================

def _make_dirs():
    for d in (LOGS_DIR, VIZ_DIR, CHECKPOINT_DIR):
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

def compute_batch_loss(model, batch: dict, device: torch.device):
    """
    Forward pass + compute all four loss terms.

    Returns:
        total_loss  (scalar, differentiable)
        dict of individual scalar .item() values for logging
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

    # Forward (teacher forcing)
    pred_norm = model(
        tf, wi,
        gt_positions=gtn,
        key_padding_mask=kpm,
        teacher_forcing=True,
    )                                          # [B, k, 4]

    # Zero out padded positions so they don't contribute to losses
    valid = (~kpm).float().unsqueeze(-1)       # [B, k, 1]
    pred_norm = pred_norm * valid

    # De-normalise for HPWL / area losses (raw pixel scale)
    pred_raw = pred_norm * crefs.view(-1, 1, 1)  # [B, k, 4]

    l_coord = coord_loss(pred_norm, gtn, cons)
    l_wl    = wirelength_loss(pred_raw, wiu, pins, p2b, hbase)
    l_area  = area_loss(pred_raw, abase)
    l_viol  = violation_loss(pred_norm, gtn, cons)

    total = (l_coord
             + LAMBDA_WIRELENGTH * l_wl
             + LAMBDA_AREA * l_area
             + LAMBDA_VIOLATION * l_viol)

    return total, {
        "total":      total.item(),
        "coord":      l_coord.item(),
        "wirelength": l_wl.item(),
        "area":       l_area.item(),
        "violation":  l_viol.item(),
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

def run_official_validation(epoch: int, log_file: Path, dry_run: bool = False) -> float:
    """
    Run the official evaluator as a subprocess.
    Prints output to terminal and appends to log_file.
    Returns the parsed Total Score (lower is better), or inf on failure.
    """
    header = f"========== Epoch {epoch} =========="
    print(header)

    if dry_run:
        msg = "[smoke-test] Skipping official evaluator — dry run.\n"
        print(msg)
        with open(log_file, "a") as f:
            f.write(header + "\n" + msg + "\n")
        return float("inf")

    cmd = [
        sys.executable,
        str(CONTEST_DIR / "iccad2026_evaluate.py"),
        "--evaluate",
        str(CONTEST_DIR / "my_optimizer.py")
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, cwd=str(_REPO_ROOT),
        )
        lines = []
        for line in proc.stdout:
            print(line, end="", flush=True)
            lines.append(line)
        proc.wait()
        output = "".join(lines)
    except Exception as e:
        output = f"[ERROR] Failed to run evaluator: {e}\n"
        print(output, end="")

    with open(log_file, "a") as f:
        f.write(header + "\n" + output + "\n")

    # Parse Total Score from evaluator output
    match = re.search(r"Total Score:\s*([\d.]+)", output)
    return float(match.group(1)) if match else float("inf")


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


def visualize_predictions(
    model:    nn.Module,
    epoch:    int,
    device:   torch.device,
    dry_run:  bool = False,
):
    """
    For each block size in VIZ_BLOCK_SIZES, run AR inference on the
    corresponding validation case and save a GT-vs-predicted figure.
    """
    if dry_run:
        for sz in VIZ_BLOCK_SIZES:
            print(f"  [smoke-test] viz epoch={epoch} case size={sz} — skipped (dry run)")
        return

    try:
        from lite_dataset_test import FloorplanDatasetLiteTest
    except ImportError:
        print("[viz] FloorplanDatasetLiteTest not importable — skipping.")
        return

    dataset = FloorplanDatasetLiteTest(str(_REPO_ROOT) + "/")

    model.eval()
    with torch.no_grad():
        for sz in VIZ_BLOCK_SIZES:
            test_id = sz - 21      # index 0 = 21 blocks, index 10 = 31 blocks, ...
            if test_id >= len(dataset):
                continue

            sample = dataset[test_id]
            inputs, labels = sample["input"], sample["label"]
            area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
            polygons, val_metrics = labels

            block_count = int((area_target != -1).sum().item())
            fp_sol = _polygons_to_fp_sol(polygons, block_count)
            # pad to full length for preprocess_sample
            fp_full = torch.zeros(area_target.shape[0], 4)
            fp_full[:block_count] = fp_sol

            s = preprocess_sample(
                area_target, b2b_conn, p2b_conn,
                pins_pos, constraints, fp_full, val_metrics,
            )

            tf_  = s["token_features"].unsqueeze(0).to(device)   # [1, k, 18]
            wi_  = s["w_int"].unsqueeze(0).to(device)             # [1, k, k]

            pred_norm = model(tf_, wi_, teacher_forcing=False)    # [1, k, 4]
            pred_raw  = (pred_norm[0] * s["canvas_ref"]).cpu()    # [k, 4]
            gt_raw    = s["gt_positions_raw"]                      # [k, 4] sorted

            _save_viz(gt_raw, pred_raw, sz, epoch, block_count)


def _save_viz(gt_raw, pred_raw, block_size: int, epoch: int, block_count: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab20(range(block_count))

    for ax, positions, title in [
        (axes[0], gt_raw,   f"GT  ({block_count} blocks)"),
        (axes[1], pred_raw, f"Pred ({block_count} blocks)"),
    ]:
        ax.set_title(title)
        for i in range(block_count):
            x, y, w, h = positions[i].tolist()
            rect = mpatches.Rectangle(
                (x, y), w, h,
                linewidth=0.8, edgecolor="black",
                facecolor=colors[i % len(colors)], alpha=0.7,
            )
            ax.add_patch(rect)
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    fig.suptitle(f"Epoch {epoch} — {block_size} blocks")
    fig.tight_layout()
    out = VIZ_DIR / f"epoch_{epoch:03d}_case_{block_size}.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  [viz] saved {out.name}")


# =============================================================================
# Main training loop
# =============================================================================

def train(smoke_test: bool = False):
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
        print(f"Cache detected at {CACHE_DIR} — using CachedShardIterableDataset"
              + (" (preload into RAM)" if CACHE_PRELOAD else " (streaming from disk)"))
        train_loader = get_cached_training_dataloader(
            cache_dir=CACHE_DIR,
            batch_size=batch_size_eff,
            shuffle=True,
            num_workers=2,
            preload=CACHE_PRELOAD,
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
            loss, loss_parts, pred_norm_ = compute_batch_loss(model, batch, device)

            if not torch.isfinite(loss):
                lr_now = scheduler.get_last_lr()[0]
                p = pred_norm_.detach().float()
                print(f"  [warn] non-finite loss at epoch={epoch} step={batch_idx}")
                print(f"    coord      = {loss_parts['coord']}")
                print(f"    wirelength = {loss_parts['wirelength']}")
                print(f"    area       = {loss_parts['area']}")
                print(f"    violation  = {loss_parts['violation']}")
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
                      f"  coord={loss_parts['coord']:.4f}"
                      f"  wl={loss_parts['wirelength']:.4f}"
                      f"  area={loss_parts['area']:.4f}"
                      f"  viol={loss_parts['violation']:.4f}"
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
            visualize_predictions(model, epoch, device, dry_run=smoke_test)
            score = run_official_validation(
                epoch, log_file, dry_run=smoke_test
            )

            if score < best_score:
                best_score = score
                no_improve_count = 0
                save_checkpoint(model, optimizer, epoch, epoch_losses,
                                best_score, CHECKPOINT_DIR / "best.pt")
                print(f"  → new best score: {best_score:.4f}, saved best.pt")
            else:
                no_improve_count += 1
                print(f"  → no improvement ({no_improve_count}/{patience_eff})")

            # visualize_predictions(model, epoch, device, dry_run=smoke_test)

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
    args = parser.parse_args()
    train(smoke_test=args.smoke_test)
