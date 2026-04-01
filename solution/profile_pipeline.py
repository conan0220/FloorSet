"""
Pipeline timing profiler — standalone, does NOT modify train.py.

Measures wall-clock time for each stage across N batches and prints a summary.

Usage:
    python solution/profile_pipeline.py [--batches N] [--batch-size B] [--detail-preprocess] [--no-model]
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict

import torch

_SOLUTION_DIR = Path(__file__).resolve().parent
_REPO_ROOT    = _SOLUTION_DIR.parent
sys.path.insert(0, str(_SOLUTION_DIR))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))

from train import preprocess_raw_batch, collate_samples, compute_batch_loss
from data.floorset_loader import (
    get_training_dataloader,
    count_valid_blocks, count_valid_pins, compute_canvas_ref,
    sort_block_indices, build_sort_inv,
    build_token_features,
    build_w_int_dense, reindex_w_int,
    reindex_b2b_conn, reindex_p2b_conn,
    build_gt_positions,
    compute_hpwl_baseline, compute_area_baseline,
)
from model.transformer_floorplan import TransformerFloorplan
from config import RAW_FEATURE_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Timer:
    """Lightweight wall-clock timer with GPU sync support."""

    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda

    def now(self) -> float:
        if self.use_cuda:
            torch.cuda.synchronize()
        return time.perf_counter()


def preprocess_sample_timed(
    area_target, b2b_conn, p2b_conn, pins_pos, constraints, fp_sol, metrics,
    timer: _Timer,
    detail_timings: dict,
):
    """
    Drop-in replacement for preprocess_sample() that records per-step timings
    into detail_timings (same keys as preprocess_sample steps).
    Returns the same dict as preprocess_sample().
    """
    t = timer.now

    t0 = t()
    block_count = count_valid_blocks(area_target)
    canvas_ref  = compute_canvas_ref(area_target, block_count)
    detail_timings["1.counts+canvas"].append(t() - t0)

    t0 = t()
    sort_idx = sort_block_indices(constraints, block_count)
    sort_inv = build_sort_inv(sort_idx)
    detail_timings["2.sort"].append(t() - t0)

    t0 = t()
    token_feats = build_token_features(area_target, constraints, fp_sol, block_count, canvas_ref)
    token_feats = token_feats[sort_idx]
    detail_timings["3.token_features"].append(t() - t0)

    t0 = t()
    w_int = build_w_int_dense(b2b_conn, block_count)
    w_int = reindex_w_int(w_int, sort_idx)
    detail_timings["4.w_int"].append(t() - t0)

    t0 = t()
    b2b_reindexed = reindex_b2b_conn(b2b_conn, sort_inv, block_count)
    p2b_reindexed = reindex_p2b_conn(p2b_conn, sort_inv, block_count)
    detail_timings["5.reindex_conn"].append(t() - t0)

    t0 = t()
    n_pins = count_valid_pins(pins_pos)
    pins   = pins_pos[:n_pins].float()
    detail_timings["6.pins"].append(t() - t0)

    t0 = t()
    gt_norm, gt_raw = build_gt_positions(fp_sol, sort_idx, canvas_ref)
    detail_timings["7.gt_positions"].append(t() - t0)

    t0 = t()
    hpwl_baseline = compute_hpwl_baseline(metrics)
    area_baseline = compute_area_baseline(fp_sol, block_count)
    constraints_sorted = constraints[:block_count][sort_idx].float()
    detail_timings["8.baselines"].append(t() - t0)

    return {
        "token_features":     token_feats,
        "w_int":              w_int,
        "b2b_conn":           b2b_reindexed,
        "p2b_conn":           p2b_reindexed,
        "pins_pos":           pins,
        "gt_positions_norm":  gt_norm,
        "gt_positions_raw":   gt_raw,
        "sort_idx":           sort_idx,
        "sort_inv":           sort_inv,
        "block_count":        block_count,
        "canvas_ref":         canvas_ref,
        "constraints_sorted": constraints_sorted,
        "hpwl_baseline":      hpwl_baseline,
        "area_baseline":      area_baseline,
        "metrics":            metrics.float(),
    }


def _print_summary(timings: dict[str, list[float]], batch_size: int):
    total_per_batch = [
        sum(timings[k][i] for k in timings)
        for i in range(len(next(iter(timings.values()))))
    ]

    print("\n" + "=" * 60)
    print(f"{'Stage':<22} {'mean (ms)':>10} {'min (ms)':>10} {'max (ms)':>10} {'%':>7}")
    print("-" * 60)

    mean_total = sum(total_per_batch) / len(total_per_batch) * 1000

    for stage, values in timings.items():
        ms = [v * 1000 for v in values]
        mean = sum(ms) / len(ms)
        print(f"  {stage:<20} {mean:>10.1f} {min(ms):>10.1f} {max(ms):>10.1f} {mean/mean_total*100:>6.1f}%")

    print("-" * 60)
    total_ms = [v * 1000 for v in total_per_batch]
    mean_t = sum(total_ms) / len(total_ms)
    print(f"  {'TOTAL':<20} {mean_t:>10.1f} {min(total_ms):>10.1f} {max(total_ms):>10.1f} {'100.0':>6}%")
    print(f"\n  batch_size={batch_size}  →  {mean_t/batch_size:.1f} ms/sample")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches",           type=int, default=20,
                        help="Number of batches to profile (default: 20)")
    parser.add_argument("--detail-preprocess", action="store_true",
                        help="Break down preprocess_sample() into per-step timings")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--no-model",   action="store_true",
                        help="Skip forward+backward (DataLoader/preprocess only)")
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda  = device.type == "cuda"
    timer     = _Timer(use_cuda)

    print(f"Device            : {device}")
    print(f"Batches           : {args.batches}")
    print(f"Batch size        : {args.batch_size}")
    print(f"Model             : {'disabled' if args.no_model else 'enabled'}")
    print(f"Detail preprocess : {args.detail_preprocess}")

    # ── DataLoader ───────────────────────────────────────────────────────────
    loader = get_training_dataloader(
        batch_size  = args.batch_size,
        num_samples = args.batches * args.batch_size,
        shuffle     = False,
    )

    # ── Model (lightweight init) ─────────────────────────────────────────────
    model = None
    if not args.no_model:
        model = TransformerFloorplan().to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ── Warmup (1 batch, not recorded) ───────────────────────────────────────
    print("\nWarming up...")
    for raw_batch in loader:
        samples = preprocess_raw_batch(raw_batch)
        batch   = collate_samples(samples)
        if model is not None:
            compute_batch_loss(model, batch, device)
        break

    # ── Timed loop ───────────────────────────────────────────────────────────
    timings: dict[str, list[float]] = defaultdict(list)
    detail_timings: dict[str, list[float]] = defaultdict(list)

    print(f"Profiling {args.batches} batches...\n")
    loader = get_training_dataloader(
        batch_size  = args.batch_size,
        num_samples = args.batches * args.batch_size,
        shuffle     = False,
    )

    for batch_idx, raw_batch in enumerate(loader):
        if batch_idx >= args.batches:
            break

        # Stage 1: preprocess_raw_batch
        t0 = timer.now()
        if args.detail_preprocess:
            area_target, b2b_conn, p2b_conn, pins_pos, constraints, _, fp_sol, metrics = raw_batch
            B = area_target.shape[0]
            samples = []
            for i in range(B):
                s = preprocess_sample_timed(
                    area_target[i], b2b_conn[i], p2b_conn[i],
                    pins_pos[i], constraints[i], fp_sol[i], metrics[i],
                    timer, detail_timings,
                )
                from train import _build_w_int_unnorm
                s["w_int_unnorm"] = _build_w_int_unnorm(
                    b2b_conn[i], s["block_count"], s["sort_idx"]
                )
                samples.append(s)
        else:
            samples = preprocess_raw_batch(raw_batch)
        t1 = timer.now()
        timings["preprocess"].append(t1 - t0)

        avg_blocks = sum(s["block_count"] for s in samples) / len(samples)

        # Stage 2: collate_samples
        batch = collate_samples(samples)
        t2 = timer.now()
        timings["collate"].append(t2 - t1)

        if model is None:
            print(f"  batch {batch_idx+1:>3}/{args.batches}"
                  f"  blocks={avg_blocks:5.1f}"
                  f"  preprocess={timings['preprocess'][-1]*1000:6.1f}ms"
                  f"  collate={timings['collate'][-1]*1000:5.1f}ms")
            continue

        # Stage 3: CPU→GPU transfer (inside compute_batch_loss)
        # Isolate by timing .to(device) separately
        t3 = timer.now()
        _ = {k: v.to(device) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
        t4 = timer.now()
        timings["cpu→gpu"].append(t4 - t3)

        # Stage 4: forward + backward
        optimizer.zero_grad()
        t5 = timer.now()
        loss, _, _ = compute_batch_loss(model, batch, device)
        loss.backward()
        t6 = timer.now()
        timings["forward+backward"].append(t6 - t5)

        optimizer.step()

        print(f"  batch {batch_idx+1:>3}/{args.batches}"
              f"  blocks={avg_blocks:5.1f}"
              f"  preprocess={timings['preprocess'][-1]*1000:6.1f}ms"
              f"  collate={timings['collate'][-1]*1000:5.1f}ms"
              f"  cpu→gpu={timings['cpu→gpu'][-1]*1000:5.1f}ms"
              f"  fwd+bwd={timings['forward+backward'][-1]*1000:6.1f}ms")

    _print_summary(timings, args.batch_size)

    if args.detail_preprocess and detail_timings:
        # Scale detail timings: they are per-sample; show per-sample averages
        n_samples = args.batches * args.batch_size
        print(f"\n--- preprocess_sample() breakdown (per sample, n={n_samples}) ---")
        _print_summary(detail_timings, batch_size=1)


if __name__ == "__main__":
    main()
