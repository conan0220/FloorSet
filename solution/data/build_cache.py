"""
Pre-compute the full preprocessing pipeline for all training samples and
save the results as shard files on disk.

Run ONCE before training:
    python solution/data/build_cache.py

Options:
    --num-samples N    Only cache N samples (default: all 1M)
    --shard-size  N    Samples per shard file (default: SHARD_SIZE from config)
    --cache-dir   DIR  Output directory (default: CACHE_DIR from config)
    --num-workers N    DataLoader workers for raw loading (default: 4)

Output layout:
    cache/
        shard_000000.pt   list of 2000 preprocessed sample dicts
        shard_000001.pt
        ...
        meta.pt           {"total_samples": int, "shard_size": int}
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

_SOLUTION_DIR = Path(__file__).resolve().parent.parent   # solution/
_REPO_ROOT    = _SOLUTION_DIR.parent                     # FloorSet/
sys.path.insert(0, str(_SOLUTION_DIR))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))

from config import CACHE_DIR, SHARD_SIZE
from data.floorset_loader import (
    get_training_dataloader,
    preprocess_sample,
    build_w_int_unnorm,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _preprocess_batch(raw_batch: tuple) -> list:
    """
    Convert one raw DataLoader batch into a list of preprocessed sample dicts
    (same as train.preprocess_raw_batch but imported-free).
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


def _flush_shard(buf: list, cache_dir: Path, shard_idx: int):
    path = cache_dir / f"shard_{shard_idx:06d}.pt"
    torch.save(buf, path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build preprocessing cache")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of samples to cache (default: all)")
    parser.add_argument("--shard-size",  type=int, default=SHARD_SIZE,
                        help=f"Samples per shard file (default: {SHARD_SIZE})")
    parser.add_argument("--cache-dir",   type=str, default=str(CACHE_DIR),
                        help=f"Output directory (default: {CACHE_DIR})")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers for raw loading (default: 4)")
    args = parser.parse_args()

    cache_dir  = Path(args.cache_dir)
    shard_size = args.shard_size

    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FloorSet preprocessing cache builder")
    print("=" * 60)
    print(f"  cache_dir   : {cache_dir}")
    print(f"  shard_size  : {shard_size}")
    print(f"  num_samples : {args.num_samples or 'all'}")
    print(f"  num_workers : {args.num_workers}")

    # Use batch_size=1 for maximum parallelism via num_workers
    loader = get_training_dataloader(
        batch_size=1,
        num_samples=args.num_samples,
        shuffle=False,
    )

    total_samples = 0
    shard_idx     = 0
    buf: list     = []
    t_start       = time.perf_counter()

    print()
    pbar = tqdm(
        loader,
        total=args.num_samples,
        unit="sample",
        dynamic_ncols=True,
        desc="Building cache",
    )

    for raw_batch in pbar:
        samples = _preprocess_batch(raw_batch)
        buf.extend(samples)
        total_samples += len(samples)
        pbar.update(len(samples) - 1)  # loader already counted +1 per iteration

        if len(buf) >= shard_size:
            path = _flush_shard(buf[:shard_size], cache_dir, shard_idx)
            buf = buf[shard_size:]
            shard_idx += 1
            pbar.set_postfix(shards=shard_idx, last=path.name)

    pbar.close()

    # Flush remaining samples
    if buf:
        path = _flush_shard(buf, cache_dir, shard_idx)
        shard_idx += 1

    # Write metadata
    meta = {"total_samples": total_samples, "shard_size": shard_size}
    torch.save(meta, cache_dir / "meta.pt")

    elapsed = time.perf_counter() - t_start
    print(f"\nDone.  {total_samples} samples → {shard_idx} shards  ({elapsed:.1f}s)")
    print(f"Cache: {cache_dir}")


if __name__ == "__main__":
    main()
