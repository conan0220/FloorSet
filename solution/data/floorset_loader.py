"""
FloorSet data preprocessing pipeline.

Converts raw batches from the official DataLoader into model-ready tensors.
All functions operate on single, unbatched samples (batch dimension squeezed).

Output layout of preprocess_sample():
    token_features      [k, 18]     float   - Encoder input (before Linear projection)
    w_int               [k, k]      float   - Normalised dense adjacency for Netlist Attention Bias
    b2b_conn            [e1, 3]     float   - Raw b2b edges, block indices in sorted order
    p2b_conn            [e2, 3]     float   - Raw p2b edges, block indices in sorted order
    pins_pos            [p, 2]      float   - Valid (unpadded) pin positions
    gt_positions_norm   [k, 4]      float   - GT (x,y,w,h) normalised by canvas_ref
    gt_positions_raw    [k, 4]      float   - GT (x,y,w,h) in original pixel scale
    sort_idx            [k]         long    - sort_idx[i] = original index of sorted block i
    sort_inv            [k]         long    - sort_inv[j] = sorted position of original block j
    block_count         int
    canvas_ref          float
    constraints_sorted  [k, 5]      float   - constraints[:, 0..4] in sorted block order
    hpwl_baseline       float               - metrics[6] + metrics[7]
    area_baseline       float               - bbox area computed from fp_sol GT positions
    metrics             [8]         float
"""

import random
import sys
from pathlib import Path

import torch
from torch.utils.data import IterableDataset

# Allow imports from repo root and iccad2026contest/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))

from iccad2026contest.iccad2026_evaluate import (
    get_training_dataloader as _official_train_loader,
    get_validation_dataloader as _official_val_loader,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    CANVAS_K,
    PAD_VALUE,
    BOUNDARY_CODE_TO_IDX,
    BOUNDARY_DIM,
    RAW_FEATURE_DIM,
    CACHE_DIR,
    SHARD_SIZE,
)

_DATA_PATH = str(_REPO_ROOT) + "/"


# =============================================================================
# Step 1 – Block count and canvas reference
# =============================================================================

def count_valid_blocks(area_target: torch.Tensor) -> int:
    """Number of non-padded blocks (area_target != PAD_VALUE)."""
    return int((area_target != PAD_VALUE).sum().item())


def count_valid_pins(pins_pos: torch.Tensor) -> int:
    """Number of non-padded pins (first coordinate != PAD_VALUE)."""
    return int((pins_pos[:, 0] != PAD_VALUE).sum().item())


def compute_canvas_ref(area_target: torch.Tensor, block_count: int) -> float:
    """
    canvas_ref = sqrt(sum of valid area_targets) * CANVAS_K

    Used to normalise all coordinates and areas to [0, ~1].
    """
    valid = area_target[:block_count]
    return float(valid.sum().sqrt().item()) * CANVAS_K


# =============================================================================
# Step 2 – Block sort order: preplaced → fixed-shape → soft
# =============================================================================

def sort_block_indices(constraints: torch.Tensor, block_count: int) -> torch.Tensor:
    """
    Return [k] tensor of original block indices sorted as:
        preplaced → fixed-shape (non-preplaced) → soft

    constraints[:, 0] = is_fixed_shape
    constraints[:, 1] = is_preplaced
    """
    cons = constraints[:block_count]          # [k, 5]
    is_pre   = cons[:, 1].bool()
    is_fixed = cons[:, 0].bool() & ~is_pre   # fixed-shape but not preplaced
    is_soft  = ~(cons[:, 0].bool() | is_pre)

    pre_idx   = is_pre.nonzero(as_tuple=True)[0]
    fixed_idx = is_fixed.nonzero(as_tuple=True)[0]
    soft_idx  = is_soft.nonzero(as_tuple=True)[0]

    return torch.cat([pre_idx, fixed_idx, soft_idx])   # [k]


def build_sort_inv(sort_idx: torch.Tensor) -> torch.Tensor:
    """
    Inverse permutation: sort_inv[original_idx] = sorted_position.
    sort_inv[sort_idx[i]] = i  for all i.
    """
    k = sort_idx.shape[0]
    inv = torch.zeros(k, dtype=torch.long)
    for i, orig in enumerate(sort_idx.tolist()):
        inv[orig] = i
    return inv


# =============================================================================
# Step 3 – Token features (18-dim)
# =============================================================================

def _boundary_onehot(boundary_codes: torch.Tensor) -> torch.Tensor:
    """
    Convert integer bitmask codes → [k, BOUNDARY_DIM] one-hot.
    Code 0 (no constraint) maps to all-zero vector.
    """
    k = boundary_codes.shape[0]
    out = torch.zeros(k, BOUNDARY_DIM)
    for i, code in enumerate(boundary_codes.tolist()):
        idx = BOUNDARY_CODE_TO_IDX.get(int(code))
        if idx is not None:
            out[i, idx] = 1.0
    return out


def build_token_features(
    area_target:  torch.Tensor,   # [n_padded]
    constraints:  torch.Tensor,   # [n_padded, 5]
    fp_sol:       torch.Tensor,   # [n_padded, 4]  raw [w, h, x, y]
    block_count:  int,
    canvas_ref:   float,
) -> torch.Tensor:
    """
    Build [k, RAW_FEATURE_DIM] token feature matrix in original block order.
    Caller should reorder with sort_idx afterwards.

    Feature layout (18 dims):
      [0]      area_target            normalised by canvas_ref²
      [1]      is_soft
      [2]      is_fixed_shape
      [3]      is_preplaced
      [4-5]    target_w, target_h     normalised; 0 if not fixed/preplaced
      [6-7]    target_x, target_y     normalised; 0 if not preplaced
      [8-15]   boundary_type          8-dim one-hot (all-zero = no constraint)
      [16]     is_mib
      [17]     is_cluster
    """
    cons = constraints[:block_count].float()   # [k, 5]
    fp   = fp_sol[:block_count].float()        # [k, 4]  [w, h, x, y]

    is_fixed  = cons[:, 0]
    is_pre    = cons[:, 1]
    is_mib    = cons[:, 2]
    is_clust  = cons[:, 3]
    bd_codes  = cons[:, 4].long()

    is_soft   = 1.0 - torch.clamp(is_fixed + is_pre, max=1.0)

    norm_area  = area_target[:block_count].float() / (canvas_ref ** 2)

    # target_w/h: valid for fixed-shape or preplaced blocks
    has_size  = torch.clamp(is_fixed + is_pre, max=1.0)
    target_w  = (fp[:, 0] / canvas_ref) * has_size
    target_h  = (fp[:, 1] / canvas_ref) * has_size

    # target_x/y: valid for preplaced blocks only
    target_x  = (fp[:, 2] / canvas_ref) * is_pre
    target_y  = (fp[:, 3] / canvas_ref) * is_pre

    boundary_oh = _boundary_onehot(bd_codes)   # [k, 8]

    scalar_feats = torch.stack([
        norm_area, is_soft, is_fixed, is_pre,
        target_w, target_h, target_x, target_y,
    ], dim=1)                                                    # [k, 8]

    token = torch.cat(
        [scalar_feats, boundary_oh, is_mib.unsqueeze(1), is_clust.unsqueeze(1)],
        dim=1,
    )                                                            # [k, 18]

    assert token.shape[1] == RAW_FEATURE_DIM, (
        f"Expected RAW_FEATURE_DIM={RAW_FEATURE_DIM}, got {token.shape[1]}"
    )
    return token


# =============================================================================
# Step 4 – Dense W_int adjacency matrix (for Netlist Attention Bias)
# =============================================================================

def build_w_int_dense(
    b2b_conn:    torch.Tensor,   # [n_edges, 3]  (block_i, block_j, weight)
    block_count: int,
) -> torch.Tensor:
    """
    Convert sparse b2b edge list to a dense [k, k] normalised adjacency matrix.
    Padding rows (block_i < 0) are discarded.
    The matrix is symmetrised and normalised by its maximum entry.
    """
    w_int = torch.zeros(block_count, block_count)

    valid = b2b_conn[:, 0] >= 0
    edges = b2b_conn[valid].float()
    if edges.numel() == 0:
        return w_int

    bi = edges[:, 0].long()
    bj = edges[:, 1].long()
    wt = edges[:, 2]

    in_range = (bi < block_count) & (bj < block_count)
    bi, bj, wt = bi[in_range], bj[in_range], wt[in_range]
    if bi.numel() == 0:
        return w_int

    # Accumulate weights (handles duplicate edges)
    flat_idx = bi * block_count + bj
    w_int.view(-1).scatter_add_(0, flat_idx, wt)

    # Symmetrise: take element-wise maximum of w_int and its transpose
    w_int = torch.maximum(w_int, w_int.t())

    # Normalise by maximum weight
    max_w = w_int.max()
    if max_w > 0:
        w_int = w_int / max_w

    return w_int


def reindex_w_int(w_int: torch.Tensor, sort_idx: torch.Tensor) -> torch.Tensor:
    """Permute rows and columns of w_int to match sorted block order."""
    return w_int[sort_idx][:, sort_idx]


# =============================================================================
# Step 5 – Re-index connectivity tensors to sorted block order
# =============================================================================

def _reindex_conn(
    conn:        torch.Tensor,   # [n_edges, 3]
    block_col:   int,            # which column holds block index (0 or 1)
    sort_inv:    torch.Tensor,   # [k]  original → sorted mapping
    block_count: int,
) -> torch.Tensor:
    """
    Generic helper: filter padding, filter out-of-range blocks, then
    replace block_col indices with sorted positions via sort_inv.
    Returns a float tensor of valid edges with updated block indices.
    """
    valid = conn[:, 0] >= 0
    edges = conn[valid].float().clone()

    blk = edges[:, block_col].long()
    in_range = blk < block_count
    edges    = edges[in_range]
    blk      = blk[in_range]

    edges[:, block_col] = sort_inv[blk].float()
    return edges                 # [valid_edges, 3]


def reindex_b2b_conn(
    b2b_conn:    torch.Tensor,
    sort_inv:    torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Re-index both block columns (0 and 1) of b2b_conn."""
    valid = b2b_conn[:, 0] >= 0
    edges = b2b_conn[valid].float().clone()

    bi = edges[:, 0].long()
    bj = edges[:, 1].long()
    in_range = (bi < block_count) & (bj < block_count)
    edges = edges[in_range]
    bi    = bi[in_range]
    bj    = bj[in_range]

    edges[:, 0] = sort_inv[bi].float()
    edges[:, 1] = sort_inv[bj].float()
    return edges                 # [valid_edges, 3]


def reindex_p2b_conn(
    p2b_conn:    torch.Tensor,
    sort_inv:    torch.Tensor,
    block_count: int,
) -> torch.Tensor:
    """Re-index block column (1) of p2b_conn; pin column (0) is left as-is."""
    return _reindex_conn(p2b_conn, block_col=1,
                         sort_inv=sort_inv, block_count=block_count)


# =============================================================================
# Step 6 – Ground truth positions and baselines
# =============================================================================

def build_gt_positions(
    fp_sol:    torch.Tensor,   # [n_padded, 4]  raw [w, h, x, y]
    sort_idx:  torch.Tensor,   # [k]
    canvas_ref: float,
) -> tuple:
    """
    Returns (gt_norm, gt_raw) both [k, 4] in (x, y, w, h) layout,
    reordered to match the sorted block sequence.
    """
    fp = fp_sol[sort_idx].float()                            # reorder, still [w,h,x,y]
    # Convert [w, h, x, y] → [x, y, w, h]
    gt_raw  = torch.stack([fp[:, 2], fp[:, 3], fp[:, 0], fp[:, 1]], dim=1)
    gt_norm = gt_raw / canvas_ref
    return gt_norm, gt_raw


def compute_hpwl_baseline(metrics: torch.Tensor) -> float:
    """metrics[6] (b2b_weighted_wl) + metrics[7] (p2b_weighted_wl)."""
    return float(metrics[6].item() + metrics[7].item())


def compute_area_baseline(fp_sol: torch.Tensor, block_count: int) -> float:
    """
    Bounding box area of the ground truth placement.
    fp_sol columns: [w, h, x, y] (original scale).
    """
    fp = fp_sol[:block_count].float()
    x_min = fp[:, 2].min().item()
    x_max = (fp[:, 2] + fp[:, 0]).max().item()   # x + w
    y_min = fp[:, 3].min().item()
    y_max = (fp[:, 3] + fp[:, 1]).max().item()   # y + h
    return (x_max - x_min) * (y_max - y_min)


# =============================================================================
# Main preprocessing entry point
# =============================================================================

def preprocess_sample(
    area_target:  torch.Tensor,   # [n_blocks_padded]
    b2b_conn:     torch.Tensor,   # [n_edges, 3]
    p2b_conn:     torch.Tensor,   # [n_edges, 3]
    pins_pos:     torch.Tensor,   # [n_pins, 2]
    constraints:  torch.Tensor,   # [n_blocks_padded, 5]
    fp_sol:       torch.Tensor,   # [n_blocks_padded, 4]  raw [w, h, x, y]
    metrics:      torch.Tensor,   # [8]
) -> dict:
    """
    Full preprocessing pipeline for one sample (no batch dimension).
    See module docstring for output shapes and descriptions.
    """
    # --- counts & canvas ---
    block_count = count_valid_blocks(area_target)
    canvas_ref  = compute_canvas_ref(area_target, block_count)

    # --- sort order ---
    sort_idx = sort_block_indices(constraints, block_count)    # [k]
    sort_inv = build_sort_inv(sort_idx)                        # [k]

    # --- token features (built in original order, then reordered) ---
    token_feats = build_token_features(
        area_target, constraints, fp_sol, block_count, canvas_ref
    )                                                          # [k, 18]
    token_feats = token_feats[sort_idx]

    # --- W_int dense matrix (normalised, sorted) ---
    w_int = build_w_int_dense(b2b_conn, block_count)           # [k, k]
    w_int = reindex_w_int(w_int, sort_idx)                     # [k, k]

    # --- connectivity with updated block indices ---
    b2b_reindexed = reindex_b2b_conn(b2b_conn, sort_inv, block_count)
    p2b_reindexed = reindex_p2b_conn(p2b_conn, sort_inv, block_count)

    # --- valid pins ---
    n_pins = count_valid_pins(pins_pos)
    pins   = pins_pos[:n_pins].float()                         # [p, 2]

    # --- GT positions ---
    gt_norm, gt_raw = build_gt_positions(fp_sol, sort_idx, canvas_ref)

    # --- baselines ---
    hpwl_baseline = compute_hpwl_baseline(metrics)
    area_baseline = compute_area_baseline(fp_sol, block_count)

    # --- sorted constraints (used by violation loss) ---
    constraints_sorted = constraints[:block_count][sort_idx].float()  # [k, 5]

    return {
        "token_features":     token_feats,          # [k, 18]
        "w_int":              w_int,                 # [k, k]
        "b2b_conn":           b2b_reindexed,         # [e1, 3]
        "p2b_conn":           p2b_reindexed,         # [e2, 3]
        "pins_pos":           pins,                  # [p, 2]
        "gt_positions_norm":  gt_norm,               # [k, 4]
        "gt_positions_raw":   gt_raw,                # [k, 4]
        "sort_idx":           sort_idx,              # [k]
        "sort_inv":           sort_inv,              # [k]
        "block_count":        block_count,
        "canvas_ref":         canvas_ref,
        "constraints_sorted": constraints_sorted,    # [k, 5]
        "hpwl_baseline":      hpwl_baseline,
        "area_baseline":      area_baseline,
        "metrics":            metrics.float(),       # [8]
    }


# =============================================================================
# W_int unnormalised (for wirelength loss — same as build_w_int_dense but
# without the /max_w step so weights are in original scale)
# =============================================================================

def build_w_int_unnorm(
    b2b_conn:    torch.Tensor,   # [n_edges, 3]
    block_count: int,
    sort_idx:    torch.Tensor,   # [k]
) -> torch.Tensor:
    """
    Dense unnormalised b2b weight matrix in sorted block order.
    Identical to build_w_int_dense but WITHOUT the normalisation step.
    """
    w = torch.zeros(block_count, block_count)
    valid = b2b_conn[:, 0] >= 0
    edges = b2b_conn[valid].float()
    if edges.numel() > 0:
        bi = edges[:, 0].long()
        bj = edges[:, 1].long()
        wt = edges[:, 2]
        in_range = (bi < block_count) & (bj < block_count)
        bi, bj, wt = bi[in_range], bj[in_range], wt[in_range]
        if bi.numel() > 0:
            w.view(-1).scatter_add_(0, bi * block_count + bj, wt)
            w = torch.maximum(w, w.t())
    return w[sort_idx][:, sort_idx]


# =============================================================================
# Shard-based cache: Dataset + DataLoader
# =============================================================================

class CachedShardIterableDataset(IterableDataset):
    """
    Streams preprocessed samples from shard files built by build_cache.py.

    preload=False (default):
        Loads one shard at a time from disk. Memory usage ≈ 1 shard (~100 MB).
        Supports num_workers > 0 (each worker owns a disjoint subset of shards).

    preload=True:
        Loads ALL shards into RAM at __init__ time. After that, training never
        touches the disk. Use this when RAM is large enough to hold the full
        dataset (~50 GB for 1 M samples). num_workers must be 0 to avoid
        duplicating the dataset across worker processes.
    """

    def __init__(
        self,
        cache_dir: Path,
        shuffle:   bool = True,
        seed:      int  = 42,
        preload:   bool = False,
    ):
        self.cache_dir   = Path(cache_dir)
        self.shuffle     = shuffle
        self.seed        = seed
        self.preload     = preload
        self.shard_files = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shard_files:
            raise FileNotFoundError(
                f"No shard files found in {cache_dir}. "
                "Run  python solution/data/build_cache.py  first."
            )

        if preload:
            print(f"[CachedShardIterableDataset] Preloading {len(self.shard_files)} shards into RAM...")
            self._data: list = []
            for i, path in enumerate(self.shard_files, 1):
                self._data.extend(torch.load(path, weights_only=False))
                print(f"\r  {i}/{len(self.shard_files)} shards  ({len(self._data)} samples)", end="", flush=True)
            print()
            self._total = len(self._data)
        else:
            meta = torch.load(self.cache_dir / "meta.pt", weights_only=True)
            self._total = meta["total_samples"]
            self._data  = None

    def __len__(self) -> int:
        return self._total

    def __iter__(self):
        if self.preload:
            # All data is in RAM — just shuffle and yield
            data = list(self._data)
            if self.shuffle:
                random.shuffle(data)
            yield from data
        else:
            worker_info  = torch.utils.data.get_worker_info()
            shard_files  = list(self.shard_files)

            # Split shards among DataLoader workers
            if worker_info is not None:
                shard_files = shard_files[worker_info.id :: worker_info.num_workers]

            rng = random.Random(self.seed)
            if self.shuffle:
                rng.shuffle(shard_files)

            for shard_path in shard_files:
                shard: list = torch.load(shard_path, weights_only=False)
                if self.shuffle:
                    random.shuffle(shard)
                yield from shard


def get_cached_training_dataloader(
    cache_dir:   Path = CACHE_DIR,
    batch_size:  int  = 1,
    shuffle:     bool = True,
    num_workers: int  = 2,
    seed:        int  = 42,
    preload:     bool = False,
):
    """
    DataLoader backed by the pre-computed shard cache.
    Each iteration yields a *list* of sample dicts (length == batch_size).
    Pass it directly to collate_samples() in train.py.

    preload=True: loads all shards into RAM at startup (needs large RAM,
                  forces num_workers=0 to avoid duplicating data).
    """
    from torch.utils.data import DataLoader

    if preload and num_workers > 0:
        print("[get_cached_training_dataloader] preload=True forces num_workers=0")
        num_workers = 0

    dataset = CachedShardIterableDataset(
        cache_dir=cache_dir, shuffle=shuffle, seed=seed, preload=preload,
    )

    def _identity_collate(batch):
        return batch   # list[dict], handed to collate_samples()

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_identity_collate,
        num_workers=num_workers,
        pin_memory=False,
    )


# =============================================================================
# DataLoader wrappers
# =============================================================================

def get_training_dataloader(
    batch_size:  int  = 1,
    num_samples: int  = None,
    shuffle:     bool = True,
):
    """
    Thin wrapper around the official training DataLoader.
    Returns raw batches; call preprocess_sample() on each item.
    """
    return _official_train_loader(
        data_path=_DATA_PATH,
        batch_size=batch_size,
        num_samples=num_samples,
        shuffle=shuffle,
    )


def get_validation_dataloader(batch_size: int = 1):
    """
    Thin wrapper around the official validation DataLoader.
    Returns raw batches; call preprocess_sample() on each item.
    """
    return _official_val_loader(
        data_path=_DATA_PATH,
        batch_size=batch_size,
    )


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("floorset_loader.py  —  smoke test")
    print("=" * 60)

    print("\n[1] Loading one training batch (batch_size=1) ...")
    loader = get_training_dataloader(batch_size=1, num_samples=3, shuffle=False)

    for batch_idx, batch in enumerate(loader):
        area_target, b2b_conn, p2b_conn, pins_pos, constraints, \
            tree_sol, fp_sol, metrics = batch

        # squeeze batch dimension
        area_target  = area_target.squeeze(0)
        b2b_conn     = b2b_conn.squeeze(0)
        p2b_conn     = p2b_conn.squeeze(0)
        pins_pos     = pins_pos.squeeze(0)
        constraints  = constraints.squeeze(0)
        fp_sol       = fp_sol.squeeze(0)
        metrics      = metrics.squeeze(0)

        sample = preprocess_sample(
            area_target, b2b_conn, p2b_conn, pins_pos,
            constraints, fp_sol, metrics,
        )

        print(f"\n--- Sample {batch_idx} ---")

        # (A) tensor shapes
        print("\n[A] Output tensor shapes:")
        shape_keys = [
            "token_features", "w_int", "b2b_conn", "p2b_conn",
            "pins_pos", "gt_positions_norm", "gt_positions_raw",
            "sort_idx", "sort_inv", "constraints_sorted", "metrics",
        ]
        for key in shape_keys:
            print(f"  {key:22s}: {tuple(sample[key].shape)}")
        print(f"  {'block_count':22s}: {sample['block_count']}")
        print(f"  {'canvas_ref':22s}: {sample['canvas_ref']:.4f}")
        print(f"  {'hpwl_baseline':22s}: {sample['hpwl_baseline']:.4f}")
        print(f"  {'area_baseline':22s}: {sample['area_baseline']:.4f}")

        # (B) canvas_ref sanity: compare to sqrt(sum(areas))
        k = sample["block_count"]
        raw_sum = area_target[:k].sum().sqrt().item()
        print(f"\n[B] canvas_ref check:")
        print(f"  sqrt(sum area_targets) = {raw_sum:.4f}")
        print(f"  canvas_ref / CANVAS_K  = {sample['canvas_ref'] / CANVAS_K:.4f}  (should match)")

        # (C) block sort order
        print(f"\n[C] Block sort order (first sample):")
        cons_s = sample["constraints_sorted"]      # [k, 5]
        n_pre   = int(cons_s[:, 1].sum().item())
        n_fixed = int((cons_s[:, 0] * (1 - cons_s[:, 1])).sum().item())
        n_soft  = k - n_pre - n_fixed
        print(f"  preplaced   = {n_pre}  (positions 0 .. {n_pre-1})")
        print(f"  fixed-shape = {n_fixed}  (positions {n_pre} .. {n_pre+n_fixed-1})")
        print(f"  soft        = {n_soft}  (positions {n_pre+n_fixed} .. {k-1})")
        # verify constraint flags agree with placement order
        if n_pre > 0:
            assert cons_s[:n_pre, 1].all(), "preplaced flag not set for preplaced blocks"
        if n_fixed > 0:
            assert cons_s[n_pre:n_pre+n_fixed, 0].all(), "fixed flag not set for fixed blocks"

        # (D) boundary one-hot example
        bd_col = constraints[:k][:, 4].long()
        has_bd = (bd_col != 0).nonzero(as_tuple=True)[0]
        print(f"\n[D] Boundary one-hot conversion example:")
        if has_bd.numel() == 0:
            print("  (no boundary-constrained blocks in this sample)")
        else:
            orig_idx = has_bd[0].item()
            code     = int(bd_col[orig_idx].item())
            sorted_pos = sample["sort_inv"][orig_idx].item()
            oh = sample["token_features"][sorted_pos, 8:16]
            print(f"  block (orig={orig_idx}, sorted={sorted_pos})")
            print(f"  boundary code = {code}")
            print(f"  one-hot [8:16] = {oh.tolist()}")
            expected_idx = BOUNDARY_CODE_TO_IDX.get(code, -1)
            if expected_idx >= 0:
                assert oh[expected_idx].item() == 1.0, "one-hot mismatch!"
                print(f"  ✓ index {expected_idx} is 1.0 as expected")

        if batch_idx == 0:
            break   # one sample is enough for smoke test

    print("\n" + "=" * 60)
    print("Smoke test passed — no errors.")
    print("=" * 60)
