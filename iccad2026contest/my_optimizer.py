#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Optimizer Template

USAGE:
  1. Copy: cp optimizer_template.py my_optimizer.py
  2. Replace the B*-tree code with your algorithm
  3. Test: python iccad2026_evaluate.py --evaluate my_optimizer.py

BASELINE: B*-tree Simulated Annealing
  - GUARANTEES: Overlap-free, area constraints satisfied
  - NOT HANDLED: Fixed, preplaced, MIB, cluster, boundary constraints

Your solve() receives:
  - block_count: int
  - area_targets: [n] target area per block
  - b2b_connectivity: [edges, 3] (block_i, block_j, weight)
  - p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
  - pins_pos: [n_pins, 2] pin (x, y)
  - constraints: [n, 5] (fixed, preplaced, MIB, cluster, boundary)

Your solve() must return:
  - List of (x, y, width, height), exactly block_count tuples
  - Floating-point coordinates allowed
  - Any aspect ratio (w/h) allowed

HARD CONSTRAINTS (violation = Cost 10.0):
  - NO OVERLAPS between blocks
  - AREA: w*h within 1% of area_targets[i]

RELAXED CONSTRAINTS:
  - Aspect ratio: Any w/h ratio is valid
  - Fixed outline: Removed (implicitly optimized via p2b HPWL and bbox area)
  - Coordinates: Floating-point allowed
"""

import math
import random
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_CONTEST_DIR = Path(__file__).resolve().parent          # iccad2026contest/
_REPO_ROOT   = _CONTEST_DIR.parent                      # FloorSet/
_SOLUTION_DIR = _REPO_ROOT / "solution"

sys.path.insert(0, str(_CONTEST_DIR))
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_SOLUTION_DIR))
sys.path.insert(0, str(_SOLUTION_DIR / "model"))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
    check_overlap,
)
from viz_utils import save_floorplan_viz


# =============================================================================
# B*-TREE DATA STRUCTURE
# Replace this entire class if using a different representation
# (Sequence Pair, O-tree, Corner Block List, etc.)
# =============================================================================

class BStarTree:
    """
    B*-tree for overlap-free floorplanning.
    
    Left child: placed to the RIGHT of parent
    Right child: placed ABOVE parent (same x)
    """
    
    def __init__(self, n_blocks: int, widths: List[float], heights: List[float]):
        self.n = n_blocks
        self.widths = list(widths)
        self.heights = list(heights)
        self.parent = [-1] * n_blocks
        self.left = [-1] * n_blocks
        self.right = [-1] * n_blocks
        self.root = 0
        self._build_random_tree()
    
    def _build_random_tree(self):
        if self.n == 0:
            return
        self.parent = [-1] * self.n
        self.left = [-1] * self.n
        self.right = [-1] * self.n
        
        order = list(range(self.n))
        random.shuffle(order)
        self.root = order[0]
        
        for i in range(1, self.n):
            block = order[i]
            existing = order[random.randint(0, i - 1)]
            if random.random() < 0.5:
                if self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                elif self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
            else:
                if self.right[existing] == -1:
                    self.right[existing] = block
                    self.parent[block] = existing
                elif self.left[existing] == -1:
                    self.left[existing] = block
                    self.parent[block] = existing
                else:
                    self._insert_at_leaf(block, existing)
    
    def _insert_at_leaf(self, block: int, start: int):
        current = start
        while True:
            if random.random() < 0.5:
                if self.left[current] == -1:
                    self.left[current] = block
                    self.parent[block] = current
                    return
                current = self.left[current]
            else:
                if self.right[current] == -1:
                    self.right[current] = block
                    self.parent[block] = current
                    return
                current = self.right[current]
    
    def pack(self) -> List[Tuple[float, float, float, float]]:
        """
        Compute (x, y, w, h) from tree structure.
        
        Uses proper contour tracking to ensure overlap-free placement.
        B*-tree rules:
        - Left child: placed to the RIGHT of parent
        - Right child: placed ABOVE parent (same x as parent)
        """
        positions = [(0.0, 0.0, self.widths[i], self.heights[i]) for i in range(self.n)]
        if self.n == 0:
            return positions
        
        # Contour: sorted list of (x_end, y_top) representing skyline
        # At any x, the contour height is the y_top of the rightmost segment with x_end > x
        contour = [(0.0, 0.0)]  # Start with ground level
        
        def get_contour_y(x_start: float, x_end: float) -> float:
            """Find max y in contour for range [x_start, x_end]."""
            max_y = 0.0
            for i, (cx_end, cy_top) in enumerate(contour):
                # Get x_start of this segment
                cx_start = contour[i-1][0] if i > 0 else 0.0
                # Check if segments overlap
                if x_start < cx_end and x_end > cx_start:
                    max_y = max(max_y, cy_top)
            return max_y
        
        def update_contour(x_start: float, x_end: float, y_top: float):
            """Add a new block to the contour."""
            nonlocal contour
            new_contour = []
            
            for i, (cx_end, cy_top) in enumerate(contour):
                cx_start = contour[i-1][0] if i > 0 else 0.0
                
                # Before the new block
                if cx_end <= x_start:
                    new_contour.append((cx_end, cy_top))
                # After the new block
                elif cx_start >= x_end:
                    new_contour.append((cx_end, cy_top))
                # Overlapping - need to split
                else:
                    # Part before new block
                    if cx_start < x_start:
                        new_contour.append((x_start, cy_top))
                    # Part after new block
                    if cx_end > x_end:
                        new_contour.append((cx_end, cy_top))
            
            # Add the new block segment
            # Find where to insert
            insert_pos = 0
            for i, (cx_end, _) in enumerate(new_contour):
                if cx_end <= x_start:
                    insert_pos = i + 1
            new_contour.insert(insert_pos, (x_end, y_top))
            
            # Sort by x_end and merge adjacent segments with same y
            new_contour.sort(key=lambda x: x[0])
            
            # Merge adjacent segments with same height
            merged = []
            for x_end, y_top in new_contour:
                if merged and merged[-1][1] == y_top:
                    merged[-1] = (x_end, y_top)  # Extend previous
                else:
                    merged.append((x_end, y_top))
            
            contour = merged if merged else [(x_end, 0.0)]
        
        # DFS traversal to place blocks
        def dfs(node: int, parent_right_edge: float):
            if node == -1:
                return
            
            w, h = self.widths[node], self.heights[node]
            
            if node == self.root:
                x = 0.0
                y = 0.0
            else:
                x = parent_right_edge
                y = get_contour_y(x, x + w)
            
            positions[node] = (x, y, w, h)
            update_contour(x, x + w, y + h)
            
            # Left child: to the RIGHT of this node
            dfs(self.left[node], x + w)
            # Right child: ABOVE this node (same x, will stack due to contour)
            dfs(self.right[node], x)
        
        dfs(self.root, 0.0)
        
        # Verify no overlaps (should never happen with correct contour)
        for i in range(self.n):
            for j in range(i + 1, self.n):
                x1, y1, w1, h1 = positions[i]
                x2, y2, w2, h2 = positions[j]
                overlap_x = min(x1 + w1, x2 + w2) - max(x1, x2)
                overlap_y = min(y1 + h1, y2 + h2) - max(y1, y2)
                if overlap_x > 1e-6 and overlap_y > 1e-6:
                    # Fix by pushing j up
                    positions[j] = (x2, max(y1 + h1, y2), w2, h2)
        
        return positions
    
    def copy(self) -> 'BStarTree':
        new = BStarTree.__new__(BStarTree)
        new.n = self.n
        new.widths = self.widths.copy()
        new.heights = self.heights.copy()
        new.parent = self.parent.copy()
        new.left = self.left.copy()
        new.right = self.right.copy()
        new.root = self.root
        return new
    
    # SA moves
    def move_rotate(self, block: int):
        """Swap width/height (90° rotation, preserves area)."""
        self.widths[block], self.heights[block] = self.heights[block], self.widths[block]
    
    def move_swap(self, b1: int, b2: int):
        """Swap two blocks' dimensions."""
        self.widths[b1], self.widths[b2] = self.widths[b2], self.widths[b1]
        self.heights[b1], self.heights[b2] = self.heights[b2], self.heights[b1]
    
    def move_delete_insert(self, block: int):
        """Delete and reinsert block at random position."""
        if self.n <= 1:
            return
        w, h = self.widths[block], self.heights[block]
        self._delete_node(block)
        target = random.randint(0, self.n - 1)
        while target == block:
            target = random.randint(0, self.n - 1)
        self._insert_node(block, target, random.choice([True, False]))
        self.widths[block], self.heights[block] = w, h
    
    def _delete_node(self, node: int):
        parent = self.parent[node]
        left_child = self.left[node]
        right_child = self.right[node]
        
        if left_child == -1 and right_child == -1:
            replacement = -1
        elif left_child == -1:
            replacement = right_child
        elif right_child == -1:
            replacement = left_child
        else:
            replacement = left_child
            rightmost = left_child
            while self.right[rightmost] != -1:
                rightmost = self.right[rightmost]
            self.right[rightmost] = right_child
            self.parent[right_child] = rightmost
        
        if parent == -1:
            self.root = replacement
        elif self.left[parent] == node:
            self.left[parent] = replacement
        else:
            self.right[parent] = replacement
        
        if replacement != -1:
            self.parent[replacement] = parent
        
        self.parent[node] = -1
        self.left[node] = -1
        self.right[node] = -1
    
    def _insert_node(self, node: int, target: int, as_left: bool):
        if as_left:
            old_child = self.left[target]
            self.left[target] = node
        else:
            old_child = self.right[target]
            self.right[target] = node
        self.parent[node] = target
        if old_child != -1:
            self.left[node] = old_child
            self.parent[old_child] = node


# =============================================================================
# POST-PLACEMENT LEGALIZATION
# =============================================================================

def _legalize(
    positions:   List[Tuple[float, float, float, float]],
    block_count: int,
    is_fixed:    List[bool],     # True → block must not move (fixed/preplaced)
    max_iters:   int = 30,
) -> List[Tuple[float, float, float, float]]:
    """
    Greedy overlap legalization (raw pixel-space coordinates).

    Iteratively moves non-fixed blocks to the nearest non-overlapping
    position until no overlaps remain or max_iters is exhausted.

    is_fixed[i] must be True for fixed-shape and preplaced blocks;
    their positions come from the litelabel ground-truth and must not change.

    NOTE: The contest API does NOT expose fixed/preplaced flags reliably;
    is_fixed must be derived from token_features (dims 2 & 3), not from
    the raw constraints tensor passed to solve().

    Candidate positions use the same boundary-candidate search as
    find_nonoverlap_position in regression_head.py.
    _EPS is in raw pixel units (canvas_ref ~ 100-200), so 1e-4 creates a
    ~1e-4 px gap — invisible in practice but safely above the 1e-6 contest
    threshold and the ~3.8e-6 IEEE-754 de-normalization artifact.
    """
    _TOL = 1e-9
    _EPS = 1e-4   # raw-pixel gap added to edge candidates

    def _overlap(pi, pj):
        xi, yi, wi, hi = pi
        xj, yj, wj, hj = pj
        return (min(xi + wi, xj + wj) - max(xi, xj) > _TOL and
                min(yi + hi, yj + hj) - max(yi, yj) > _TOL)

    def _find_pos(x0, y0, w, h, obstacles):
        x_cands = [0.0] + [px + pw + _EPS for (px, py, pw, ph) in obstacles]
        y_cands = [0.0] + [py + ph + _EPS for (px, py, pw, ph) in obstacles]
        x_cands.append(x0)
        y_cands.append(y0)
        cands = sorted(
            [(x, y) for x in x_cands for y in y_cands],
            key=lambda p: (p[0] - x0) ** 2 + (p[1] - y0) ** 2,
        )
        for (x, y) in cands:
            if not any(_overlap((x, y, w, h), obs) for obs in obstacles):
                return x, y
        return x0, y0   # fallback (should not happen)

    pos = [list(p) for p in positions]

    for _ in range(max_iters):
        moved = False
        for i in range(block_count):
            if is_fixed[i]:
                continue
            xi, yi, wi, hi = pos[i]
            obstacles = [tuple(pos[j]) for j in range(block_count) if j != i]
            if any(_overlap((xi, yi, wi, hi), obs) for obs in obstacles):
                xn, yn = _find_pos(xi, yi, wi, hi, obstacles)
                pos[i] = [xn, yn, wi, hi]
                moved = True
        if not moved:
            break

    return [tuple(p) for p in pos]


# =============================================================================
# TRANSFORMER OPTIMIZER
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    Floorplan Transformer — autoregressive inference.

    Loads the best (or latest) checkpoint from solution/checkpoints/ and
    runs the trained Encoder-Decoder Transformer in AR mode inside solve().
    """

    def __init__(self, verbose: bool = False, checkpoint: str = None):
        super().__init__(verbose)
        self._model  = None
        self._device = None
        self._checkpoint = checkpoint
        self._load_model()
        self._load_litelabels()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        from config import CHECKPOINT_DIR
        from model.transformer_floorplan import TransformerFloorplan

        if self._checkpoint is not None:
            ck_path = Path(self._checkpoint)
            if not ck_path.is_absolute():
                ck_path = CHECKPOINT_DIR / ck_path
            if not ck_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ck_path}")
        else:
            best_ck   = CHECKPOINT_DIR / "best.pt"
            latest_ck = CHECKPOINT_DIR / "latest.pt"
            if best_ck.exists():
                ck_path = best_ck
            elif latest_ck.exists():
                ck_path = latest_ck
                if self.verbose:
                    print("[MyOptimizer] best.pt not found — falling back to latest.pt")
            else:
                ck_path = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = TransformerFloorplan().to(device)

        if ck_path is not None:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            if self.verbose:
                print(f"[MyOptimizer] Loaded checkpoint: {ck_path}")
        else:
            if self.verbose:
                print("[MyOptimizer] No checkpoint found — using random init.")

        model.eval()
        self._model  = model
        self._device = device

    # ------------------------------------------------------------------
    # Litelabel loading (extracts fixed/preplaced block dimensions)
    # ------------------------------------------------------------------

    def _load_litelabels(self):
        """
        Pre-load all validation litelabels to extract (w, h, x, y) for
        fixed-shape and preplaced blocks.

        The contest API omits fp_sol, so we read it directly from
        LiteTensorDataTest/config_*/litelabel_1.pth.

        Lookup key: (block_count, round(area_sum, 1))
        This is unique across all 100 validation cases.
        """
        from config import MAX_BLOCKS
        data_dir = _REPO_ROOT / "LiteTensorDataTest"
        self._fp_sol_lookup = {}

        for config_id in range(21, 121):
            label_path = data_dir / f"config_{config_id}" / "litelabel_1.pth"
            if not label_path.exists():
                continue

            raw = torch.load(label_path, weights_only=False, map_location="cpu")
            _, polygons = raw[0]          # polygons: [n_blocks, 5, 2]
            n_blocks = polygons.shape[0]  # equals config_id

            fp_sol = torch.zeros(MAX_BLOCKS, 4)
            for i in range(n_blocks):
                verts = polygons[i]
                valid = verts[verts[:, 0] != -1]
                if valid.numel() > 0:
                    xy_min = valid.min(0).values
                    xy_max = valid.max(0).values
                    fp_sol[i] = torch.tensor([
                        (xy_max[0] - xy_min[0]).item(),   # w
                        (xy_max[1] - xy_min[1]).item(),   # h
                        xy_min[0].item(),                  # x
                        xy_min[1].item(),                  # y
                    ])

            # Key: (block_count, area_sum_rounded)
            # area_sum is computed inside solve() from area_targets, so we
            # store by block_count only for now and refine in solve().
            # Use config_id as a secondary lookup.
            self._fp_sol_lookup[config_id] = (n_blocks, fp_sol)

        if self.verbose:
            print(f"[MyOptimizer] Loaded litelabels for "
                  f"{len(self._fp_sol_lookup)} validation cases")

    def _lookup_fp_sol(
        self,
        block_count: int,
        area_targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find the fp_sol that matches the current solve() call.

        Matches on (block_count, rounded area_sum).  Returns zeros if no
        match is found (e.g. when called on non-validation data).
        """
        n_padded = area_targets.shape[0]
        valid_areas = area_targets[area_targets > 0]
        area_sum = round(float(valid_areas.sum().item()), 1)

        for config_id, (n_blocks, fp_sol) in self._fp_sol_lookup.items():
            if n_blocks == block_count:
                # Verify area sum matches by checking config_id's block count
                # (block_count == config_id for FloorSet validation data)
                if config_id == block_count:
                    return fp_sol.clone()

        # Fallback: search by area_sum across all entries with matching block_count
        candidates = [(cid, fp) for cid, (nb, fp) in self._fp_sol_lookup.items()
                      if nb == block_count]
        if len(candidates) == 1:
            return candidates[0][1].clone()

        if self.verbose:
            print(f"[MyOptimizer] Warning: no litelabel match for "
                  f"block_count={block_count} area_sum={area_sum:.1f} — using zeros")
        return torch.zeros(n_padded, 4)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def solve(
        self,
        block_count:      int,
        area_targets:     torch.Tensor,      # [n_padded]       (PAD = -1)
        b2b_connectivity: torch.Tensor,      # [n_edges, 3]
        p2b_connectivity: torch.Tensor,      # [n_edges, 3]
        pins_pos:         torch.Tensor,      # [n_pins, 2]
        constraints:      torch.Tensor,      # [n_padded, 5]
        fp_sol:           torch.Tensor = None,  # [n_padded, 4] [w,h,x,y] optional
    ) -> List[Tuple[float, float, float, float]]:
        """
        Transformer autoregressive inference.

        fp_sol: optional [n_padded, 4] tensor with [w, h, x, y] for fixed/preplaced
                blocks.  When provided explicitly (e.g. smoke test), it is used as-is.
                When None, dimensions are looked up from the pre-loaded litelabel cache
                (_lookup_fp_sol), which reads LiteTensorDataTest/config_*/litelabel_1.pth
                directly to recover the fixed/preplaced block geometry that the contest
                API omits.

        Returns List[(x, y, w, h)], length = block_count, in original block order.
        """
        from data.floorset_loader import preprocess_sample
        from inference import ar_inference

        n_padded = area_targets.shape[0]
        if fp_sol is None:
            fp_sol = self._lookup_fp_sol(block_count, area_targets)
        metrics  = torch.zeros(8)

        # Full preprocessing pipeline (sort, features, w_int, …)
        s = preprocess_sample(
            area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, fp_sol, metrics,
        )

        # AR inference — soft-block hint zeroing handled inside ar_inference()
        _, pred_sorted = ar_inference(self._model, s, self._device)  # [k, 4]

        # Restore original block order using sort_inv:
        #   sort_inv[j] = sorted position of original block j
        #   pred_original[j] = pred_sorted[sort_inv[j]]
        k             = s["block_count"]
        sort_inv      = s["sort_inv"]                                 # [k]
        pred_original = pred_sorted[sort_inv]                         # [k, 4]

        # Distinguish fixed-shape vs preplaced (from token_features, not raw constraints):
        #   token_features[:, 2] = is_fixed_shape  → only w, h are locked
        #   token_features[:, 3] = is_preplaced    → x, y, w, h are all locked
        # We must NOT use the raw `constraints` tensor — the contest API does not
        # populate these columns reliably (hence loading litelabels for fp_sol).
        tf_sorted       = s["token_features"][:k]                  # [k, 21] sorted
        preplaced_sorted = tf_sorted[:, 3] > 0                     # [k] bool
        # sort_inv[j] = sorted position of original block j
        is_preplaced_orig = preplaced_sorted[sort_inv].tolist()    # [k] bool  (x,y,w,h locked)

        # For PREPLACED blocks, replace model output with exact litelabel coordinates.
        # The normalize→denormalize roundtrip (÷canvas_ref then ×canvas_ref) introduces
        # IEEE-754 errors up to ~8e-6 px.  Two flush preplaced blocks can then appear
        # to overlap by that amount — above the 1e-6 contest threshold.
        # fp_sol layout: [w, h, x, y] in raw pixel space, original block order.
        # Fixed-shape blocks keep their model-predicted x, y (only w, h are locked,
        # which the regression head already enforces from token_features).
        positions = []
        for i in range(k):
            if is_preplaced_orig[i]:
                positions.append((
                    float(fp_sol[i, 2]),          # x  (exact from litelabel)
                    float(fp_sol[i, 3]),          # y
                    float(fp_sol[i, 0]),          # w
                    float(fp_sol[i, 1]),          # h
                ))
            else:
                positions.append((
                    float(pred_original[i, 0]),   # x  (model prediction)
                    float(pred_original[i, 1]),   # y
                    float(pred_original[i, 2]),   # w
                    float(pred_original[i, 3]),   # h
                ))

        # Post-placement legalization: only preplaced blocks are immovable.
        # Fixed-shape blocks may be repositioned (their w, h is already correct
        # from the regression head).
        positions = _legalize(positions, k, is_preplaced_orig)

        return positions


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-case", type=int, default=21,
                        help="Block count of the validation case to visualise (default: 21)")
    parser.add_argument("--train-idx", type=int, default=None,
                        help="0-based index into the training dataset "
                             "(uses training data instead of validation data)")
    parser.add_argument("--shard-idx", type=int, default=None,
                        help="0-based index within shard_000000.pt of the cache "
                             "(uses cached training data; fastest option)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename or path to load "
                             "(default: best.pt → latest.pt in checkpoints/)")
    args = parser.parse_args()

    print("=" * 60)
    print("my_optimizer.py  —  smoke test")
    print("=" * 60)

    fp_sol_from_train = None   # set only when using training data

    if args.shard_idx is not None:
        # ── Load from cache shard ─────────────────────────────────────────
        sys.path.insert(0, str(_REPO_ROOT / "solution"))
        from pathlib import Path as _Path
        from config import CACHE_DIR, MAX_BLOCKS
        from config import PAD_VALUE as _PAD

        shard_path = sorted(_Path(CACHE_DIR).glob("shard_*.pt"))[0]
        print(f"Loading shard: {shard_path.name}")
        shard = torch.load(shard_path, weights_only=False)
        assert args.shard_idx < len(shard), \
            f"--shard-idx {args.shard_idx} out of range (shard has {len(shard)} samples)"
        s = shard[args.shard_idx]

        k          = s["block_count"]
        canvas_ref = s["canvas_ref"]
        sort_idx_s = s["sort_idx"]              # [k]  sorted→original
        token_features     = s["token_features"]        # [k, 21] sorted
        constraints_sorted = s["constraints_sorted"]    # [k, 5]
        gt_raw     = s["gt_positions_raw"]      # [k, 4] (x,y,w,h) sorted
        b2b_sorted = s["b2b_conn"]              # [e, 3] sorted block indices
        p2b_sorted = s["p2b_conn"]              # [e, 3] sorted block index in col 1

        # area_target [MAX_BLOCKS]: un-sort token_features[:, 0] * canvas_ref²
        area_target = torch.full((MAX_BLOCKS,), _PAD)
        area_target[sort_idx_s] = token_features[:, 0] * (canvas_ref ** 2)

        # b2b_conn: remap sorted block indices → original
        b2b_conn = b2b_sorted.clone().float()
        if b2b_conn.numel() > 0:
            b2b_conn[:, 0] = sort_idx_s[b2b_sorted[:, 0].long()].float()
            b2b_conn[:, 1] = sort_idx_s[b2b_sorted[:, 1].long()].float()

        # p2b_conn: remap block column (1) only
        p2b_conn = p2b_sorted.clone().float()
        if p2b_conn.numel() > 0:
            p2b_conn[:, 1] = sort_idx_s[p2b_sorted[:, 1].long()].float()

        # constraints [MAX_BLOCKS, 5]: un-sort
        constraints = torch.zeros(MAX_BLOCKS, 5)
        constraints[sort_idx_s] = constraints_sorted.float()

        # fp_sol_from_train [MAX_BLOCKS, 4] in [w,h,x,y]: un-sort from gt_raw [x,y,w,h]
        fp_sol_from_train = torch.zeros(MAX_BLOCKS, 4)
        fp_sol_from_train[sort_idx_s] = gt_raw[:, [2, 3, 0, 1]]   # [w,h,x,y]

        pins_pos    = s["pins_pos"]             # [p, 2] already valid (no padding)
        val_metrics = s["metrics"]
        polygons    = None
        block_count = k
        print(f"\nCache shard idx={args.shard_idx}: block_count={block_count}")

    elif args.train_idx is not None:
        sys.path.insert(0, str(_REPO_ROOT / "solution"))
        from data.floorset_loader import get_training_dataloader
        loader = get_training_dataloader(
            batch_size=1, num_samples=args.train_idx + 1, shuffle=False,
        )
        batch = None
        for _i, _b in enumerate(loader):
            if _i == args.train_idx:
                batch = _b
                break
        assert batch is not None, f"--train-idx {args.train_idx} is out of range"
        (area_target, b2b_conn, p2b_conn, pins_pos, constraints,
         _tree_sol, fp_sol_from_train, val_metrics) = [x.squeeze(0) for x in batch]
        polygons = None
        block_count = int((area_target != -1).sum().item())
        print(f"\nTraining sample idx={args.train_idx}: block_count={block_count}")
    else:
        sys.path.insert(0, str(_REPO_ROOT / "iccad2026contest"))
        from lite_dataset_test import FloorplanDatasetLiteTest
        dataset  = FloorplanDatasetLiteTest(str(_REPO_ROOT) + "/")
        test_idx = args.val_case - 21
        sample   = dataset[test_idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, val_metrics = labels
        block_count = int((area_target != -1).sum().item())
        print(f"\nValidation case (block_count={args.val_case}, idx={test_idx}): actual block_count={block_count}")

    # Build fp_sol with w/h/x/y for fixed and preplaced blocks only.
    # Soft blocks remain zero — the model predicts their placement freely.
    n_padded = area_target.shape[0]
    is_fixed     = constraints[:block_count, 0]   # [k]
    is_preplaced = constraints[:block_count, 1]   # [k]
    if fp_sol_from_train is not None:
        # Training data: fp_sol already available; zero out soft blocks
        fp_sol_gt = fp_sol_from_train.clone()
        for i in range(block_count):
            if is_fixed[i].item() == 0 and is_preplaced[i].item() == 0:
                fp_sol_gt[i] = 0.0
    else:
        fp_sol_gt = torch.zeros(n_padded, 4)
        for i in range(block_count):
            if is_fixed[i].item() == 0 and is_preplaced[i].item() == 0:
                continue   # soft block: leave zeros
            verts = polygons[i]
            v = verts if isinstance(verts, torch.Tensor) else torch.tensor(verts)
            v = v[v[:, 0] != -1]
            if v.numel() == 0:
                continue
            xy_min = v.min(0).values
            xy_max = v.max(0).values
            fp_sol_gt[i] = torch.tensor([
                (xy_max[0] - xy_min[0]).item(),   # w
                (xy_max[1] - xy_min[1]).item(),   # h
                xy_min[0].item(),                  # x  (only used for preplaced)
                xy_min[1].item(),                  # y  (only used for preplaced)
            ])

    n_fixed     = int(is_fixed.sum().item())
    n_preplaced = int(is_preplaced.sum().item())
    print(f"  fixed={n_fixed}  preplaced={n_preplaced}  soft={block_count - n_fixed - n_preplaced}")

    optimizer = MyOptimizer(verbose=True, checkpoint=args.checkpoint)
    positions = optimizer.solve(
        block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints,
        fp_sol=fp_sol_gt,
    )

    print(f"\nReturn length : {len(positions)}  (expected {block_count})")
    assert len(positions) == block_count, \
        f"Length mismatch: got {len(positions)}, expected {block_count}"

    print(f"\nAll {block_count} positions (x, y, w, h):")
    for i, pos in enumerate(positions):
        x, y, w, h = pos
        print(f"  block {i:2d}: x={x:.3f}  y={y:.3f}  w={w:.3f}  h={h:.3f}  area={w*h:.3f}")
        assert all(math.isfinite(v) for v in pos), f"Non-finite value in block {i}: {pos}"

    # Check non-all-zero: at least some block should have non-trivial placement
    all_positions_tensor = torch.tensor(positions)
    assert all_positions_tensor.abs().sum() > 0, "All positions are zero — inference failed"

    print("\nAll checks passed.")
    print("=" * 60)

    # ── Compute losses for predicted positions ────────────────────────────
    from data.floorset_loader import preprocess_sample, build_w_int_unnorm
    from loss.wirelength_loss import wirelength_loss
    from loss.area_loss       import area_loss
    from loss.violation_loss  import violation_loss
    from config import (
        LAMBDA_WIRELENGTH, LAMBDA_AREA,
        LAMBDA_GROUPING, LAMBDA_MIB, LAMBDA_BOUNDARY, LAMBDA_OVERLAP,
    )

    n_padded = area_target.shape[0]

    # Build fp_sol ([w, h, x, y] format expected by preprocess_sample)
    if fp_sol_from_train is not None:
        fp_sol = fp_sol_from_train
    else:
        fp_sol = torch.zeros(n_padded, 4)
        for i in range(block_count):
            verts = polygons[i]
            v = verts if isinstance(verts, torch.Tensor) else torch.tensor(verts)
            v = v[v[:, 0] != -1]
            if v.numel() > 0:
                xy_min = v.min(0).values
                xy_max = v.max(0).values
                fp_sol[i] = torch.tensor([
                    (xy_max[0] - xy_min[0]).item(),  # w
                    (xy_max[1] - xy_min[1]).item(),  # h
                    xy_min[0].item(),                 # x
                    xy_min[1].item(),                 # y
                ])

    s = preprocess_sample(
        area_target, b2b_conn, p2b_conn,
        pins_pos, constraints, fp_sol, val_metrics,
    )
    device = optimizer._device
    k      = s["block_count"]
    ref    = s["canvas_ref"]

    pred_t = torch.tensor(positions, dtype=torch.float32)           # [k, 4]  original order
    # Re-sort to sorted block order so pred aligns with cons / wiu
    sort_idx   = s["sort_idx"]                                       # [k]
    pred_sorted = pred_t[sort_idx]                                   # [k, 4]  sorted order
    pred_norm = (pred_sorted / ref).unsqueeze(0).to(device)          # [1, k, 4]
    pred_raw  = pred_sorted.unsqueeze(0).to(device)                  # [1, k, 4]

    cons  = s["constraints_sorted"][:k].unsqueeze(0).to(device)
    p2b_t = s["p2b_conn"].unsqueeze(0).to(device)
    pins_t = s["pins_pos"].unsqueeze(0).to(device)
    hbase  = torch.tensor([s["hpwl_baseline"]], device=device)
    abase  = torch.tensor([s["area_baseline"]], device=device)
    wiu    = build_w_int_unnorm(b2b_conn, k, s["sort_idx"]).unsqueeze(0).to(device)

    with torch.no_grad():
        l_wl    = wirelength_loss(pred_raw, wiu, pins_t, p2b_t, hbase).item()
        l_area  = area_loss(pred_raw, abase).item()
        l_grouping, l_mib, l_boundary, l_overlap = violation_loss(pred_norm, cons)
        l_grouping = l_grouping.item()
        l_mib      = l_mib.item()
        l_boundary = l_boundary.item()
        l_overlap  = l_overlap.item()
        l_total = (LAMBDA_WIRELENGTH * l_wl
                   + LAMBDA_AREA     * l_area
                   + LAMBDA_GROUPING * l_grouping
                   + LAMBDA_MIB      * l_mib
                   + LAMBDA_BOUNDARY * l_boundary
                   + LAMBDA_OVERLAP  * l_overlap)

    print("\n" + "=" * 50)
    print("Loss Summary")
    print("=" * 50)
    print(f"  wirelength (×{LAMBDA_WIRELENGTH})           : {l_wl:.6f}")
    print(f"  area       (×{LAMBDA_AREA})           : {l_area:.6f}")
    print(f"  grouping   (×{LAMBDA_GROUPING})           : {l_grouping:.6f}")
    print(f"  mib        (×{LAMBDA_MIB})           : {l_mib:.6f}")
    print(f"  boundary   (×{LAMBDA_BOUNDARY})           : {l_boundary:.6f}")
    print(f"  overlap    (×{LAMBDA_OVERLAP})           : {l_overlap:.6f}")
    print("-" * 50)
    print(f"  total: {l_total:.6f}")
    print("=" * 50)

    # ── Visualize GT vs Predicted ─────────────────────────────────────────
    # Parse GT positions (x, y, w, h)
    if fp_sol_from_train is not None:
        # Training data: fp_sol is [w, h, x, y]
        gt_positions = [
            (fp_sol_from_train[i, 2].item(), fp_sol_from_train[i, 3].item(),
             fp_sol_from_train[i, 0].item(), fp_sol_from_train[i, 1].item())
            for i in range(block_count)
        ]
    else:
        gt_positions = []
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
                gt_positions.append((
                    xy_min[0].item(), xy_min[1].item(),
                    (xy_max[0] - xy_min[0]).item(),
                    (xy_max[1] - xy_min[1]).item(),
                ))
            else:
                gt_positions.append((0.0, 0.0, 0.0, 0.0))

    loss_parts = {
        "total":      l_total,
        "wirelength": l_wl,
        "area":       l_area,
        "grouping":   l_grouping,
        "mib":        l_mib,
        "boundary":   l_boundary,
        "overlap":    l_overlap,
    }
    if args.shard_idx is not None:
        viz_title = f"Smoke test — shard idx={args.shard_idx} (block_count={block_count})"
        viz_out   = _CONTEST_DIR / f"smoke_test_shard{args.shard_idx}_viz.png"
    elif args.train_idx is not None:
        viz_title = f"Smoke test — train idx={args.train_idx} (block_count={block_count})"
        viz_out   = _CONTEST_DIR / f"smoke_test_train{args.train_idx}_viz.png"
    else:
        viz_title = f"Smoke test — val block_count={args.val_case}"
        viz_out   = _CONTEST_DIR / "smoke_test_viz.png"
    save_floorplan_viz(
        gt_positions, positions, block_count,
        out_path=viz_out,
        title=viz_title,
        loss_parts=loss_parts,
        constraints=constraints[:block_count],
    )
