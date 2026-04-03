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
# TRANSFORMER OPTIMIZER
# =============================================================================

class MyOptimizer(FloorplanOptimizer):
    """
    Floorplan Transformer — autoregressive inference.

    Loads the best (or latest) checkpoint from solution/checkpoints/ and
    runs the trained Encoder-Decoder Transformer in AR mode inside solve().
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._model  = None
        self._device = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self):
        from config import CHECKPOINT_DIR
        from model.transformer_floorplan import TransformerFloorplan

        best_ck   = CHECKPOINT_DIR / "best.pt"
        latest_ck = CHECKPOINT_DIR / "latest.pt"

        if best_ck.exists():
            ck_path = best_ck
        elif latest_ck.exists():
            ck_path = latest_ck
        else:
            ck_path = None

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = TransformerFloorplan().to(device)

        if ck_path is not None:
            ck = torch.load(ck_path, map_location=device, weights_only=False)
            model.load_state_dict(ck["model_state"])
            if self.verbose:
                print(f"[MyOptimizer] Loaded checkpoint: {ck_path.name}")
        else:
            if self.verbose:
                print("[MyOptimizer] No checkpoint found — using random init.")

        model.eval()
        self._model  = model
        self._device = device

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
    ) -> List[Tuple[float, float, float, float]]:
        """
        Transformer autoregressive inference.

        Returns List[(x, y, w, h)], length = block_count, in original block order.
        """
        from data.floorset_loader import preprocess_sample

        # Build dummy fp_sol (zeros): features 4-7 will be 0 at inference
        # (target_w/h/x/y unavailable without GT solution)
        n_padded = area_targets.shape[0]
        fp_sol   = torch.zeros(n_padded, 4)   # [w, h, x, y] — zeroed
        metrics  = torch.zeros(8)

        # Full preprocessing pipeline (sort, features, w_int, …)
        s = preprocess_sample(
            area_targets, b2b_connectivity, p2b_connectivity,
            pins_pos, constraints, fp_sol, metrics,
        )

        k  = s["block_count"]
        tf = s["token_features"].unsqueeze(0).to(self._device)  # [1, k, 18]
        wi = s["w_int"].unsqueeze(0).to(self._device)            # [1, k, k]

        with torch.no_grad():
            pred_norm = self._model(tf, wi, teacher_forcing=False)  # [1, k, 4]

        # De-normalise to raw pixel scale
        pred_sorted = pred_norm[0].cpu() * s["canvas_ref"]          # [k, 4]

        # Restore original block order using sort_inv:
        #   sort_inv[j] = sorted position of original block j
        #   pred_original[j] = pred_sorted[sort_inv[j]]
        sort_inv     = s["sort_inv"]                                 # [k]
        pred_original = pred_sorted[sort_inv]                        # [k, 4]

        return [
            (
                float(pred_original[i, 0]),  # x
                float(pred_original[i, 1]),  # y
                float(pred_original[i, 2]),  # w
                float(pred_original[i, 3]),  # h
            )
            for i in range(k)
        ]


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-case", type=int, default=21,
                        help="Block count of the validation case to visualise (default: 21)")
    args = parser.parse_args()

    print("=" * 60)
    print("my_optimizer.py  —  smoke test")
    print("=" * 60)

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

    optimizer = MyOptimizer(verbose=True)
    positions = optimizer.solve(
        block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints
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
    from loss.coord_loss      import coord_loss
    from loss.wirelength_loss import wirelength_loss
    from loss.area_loss       import area_loss
    from loss.violation_loss  import violation_loss
    from config import (
        LAMBDA_WIRELENGTH, LAMBDA_AREA, LAMBDA_VIOLATION,
    )

    n_padded = area_target.shape[0]

    # Build fp_sol from GT polygons ([w, h, x, y] format expected by preprocess_sample)
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
    # Re-sort to sorted block order so pred aligns with gtn / cons / wiu
    sort_idx   = s["sort_idx"]                                       # [k]
    pred_sorted = pred_t[sort_idx]                                   # [k, 4]  sorted order
    pred_norm = (pred_sorted / ref).unsqueeze(0).to(device)          # [1, k, 4]
    pred_raw  = pred_sorted.unsqueeze(0).to(device)                  # [1, k, 4]

    gtn   = s["gt_positions_norm"][:k].unsqueeze(0).to(device)
    cons  = s["constraints_sorted"][:k].unsqueeze(0).to(device)
    p2b_t = s["p2b_conn"].unsqueeze(0).to(device)
    pins_t = s["pins_pos"].unsqueeze(0).to(device)
    hbase  = torch.tensor([s["hpwl_baseline"]], device=device)
    abase  = torch.tensor([s["area_baseline"]], device=device)
    wiu    = build_w_int_unnorm(b2b_conn, k, s["sort_idx"]).unsqueeze(0).to(device)

    # Run TF forward to obtain logits for coord_loss
    tf_t_  = s["token_features"][:k].unsqueeze(0).to(device)   # [1, k, 18]
    wi_t_  = s["w_int"][:k, :k].unsqueeze(0).to(device)         # [1, k, k]
    kpm_   = torch.zeros(1, k, dtype=torch.bool, device=device) # no padding

    with torch.no_grad():
        _, logits_tf = optimizer._model(tf_t_, wi_t_, gt_positions=gtn,
                                        teacher_forcing=True)    # [1, k, G*G]
        l_coord = coord_loss(logits_tf, gtn, cons, kpm_).item()
        l_wl    = wirelength_loss(pred_raw, wiu, pins_t, p2b_t, hbase).item()
        l_area  = area_loss(pred_raw, abase).item()
        l_viol  = violation_loss(pred_norm, cons).item()
        l_total = (l_coord
                   + LAMBDA_WIRELENGTH * l_wl
                   + LAMBDA_AREA       * l_area
                   + LAMBDA_VIOLATION  * l_viol)

    print("\n" + "=" * 50)
    print("Loss Summary")
    print("=" * 50)
    print(f"  coord      (×1.0) : {l_coord:.6f}")
    print(f"  wirelength (×{LAMBDA_WIRELENGTH})  : {l_wl:.6f}")
    print(f"  area       (×{LAMBDA_AREA})  : {l_area:.6f}")
    print(f"  violation  (×{LAMBDA_VIOLATION})  : {l_viol:.6f}")
    print("-" * 50)
    print(f"  total              : {l_total:.6f}")
    print("=" * 50)

    # ── Visualize GT vs Predicted ─────────────────────────────────────────
    # Parse GT positions from polygons
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
        "coord":      l_coord,
        "wirelength": l_wl,
        "area":       l_area,
        "violation":  l_viol,
    }
    save_floorplan_viz(
        gt_positions, positions, block_count,
        out_path=_CONTEST_DIR / "smoke_test_viz.png",
        title="Smoke test — validation case 0",
        loss_parts=loss_parts,
        constraints=constraints[:block_count],
    )
