#!/usr/bin/env python3
"""
ICCAD 2026 FloorSet Challenge - Contest Evaluator (Spec-Compliant)

依照 FloorplanningContest_ICCAD_2026_v7.pdf 完整實作的評分系統。

與 iccad2026_evaluate.py 的差異：
  1. HPWL_int (b2b):  bounding box 半周長（spec Eq. on p.6），非 centroid 距離
  2. HPWL_ext (p2b):  block 中心到 terminal 的 Manhattan 距離（同 spec，現有正確）
  3. Soft constraints: 正確計算全部 5 種違反類型（現有全為 0）
       - V_fixed     : 尺寸不符的 fixed-shape block 數
       - V_preplaced : 位置或尺寸不符的 preplaced block 數
       - V_boundary  : 未接觸 bbox 指定邊/角的 block 數
       - V_grouping  : 每個 grouping group 的 connected components - 1
       - V_mib       : 每個 MIB group 的 distinct (w,h) 對數 - 1
  4. N_soft:          依 spec Eq.3 正規化（含 group size: Σ(|G_p|-1), Σ(|M_q|-1)）
  5. Total Score:     依 block count 指數加權 Σ Cost[i]·e^{n_i} / Σ e^{n_j}（spec p.9）
  6. Runtime:         固定 RuntimeFactor = 1.0（不計算）

Usage:
    python contest_evaluate.py --evaluate my_optimizer.py
    python contest_evaluate.py --evaluate my_optimizer.py --test-id 0 --verbose
    python contest_evaluate.py --evaluate my_optimizer.py --output results.json
"""

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest
from iccad2026_evaluate import FloorplanOptimizer  # 複用 base class

# =============================================================================
# 比賽參數 (spec p.5)
# =============================================================================
ALPHA = 0.5        # quality metrics 權重
BETA = 2.0         # 指數違反懲罰
GAMMA = 0.3        # runtime 衰減
M_PENALTY = 10.0   # 不可行解懲罰
AREA_TOLERANCE = 0.01  # 1% 面積容差
RUNTIME_FACTOR = 1.0   # 固定不計算 runtime

# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ContestMetrics:
    is_feasible: bool
    overlap_violations: int
    area_violations: int
    # HPWL
    hpwl_int: float       # b2b bounding box HPWL
    hpwl_ext: float       # p2b centroid-to-terminal HPWL
    hpwl_total: float
    hpwl_baseline: float
    hpwl_gap: float
    # Bounding box area
    bbox_area: float
    bbox_area_baseline: float
    area_gap: float
    # Soft constraint violations (spec Eq.3)
    v_fixed: int
    v_preplaced: int
    v_boundary: int
    v_grouping: int
    v_mib: int
    total_soft_violations: int
    n_soft: int            # 正規化因子
    violations_relative: float
    # Cost
    cost: float


@dataclass
class TestResult:
    test_id: int
    block_count: int
    is_feasible: bool
    hpwl_gap: float
    area_gap: float
    violations_relative: float
    cost: float
    positions: Optional[List[Tuple[float, float, float, float]]] = None
    error: Optional[str] = None


# =============================================================================
# HPWL 計算（依 spec）
# =============================================================================

def calculate_hpwl_int(
    positions: List[Tuple[float, float, float, float]],
    b2b_connectivity: torch.Tensor
) -> float:
    """
    HPWL_int：inter-module wirelength，使用兩個 block 的 bounding box 半周長。

    spec (p.6):
      HPWL_int = Σ_{i=1}^{k} Σ_{j>i} W_ij^(int)
                 · (max(xi+wi, xj+wj) - min(xi, xj)
                  + max(yi+hi, yj+hj) - min(yi, yj))

    注意：這與 centroid 距離 |cx_i - cx_j| + |cy_i - cy_j| 不同！
    Bounding box 公式 ≥ centroid 距離（等號在兩 block 不重疊且 centroid 在 bbox 邊界時成立）。
    """
    if b2b_connectivity is None or len(b2b_connectivity) == 0:
        return 0.0

    total = 0.0
    for edge in b2b_connectivity:
        if float(edge[0]) == -1:
            continue
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        if i >= len(positions) or j >= len(positions):
            continue
        xi, yi, wi, hi = positions[i]
        xj, yj, wj, hj = positions[j]
        bbox_width  = max(xi + wi, xj + wj) - min(xi, xj)
        bbox_height = max(yi + hi, yj + hj) - min(yi, yj)
        total += w * (bbox_width + bbox_height)
    return total


def calculate_hpwl_ext(
    positions: List[Tuple[float, float, float, float]],
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor
) -> float:
    """
    HPWL_ext：external connectivity，使用 block 中心到 terminal 的 Manhattan 距離。

    spec (p.7):
      HPWL_ext = Σ_{i=1}^{k} Σ_{j=1}^{r} W_ij^(ext)
                 · (|xi + wi/2 - x_tj| + |yi + hi/2 - y_tj|)
    """
    if p2b_connectivity is None or len(p2b_connectivity) == 0:
        return 0.0
    if pins_pos is None or len(pins_pos) == 0:
        return 0.0

    total = 0.0
    for edge in p2b_connectivity:
        if float(edge[0]) == -1:
            continue
        pin_idx, block_idx, w = int(edge[0]), int(edge[1]), float(edge[2])
        if pin_idx >= len(pins_pos) or block_idx >= len(positions):
            continue
        px, py = float(pins_pos[pin_idx][0]), float(pins_pos[pin_idx][1])
        xi, yi, wi, hi = positions[block_idx]
        cx = xi + wi * 0.5
        cy = yi + hi * 0.5
        total += w * (abs(cx - px) + abs(cy - py))
    return total


# =============================================================================
# Hard Constraint Checks（同 iccad2026_evaluate.py）
# =============================================================================

def check_overlap(positions: List[Tuple[float, float, float, float]]) -> int:
    """兩個 block 實際重疊（接觸邊不算）的違反數。"""
    violations = 0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, w1, h1 = positions[i]
            x2, y2, w2, h2 = positions[j]
            overlap_x = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
            overlap_y = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
            if overlap_x > 1e-6 and overlap_y > 1e-6:
                violations += 1
    return violations


def check_area_tolerance(
    positions: List[Tuple[float, float, float, float]],
    area_targets: torch.Tensor,
    tolerance: float = AREA_TOLERANCE
) -> int:
    """w*h 是否在 area_targets[i] 的 1% 容差內。"""
    violations = 0
    for i, (x, y, w, h) in enumerate(positions):
        if i >= len(area_targets) or float(area_targets[i]) == -1:
            continue
        target = float(area_targets[i])
        if target <= 0:
            continue
        if abs(w * h - target) / target > tolerance:
            violations += 1
    return violations


def calculate_bbox_area(positions: List[Tuple[float, float, float, float]]) -> float:
    """所有 block 的 bounding box 面積。"""
    if not positions:
        return 0.0
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    return (x_max - x_min) * (y_max - y_min)


# =============================================================================
# Soft Constraint Violations（spec p.7–8）
# =============================================================================

def _blocks_share_edge(
    xi: float, yi: float, wi: float, hi: float,
    xj: float, yj: float, wj: float, hj: float,
    tol: float = 1e-4
) -> bool:
    """
    兩個矩形是否共用一段邊（非僅角點接觸）。
    Grouping 要求 blocks 共用 non-zero length 的邊段。
    """
    # 左右接觸
    if abs((xi + wi) - xj) < tol or abs((xj + wj) - xi) < tol:
        y_overlap = min(yi + hi, yj + hj) - max(yi, yj)
        return y_overlap > tol
    # 上下接觸
    if abs((yi + hi) - yj) < tol or abs((yj + hj) - yi) < tol:
        x_overlap = min(xi + wi, xj + wj) - max(xi, xj)
        return x_overlap > tol
    return False


def _count_connected_components(n: int, adj: Dict[int, List[int]]) -> int:
    """BFS 計算連通分量數量。"""
    visited = [False] * n
    cc = 0
    for start in range(n):
        if not visited[start]:
            cc += 1
            queue = [start]
            while queue:
                node = queue.pop()
                if visited[node]:
                    continue
                visited[node] = True
                queue.extend(adj[node])
    return cc


def compute_soft_violations(
    positions: List[Tuple[float, float, float, float]],
    constraints: torch.Tensor,
    target_positions: List[Tuple[float, float, float, float]],
    tol: float = 1e-3
) -> Dict[str, int]:
    """
    計算所有 soft constraint violations 及正規化因子 N_soft。

    spec Eq.3 (p.7):
      Violations_relative = (V_fixed + V_preplaced + V_grouping + V_boundary + V_mib) / N_soft

      N_soft = |B_fixed| + |B_preplaced| + |B_boundary|
             + Σ_{p=1}^{P} (|G_p| - 1)   ← grouping groups
             + Σ_{q=1}^{Q} (|M_q| - 1)   ← MIB groups

    Constraints tensor columns:
      [0] fixed       : fixed-shape block（1 = 是）
      [1] preplaced   : preplaced block（1 = 是）
      [2] MIB group   : MIB group ID（0 = 不屬於任何 group）
      [3] cluster/grouping group ID（0 = 不屬於任何 group）
      [4] boundary    : bitmask 1=left, 2=right, 4=top, 8=bottom

    Returns: dict with keys v_fixed, v_preplaced, v_boundary, v_grouping, v_mib, n_soft
    """
    n = len(positions)
    if constraints is None or len(constraints) == 0:
        return dict(v_fixed=0, v_preplaced=0, v_boundary=0,
                    v_grouping=0, v_mib=0, n_soft=1)

    # --- 解析 constraint ---
    fixed_blocks     = []
    preplaced_blocks = []
    boundary_blocks  = []  # (block_idx, btype)
    grouping_groups  = {}  # group_id -> [block_idx, ...]
    mib_groups       = {}  # group_id -> [block_idx, ...]

    for i in range(n):
        if i >= len(constraints):
            break
        c = constraints[i]
        if float(c[0]) != 0:
            fixed_blocks.append(i)
        if float(c[1]) != 0:
            preplaced_blocks.append(i)
        mib_id = int(float(c[2]))
        if mib_id > 0:
            mib_groups.setdefault(mib_id, []).append(i)
        grp_id = int(float(c[3]))
        if grp_id > 0:
            grouping_groups.setdefault(grp_id, []).append(i)
        btype = int(float(c[4]))
        if btype != 0:
            boundary_blocks.append((i, btype))

    # --- N_soft（正規化因子）---
    n_grouping_norm = sum(max(len(g) - 1, 0) for g in grouping_groups.values())
    n_mib_norm      = sum(max(len(g) - 1, 0) for g in mib_groups.values())
    n_soft = (len(fixed_blocks) + len(preplaced_blocks) + len(boundary_blocks)
              + n_grouping_norm + n_mib_norm)
    if n_soft == 0:
        return dict(v_fixed=0, v_preplaced=0, v_boundary=0,
                    v_grouping=0, v_mib=0, n_soft=1)

    # --- V_fixed: dimensions 不符的 fixed-shape blocks ---
    # spec: 1_b = 1 if w_b ≠ w_b^input OR h_b ≠ h_b^input
    v_fixed = 0
    for i in fixed_blocks:
        if i >= len(target_positions):
            continue
        _, _, w,  h  = positions[i]
        _, _, tw, th = target_positions[i]
        if abs(w - tw) > tol or abs(h - th) > tol:
            v_fixed += 1

    # --- V_preplaced: location 或 dimensions 不符的 preplaced blocks ---
    # spec: 1_b = 1 if (x,y) ≠ target OR (w,h) ≠ target
    v_preplaced = 0
    for i in preplaced_blocks:
        if i >= len(target_positions):
            continue
        x,  y,  w,  h  = positions[i]
        tx, ty, tw, th = target_positions[i]
        if (abs(x - tx) > tol or abs(y - ty) > tol
                or abs(w - tw) > tol or abs(h - th) > tol):
            v_preplaced += 1

    # --- V_boundary: 未觸及 solution bounding box 指定邊/角 ---
    # spec: boundary 是解的 bounding box（非固定 canvas）
    v_boundary = 0
    if boundary_blocks and n > 0:
        x_min = min(p[0]        for p in positions)
        y_min = min(p[1]        for p in positions)
        x_max = max(p[0] + p[2] for p in positions)
        y_max = max(p[1] + p[3] for p in positions)

        for (i, btype) in boundary_blocks:
            xi, yi, wi, hi = positions[i]
            touches_left   = abs(xi         - x_min) < tol
            touches_right  = abs(xi + wi    - x_max) < tol
            touches_top    = abs(yi + hi    - y_max) < tol
            touches_bottom = abs(yi         - y_min) < tol

            # bitmask: 1=left, 2=right, 4=top, 8=bottom
            ok = True
            if btype & 1:  ok = ok and touches_left
            if btype & 2:  ok = ok and touches_right
            if btype & 4:  ok = ok and touches_top
            if btype & 8:  ok = ok and touches_bottom
            if not ok:
                v_boundary += 1

    # --- V_grouping: 每個 group 的 connected components - 1 ---
    # spec: A group is satisfied if all its blocks form a single connected
    #       component through shared edges.
    #       V_grouping = Σ_p (c_p - 1), where c_p = number of connected components
    v_grouping = 0
    for grp_id, block_idxs in grouping_groups.items():
        ng = len(block_idxs)
        if ng <= 1:
            continue
        # 建立鄰接表：同 group 內共用邊的 block 互連
        local_adj = {k: [] for k in range(ng)}
        for a in range(ng):
            for b in range(a + 1, ng):
                bi, bj = block_idxs[a], block_idxs[b]
                xi, yi, wi, hi = positions[bi]
                xj, yj, wj, hj = positions[bj]
                if _blocks_share_edge(xi, yi, wi, hi, xj, yj, wj, hj, tol):
                    local_adj[a].append(b)
                    local_adj[b].append(a)
        cc = _count_connected_components(ng, local_adj)
        v_grouping += (cc - 1)

    # --- V_mib: 每個 MIB group 的 distinct (w,h) 對數 - 1 ---
    # spec: s_q = number of distinct (w,h) pairs; violation = s_q - 1
    v_mib = 0
    for mib_id, block_idxs in mib_groups.items():
        if len(block_idxs) <= 1:
            continue
        # 四捨五入到 tol 精度，再取 distinct 對數
        shapes = set()
        for i in block_idxs:
            _, _, w, h = positions[i]
            rw = round(w / tol) * tol
            rh = round(h / tol) * tol
            shapes.add((rw, rh))
        v_mib += (len(shapes) - 1)

    return dict(
        v_fixed=v_fixed,
        v_preplaced=v_preplaced,
        v_boundary=v_boundary,
        v_grouping=v_grouping,
        v_mib=v_mib,
        n_soft=n_soft
    )


# =============================================================================
# Cost 計算（spec Eq.2）
# =============================================================================

def compute_cost(
    hpwl_gap: float,
    area_gap: float,
    violations_relative: float,
    runtime_factor: float,
    is_feasible: bool
) -> float:
    """
    spec Eq.2 (p.5):
      Cost = (1 + α·(HPWL_gap + Area_gap)) × e^{β·V_rel} × max(0.7, R^γ)
           = M  if infeasible

    這裡 runtime_factor 由呼叫端傳入（本模組固定為 1.0）。
    """
    if not is_feasible:
        return M_PENALTY

    quality_factor   = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    violation_factor = math.exp(BETA * violations_relative)
    runtime_adj      = max(0.7, math.pow(max(0.01, runtime_factor), GAMMA))

    return quality_factor * violation_factor * runtime_adj


# =============================================================================
# Total Score（spec p.9）
# =============================================================================

def compute_total_score(results: List[TestResult]) -> float:
    """
    spec (p.9):
      Total Score = Σ_{i=21}^{120} Cost[i] · e^{n_i} / Σ_{j=21}^{120} e^{n_j}

    n_i = block count of test case i（大 block count → 指數大 → 權重大）。
    Lower Total Score is better。完美解的 Total Score ≈ 0.7（最快 runtime）或 1.0（median runtime）。
    """
    if not results:
        return 0.0

    weights = [math.exp(r.block_count) for r in results]
    total_w = sum(weights)
    if total_w == 0:
        return sum(r.cost for r in results) / len(results)

    return sum(r.cost * w for r, w in zip(results, weights)) / total_w


# =============================================================================
# 完整評分（單一 test case）
# =============================================================================

def evaluate_solution_spec(
    positions: List[Tuple[float, float, float, float]],
    target_positions: List[Tuple[float, float, float, float]],
    baseline: Dict[str, float],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
) -> ContestMetrics:
    """
    依 spec 計算單一 test case 的完整 metrics。

    Args:
        positions:        優化器輸出的 [(x, y, w, h), ...] 列表
        target_positions: ground truth 的 [(x, y, w, h), ...] 列表
        baseline:         {'hpwl_baseline': float, 'area_baseline': float}
        b2b_connectivity: [edges, 3] (i, j, weight)
        p2b_connectivity: [edges, 3] (pin_idx, block_idx, weight)
        pins_pos:         [n_pins, 2] (x, y)
        area_targets:     [n] 各 block 的目標面積
        constraints:      [n, 5] (fixed, preplaced, MIB, cluster, boundary)
    """
    # --- Hard constraints ---
    overlap_viol = check_overlap(positions)
    area_viol    = check_area_tolerance(positions, area_targets)
    is_feasible  = (overlap_viol == 0) and (area_viol == 0)

    # --- HPWL（spec 公式）---
    hpwl_int   = calculate_hpwl_int(positions, b2b_connectivity)
    hpwl_ext   = calculate_hpwl_ext(positions, p2b_connectivity, pins_pos)
    hpwl_total = hpwl_int + hpwl_ext

    hpwl_base = baseline.get('hpwl_baseline', hpwl_total)
    hpwl_gap  = (hpwl_total - hpwl_base) / max(hpwl_base, 1e-6)

    # --- Bounding box area ---
    bbox_area      = calculate_bbox_area(positions)
    area_base      = baseline.get('area_baseline', bbox_area)
    area_gap       = (bbox_area - area_base) / max(area_base, 1e-6)

    # --- Soft constraint violations（spec Eq.3）---
    viol = compute_soft_violations(positions, constraints, target_positions)
    v_rel = viol['v_fixed'] + viol['v_preplaced'] + viol['v_boundary'] \
          + viol['v_grouping'] + viol['v_mib']
    violations_relative = v_rel / max(viol['n_soft'], 1)

    # --- Cost（spec Eq.2, runtime=1.0）---
    cost = compute_cost(hpwl_gap, area_gap, violations_relative,
                        runtime_factor=RUNTIME_FACTOR, is_feasible=is_feasible)

    return ContestMetrics(
        is_feasible=is_feasible,
        overlap_violations=overlap_viol,
        area_violations=area_viol,
        hpwl_int=hpwl_int,
        hpwl_ext=hpwl_ext,
        hpwl_total=hpwl_total,
        hpwl_baseline=hpwl_base,
        hpwl_gap=hpwl_gap,
        bbox_area=bbox_area,
        bbox_area_baseline=area_base,
        area_gap=area_gap,
        v_fixed=viol['v_fixed'],
        v_preplaced=viol['v_preplaced'],
        v_boundary=viol['v_boundary'],
        v_grouping=viol['v_grouping'],
        v_mib=viol['v_mib'],
        total_soft_violations=v_rel,
        n_soft=viol['n_soft'],
        violations_relative=violations_relative,
        cost=cost,
    )


# =============================================================================
# Baseline 提取（從 ground truth 計算，使用 spec 公式）
# =============================================================================

def extract_baseline_from_labels(
    polygons,
    metrics,
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
    block_count: int
) -> Tuple[Dict[str, float], List[Tuple[float, float, float, float]]]:
    """
    從 dataset ground truth 提取 baseline metrics 及 target positions。

    HPWL baseline 用 spec 公式（HPWL_int bounding box + HPWL_ext centroid）重新計算，
    不直接使用 metrics[-2]（那是 centroid-based b2b，與 spec 不同）。
    Area baseline 優先使用 metrics[0]（ground truth bounding box area）。
    """
    # 從 polygon 頂點提取 (x_min, y_min, w, h)
    target_positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min = float(valid[:, 0].min())
            y_min = float(valid[:, 1].min())
            x_max = float(valid[:, 0].max())
            y_max = float(valid[:, 1].max())
            target_positions.append((x_min, y_min, x_max - x_min, y_max - y_min))
        else:
            target_positions.append((0.0, 0.0, 1.0, 1.0))

    # 用 spec 公式重新計算 baseline HPWL
    hpwl_int_base = calculate_hpwl_int(target_positions, b2b_conn)
    hpwl_ext_base = calculate_hpwl_ext(target_positions, p2b_conn, pins_pos)
    hpwl_baseline = hpwl_int_base + hpwl_ext_base

    # Area baseline：優先使用 metrics[0]（ground truth）
    area_baseline = calculate_bbox_area(target_positions)
    if metrics is not None and len(metrics) >= 1 and float(metrics[0]) > 0:
        area_baseline = float(metrics[0])

    baseline = {
        'hpwl_baseline': hpwl_baseline,
        'area_baseline': area_baseline,
    }
    return baseline, target_positions


# =============================================================================
# Evaluator
# =============================================================================

class ContestEvaluator:
    """依 spec 的完整評分引擎。"""

    def __init__(self, data_path: str = "../", verbose: bool = True):
        self.data_path = Path(data_path)
        self.verbose   = verbose
        self.dataset   = None

    def _load_dataset(self):
        if self.dataset is None:
            if self.verbose:
                print("Loading validation dataset...")
            self.dataset = FloorplanDatasetLiteTest(str(self.data_path))
            if self.verbose:
                print(f"Loaded {len(self.dataset)} validation cases")

    def _load_optimizer(self, optimizer_path: str) -> FloorplanOptimizer:
        path = Path(optimizer_path)
        if not path.exists():
            raise FileNotFoundError(f"Optimizer file not found: {optimizer_path}")
        spec = importlib.util.spec_from_file_location("optimizer_module", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for name in dir(module):
            obj = getattr(module, name)
            if (isinstance(obj, type)
                    and issubclass(obj, FloorplanOptimizer)
                    and obj is not FloorplanOptimizer):
                return obj(verbose=self.verbose)
        for name in ['MyOptimizer', 'Optimizer', 'ContestOptimizer']:
            if hasattr(module, name):
                return getattr(module, name)(verbose=self.verbose)
        raise ValueError(f"No FloorplanOptimizer subclass found in {optimizer_path}")

    def evaluate(
        self,
        optimizer_path: str,
        test_ids: Optional[List[int]] = None,
        output_path: Optional[str] = None,
    ) -> List[TestResult]:
        self._load_dataset()
        optimizer = self._load_optimizer(optimizer_path)

        if test_ids is None:
            test_ids = list(range(len(self.dataset)))

        results: List[TestResult] = []
        iterator = tqdm(test_ids, desc="Evaluating") if self.verbose else test_ids

        for idx in iterator:
            try:
                sample = self.dataset[idx]
                inputs, labels  = sample['input'], sample['label']
                area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
                polygons, metrics = labels
                block_count = int((area_target != -1).sum().item())

                baseline, target_pos = extract_baseline_from_labels(
                    polygons, metrics, b2b_conn, p2b_conn, pins_pos, block_count
                )

                # 若 optimizer 支援 set_target_positions()，傳入 ground truth 供 fixed/preplaced 使用
                if hasattr(optimizer, 'set_target_positions'):
                    optimizer.set_target_positions(target_pos)

                # 執行 optimizer
                positions = optimizer.solve(
                    block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints
                )

                # 評分
                cm = evaluate_solution_spec(
                    positions, target_pos, baseline,
                    b2b_conn, p2b_conn, pins_pos,
                    area_target, constraints,
                )

                results.append(TestResult(
                    test_id=idx,
                    block_count=block_count,
                    is_feasible=cm.is_feasible,
                    hpwl_gap=cm.hpwl_gap,
                    area_gap=cm.area_gap,
                    violations_relative=cm.violations_relative,
                    cost=cm.cost,
                    positions=positions,
                ))

                if self.verbose:
                    self._print_case(idx, block_count, cm)

            except Exception as e:
                import traceback
                results.append(TestResult(
                    test_id=idx, block_count=0, is_feasible=False,
                    hpwl_gap=0.0, area_gap=0.0, violations_relative=1.0,
                    cost=M_PENALTY, error=str(e),
                ))
                if self.verbose:
                    print(f"  [ERROR] test_id={idx}: {e}")

        self._print_summary(results, Path(optimizer_path).stem)

        if output_path:
            self._save_results(results, output_path, Path(optimizer_path).stem)

        return results

    def _print_case(self, idx: int, block_count: int, cm: ContestMetrics):
        feasible = "FEASIBLE" if cm.is_feasible else "INFEASIBLE"
        print(
            f"  [{idx:3d}] n={block_count:3d} | {feasible:10s} | "
            f"HPWL_gap={cm.hpwl_gap:+.4f}  Area_gap={cm.area_gap:+.4f} | "
            f"V_rel={cm.violations_relative:.3f} "
            f"(fix={cm.v_fixed} pre={cm.v_preplaced} bnd={cm.v_boundary} "
            f"grp={cm.v_grouping} mib={cm.v_mib} / N={cm.n_soft}) | "
            f"Cost={cm.cost:.4f}"
        )

    def _print_summary(self, results: List[TestResult], name: str):
        costs   = [r.cost for r in results]
        n_feas  = sum(1 for r in results if r.is_feasible)
        total_s = compute_total_score(results)
        avg_c   = sum(costs) / len(costs) if costs else 0.0

        print()
        print("=" * 60)
        print(f"EVALUATION RESULTS (spec-compliant): {name}")
        print("=" * 60)
        print(f"Total Score (exp-weighted): {total_s:.4f}")
        print(f"Tests:    {len(results)}")
        print(f"Feasible: {n_feas}")
        print(f"Avg Cost: {avg_c:.4f}")
        print(f"Note: RuntimeFactor fixed at {RUNTIME_FACTOR} (not evaluated)")
        print("=" * 60)

    def _save_results(self, results: List[TestResult], output_path: str, name: str):
        total_s = compute_total_score(results)
        costs   = [r.cost for r in results]
        data = {
            'submission_name': name,
            'timestamp': datetime.now().isoformat(),
            'total_score': total_s,
            'avg_cost': sum(costs) / len(costs) if costs else 0.0,
            'num_feasible': sum(1 for r in results if r.is_feasible),
            'runtime_factor_note': f'Fixed at {RUNTIME_FACTOR} (not evaluated)',
            'test_results': [
                {k: v for k, v in asdict(r).items() if k != 'positions'}
                for r in results
            ],
        }
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Results saved to {output_path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ICCAD 2026 FloorSet - Spec-Compliant Evaluator"
    )
    parser.add_argument('--evaluate', metavar='OPTIMIZER_PY',
                        help='optimizer .py 檔案路徑')
    parser.add_argument('--test-id', type=int, default=None,
                        help='只評估指定的 test case（0-99）')
    parser.add_argument('--verbose', action='store_true',
                        help='顯示每個 test case 的詳細結果')
    parser.add_argument('--output', default=None,
                        help='結果輸出的 JSON 路徑')
    parser.add_argument('--data-path', default='../',
                        help='dataset 根目錄（預設 ../）')
    args = parser.parse_args()

    if not args.evaluate:
        parser.print_help()
        sys.exit(1)

    test_ids = [args.test_id] if args.test_id is not None else None
    output   = args.output or (Path(args.evaluate).stem + '_contest_results.json')

    evaluator = ContestEvaluator(
        data_path=args.data_path,
        verbose=True,
    )
    evaluator.evaluate(
        optimizer_path=args.evaluate,
        test_ids=test_ids,
        output_path=output,
    )


if __name__ == '__main__':
    main()
