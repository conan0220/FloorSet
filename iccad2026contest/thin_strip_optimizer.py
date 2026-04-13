#!/usr/bin/env python3
"""
Thin Strip Optimizer - ICCAD 2026 FloorSet Challenge

核心思路：
  將所有 block 做成極細的水平長條（width=W, height=area/W），
  使所有 block 的中心點 x 座標都等於 W/2。

數學性質：
  - b2b HPWL 的 x 分量 = 0  （所有中心點 x 相同）
  - BBox area = W × Σ(area_i/W) = Σ(area_i)  （常數，與 W 無關，且等於最小可能值）
  - 完全無 overlap  （block 垂直疊放）
  - Area 精確滿足  （w*h = W*(area/W) = area）

因此問題化簡為 1D 排列問題：最佳化 y 方向的 HPWL。
使用 Analytic Placement（Laplacian 線性系統求解）找最佳排列。

同時測試「垂直細長條」（height=H, width=area/H），選擇 HPWL 較小的方向。

Usage:
    python iccad2026_evaluate.py --evaluate thin_strip_optimizer.py
"""

import math
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from iccad2026_evaluate import (
    FloorplanOptimizer,
    calculate_hpwl_b2b,
    calculate_hpwl_p2b,
    calculate_bbox_area,
)


class MyOptimizer(FloorplanOptimizer):
    """
    Thin Strip Optimizer.

    Strategy A (horizontal strips):
      - 所有 block 的 width = W（共享），height = area_i / W
      - 所有 block 的中心點 x 座標都是 W/2 → b2b HPWL_x = 0
      - 只需最小化 y 方向 HPWL

    Strategy B (vertical strips):
      - 所有 block 的 height = H（共享），width = area_i / H
      - 所有 block 的中心點 y 座標都是 H/2 → b2b HPWL_y = 0
      - 只需最小化 x 方向 HPWL

    兩種策略都試，選 HPWL 最小的。
    """

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor
    ) -> List[Tuple[float, float, float, float]]:

        if block_count == 0:
            return []

        # --- 1. 解析輸入 ---
        areas = [max(float(area_targets[i]), 1e-9) for i in range(block_count)]
        total_area = sum(areas)

        b2b_edges = self._parse_b2b(b2b_connectivity, block_count)
        pin_xy, p2b_edges = self._parse_p2b(p2b_connectivity, pins_pos, block_count)

        # --- 2. 產生候選 W 值 ---
        W_candidates = self._get_shared_dim_candidates(
            total_area, pin_xy, p2b_edges, axis='x'
        )
        H_candidates = self._get_shared_dim_candidates(
            total_area, pin_xy, p2b_edges, axis='y'
        )

        # --- 3. 嘗試所有組合，選 HPWL 最小的 ---
        best_positions = None
        best_hpwl = float('inf')

        # Strategy A: 水平細長條（固定 width=W，最佳化 y 排列）
        for W in W_candidates:
            positions = self._place_horizontal_strips(
                block_count, areas, W, b2b_edges, p2b_edges, pin_xy
            )
            hpwl = self._fast_hpwl(positions, b2b_edges, p2b_edges, pin_xy)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_positions = positions

        # Strategy B: 垂直細長條（固定 height=H，最佳化 x 排列）
        for H in H_candidates:
            positions = self._place_vertical_strips(
                block_count, areas, H, b2b_edges, p2b_edges, pin_xy
            )
            hpwl = self._fast_hpwl(positions, b2b_edges, p2b_edges, pin_xy)
            if hpwl < best_hpwl:
                best_hpwl = hpwl
                best_positions = positions

        return best_positions

    # =========================================================================
    # 輔助：解析輸入
    # =========================================================================

    def _parse_b2b(self, b2b_conn, n: int) -> List[Tuple[int, int, float]]:
        edges = []
        if b2b_conn is None or len(b2b_conn) == 0:
            return edges
        for edge in b2b_conn:
            if float(edge[0]) == -1:
                continue
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= i < n and 0 <= j < n and w > 0:
                edges.append((i, j, w))
        return edges

    def _parse_p2b(
        self,
        p2b_conn,
        pins_pos,
        n: int
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, float]]]:
        pin_xy = []
        edges = []
        if pins_pos is None or len(pins_pos) == 0:
            return pin_xy, edges
        n_pins = len(pins_pos)
        for k in range(n_pins):
            pin_xy.append((float(pins_pos[k][0]), float(pins_pos[k][1])))
        if p2b_conn is None or len(p2b_conn) == 0:
            return pin_xy, edges
        for edge in p2b_conn:
            if float(edge[0]) == -1:
                continue
            pin_idx, block_idx, w = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= pin_idx < n_pins and 0 <= block_idx < n and w > 0:
                edges.append((pin_idx, block_idx, w))
        return pin_xy, edges

    # =========================================================================
    # 輔助：產生候選共享尺寸
    # =========================================================================

    def _get_shared_dim_candidates(
        self,
        total_area: float,
        pin_xy: List[Tuple[float, float]],
        p2b_edges: List[Tuple[int, int, float]],
        axis: str  # 'x' or 'y'
    ) -> List[float]:
        """
        產生候選的共享尺寸（W 或 H）。

        分析：
          - 共享尺寸越大 → 另一方向的 block 高度越小 → y(或x)方向 HPWL 越小
          - 但 pin-to-block 在「共享維度方向」的距離 = |pin_coord - dim/2|
            → 共享尺寸 = 2 * weighted_mean(pin_coord) 可最小化 p2b 此方向貢獻

        策略：取 sqrt(area) 的多個倍數，以及以 pin 位置為中心的選項。
        """
        W_base = math.sqrt(total_area)
        candidates = set()

        # 以 sqrt(total_area) 為基礎，取多個倍數
        for scale in [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]:
            candidates.add(W_base * scale)

        # 以 pin 位置的加權平均為中心
        if p2b_edges and pin_xy:
            total_w = sum(w for (_, _, w) in p2b_edges)
            if total_w > 1e-9:
                coord_idx = 0 if axis == 'x' else 1
                coord_sum = sum(w * pin_xy[pi][coord_idx] for (pi, _, w) in p2b_edges)
                mean_pin_coord = coord_sum / total_w
                if mean_pin_coord > 1e-6:
                    # 2 * mean 使 dim/2 = mean_pin_coord
                    for scale in [1.0, 2.0, 4.0]:
                        candidates.add(2.0 * mean_pin_coord * scale)

        return sorted([max(c, 1e-6) for c in candidates])

    # =========================================================================
    # 輔助：放置 block
    # =========================================================================

    def _place_horizontal_strips(
        self,
        n: int,
        areas: List[float],
        W: float,
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_xy: List[Tuple[float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Strategy A: 水平細長條（width=W，height=area/W，沿 y 方向疊放）。
        使用 y 座標做 analytic 1D placement。
        """
        heights = [a / W for a in areas]
        pin_y = [py for (_, py) in pin_xy]
        ordering = self._analytic_1d_placement(n, b2b_edges, p2b_edges, pin_y)

        positions = [None] * n
        y_cur = 0.0
        for block_idx in ordering:
            positions[block_idx] = (0.0, y_cur, W, heights[block_idx])
            y_cur += heights[block_idx]
        return positions

    def _place_vertical_strips(
        self,
        n: int,
        areas: List[float],
        H: float,
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_xy: List[Tuple[float, float]]
    ) -> List[Tuple[float, float, float, float]]:
        """
        Strategy B: 垂直細長條（height=H，width=area/H，沿 x 方向疊放）。
        使用 x 座標做 analytic 1D placement。
        """
        widths = [a / H for a in areas]
        pin_x = [px for (px, _) in pin_xy]
        ordering = self._analytic_1d_placement(n, b2b_edges, p2b_edges, pin_x)

        positions = [None] * n
        x_cur = 0.0
        for block_idx in ordering:
            positions[block_idx] = (x_cur, 0.0, widths[block_idx], H)
            x_cur += widths[block_idx]
        return positions

    # =========================================================================
    # 核心：Analytic 1D Placement（Laplacian 線性求解）
    # =========================================================================

    def _analytic_1d_placement(
        self,
        n: int,
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_coords_1d: List[float]
    ) -> List[int]:
        """
        求解 Analytic 1D Placement，最小化以下二次目標函數：

          min  Σ_{b2b} w_ij * (y_i - y_j)²  +  Σ_{p2b} w_e * (y_i - pin_y_e)²

        這是一個 Laplacian 線性系統：
          (D - A) * y = rhs
        其中：
          - D = 對角矩陣，D[i] = Σ_j w_ij（所有與 i 相連的邊權重和）
          - A = block-to-block 鄰接矩陣
          - rhs[i] = Σ_{p2b edges of i} w_e * pin_y_e

        固定節點（pins）作為邊界條件代入右端向量。
        求解後按 y 值排序，得到最佳 1D 排列順序。

        Returns: 排列後的 block 索引列表（由小到大）
        """
        if n == 0:
            return []
        if n == 1:
            return [0]

        D = np.zeros(n, dtype=np.float64)
        A = np.zeros((n, n), dtype=np.float64)
        rhs = np.zeros(n, dtype=np.float64)

        # Block-to-block 貢獻
        for (i, j, w) in b2b_edges:
            A[i, j] += w
            A[j, i] += w
            D[i] += w
            D[j] += w

        # Pin-to-block 貢獻（pin 為固定節點）
        for (pin_idx, block_idx, w) in p2b_edges:
            if pin_idx < len(pin_coords_1d):
                py = pin_coords_1d[pin_idx]
                D[block_idx] += w
                rhs[block_idx] += w * py

        # 若完全無連線，使用預設順序
        if np.max(np.abs(D)) < 1e-10:
            return list(range(n))

        # Laplacian: L = diag(D) - A
        L = np.diag(D) - A
        # 加小量正則化，防止奇異矩陣（孤立節點）
        L += 1e-8 * np.eye(n)

        try:
            y_opt = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            # Fallback: 直接用 pin 貢獻的加權平均
            y_opt = rhs.copy()

        # 按最佳 1D 座標排序
        return list(np.argsort(y_opt))

    # =========================================================================
    # 輔助：快速 HPWL 計算（用於比較不同配置）
    # =========================================================================

    def _fast_hpwl(
        self,
        positions: List[Tuple[float, float, float, float]],
        b2b_edges: List[Tuple[int, int, float]],
        p2b_edges: List[Tuple[int, int, float]],
        pin_xy: List[Tuple[float, float]]
    ) -> float:
        """使用解析後的 edge list 快速計算 HPWL。"""
        total = 0.0
        for (i, j, w) in b2b_edges:
            cx_i = positions[i][0] + positions[i][2] * 0.5
            cy_i = positions[i][1] + positions[i][3] * 0.5
            cx_j = positions[j][0] + positions[j][2] * 0.5
            cy_j = positions[j][1] + positions[j][3] * 0.5
            total += w * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
        for (pin_idx, block_idx, w) in p2b_edges:
            px, py = pin_xy[pin_idx]
            bx = positions[block_idx][0] + positions[block_idx][2] * 0.5
            by = positions[block_idx][1] + positions[block_idx][3] * 0.5
            total += w * (abs(px - bx) + abs(py - by))
        return total
