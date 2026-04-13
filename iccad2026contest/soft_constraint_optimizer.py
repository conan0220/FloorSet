#!/usr/bin/env python3
"""
Soft Constraint Aware Optimizer - ICCAD 2026 FloorSet Challenge

在 Thin Strip 的基礎上，優先處理所有 soft constraints：

  ┌────────────────────────────────────────────┐ ← y_max
  │ top-left │  top-only blocks  │  top-right  │  ← Top Shelf  (觸頂)
  ├────────────────────────────────────────────┤
  │ ┌── group ──┐                              │
  │ │ blk  blk  │  block  block  block  block  │  ← Main Column (horizontal strips, width=W)
  │ └───────────┘  (grouping blocks 連續放置)  │
  ├────────────────────────────────────────────┤
  │ bot-left │ bottom-only blocks │ bot-right  │  ← Bottom Shelf (觸底)
  └────────────────────────────────────────────┘ ← y_min = 0

免費得到的 constraint 滿足：
  - MIB:    所有 block 同寬 W → h_i = area_i/W，同 group 自動相同 → V_mib = 0
  - L/R:    main column 所有 block 跨越 x=0 到 x=W → 全部觸左/右 boundary
  - Bottom: Bottom shelf 所有 block 的 y=0 → 全部觸底
  - Top:    Top shelf 所有 block 的 y+h=y_max → 全部觸頂
  - Corners: 放在 shelf 的對應端點即可（corner = edge1 + edge2）

主動處理的：
  - Fixed-shape: 使用 target (w, h)（由 contest_evaluate 透過 set_target_positions 傳入）
  - Preplaced:   使用 target (x, y, w, h)
  - Grouping:    Main column 內同 group 的 blocks 連續放置 → 自動共邊

Usage:
    python contest_evaluate.py --evaluate soft_constraint_optimizer.py
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from iccad2026_evaluate import FloorplanOptimizer


class MyOptimizer(FloorplanOptimizer):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)
        self._target_positions: Optional[List[Tuple[float, float, float, float]]] = None

    def set_target_positions(
        self, target_positions: List[Tuple[float, float, float, float]]
    ):
        """由 ContestEvaluator 在 solve() 前呼叫，提供 fixed/preplaced 的目標尺寸/位置。"""
        self._target_positions = target_positions

    # =========================================================================
    # 主入口
    # =========================================================================

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
    ) -> List[Tuple[float, float, float, float]]:

        if block_count == 0:
            return []

        # --- 1. 解析基本資料 ---
        areas = [max(float(area_targets[i]), 1e-9) for i in range(block_count)]

        b2b_edges = self._parse_b2b(b2b_connectivity, block_count)
        pin_xy, p2b_edges = self._parse_p2b(p2b_connectivity, pins_pos, block_count)

        # --- 2. 解析所有 soft constraints ---
        parsed = self._parse_constraints(block_count, areas, constraints)
        preplaced    = parsed['preplaced']    # {idx: (x,y,w,h)} - 目標位置
        fixed        = parsed['fixed']        # {idx: (w,h)}
        grp_groups   = parsed['grp_groups']   # {gid: [idx,...]}
        bnd_top      = parsed['bnd_top']      # {idx: btype}
        bnd_bottom   = parsed['bnd_bottom']   # {idx: btype}

        # 所有在 shelf 的 blocks（不進 main column）
        shelf_set = set(bnd_top.keys()) | set(bnd_bottom.keys())
        # NOTE: preplaced blocks 放入 main column 而非直接定位
        # 避免與 main column blocks 重疊導致 INFEASIBLE
        # 用 target y 作為 Laplacian anchor → 仍可接近目標位置
        main_set = set(range(block_count)) - shelf_set

        # Preplaced blocks 的 target center y 作為排列 bias
        anchor_y: Dict[int, float] = {}
        for i, (tx, ty, tw, th) in preplaced.items():
            if i in main_set:
                anchor_y[i] = ty + th * 0.5

        # --- 3. 計算 ordering（不依賴 W，只算一次）---
        main_list = sorted(main_set)
        pin_y = [py for (_, py) in pin_xy]
        ordering = self._order_main_column(
            main_list, b2b_edges, p2b_edges, pin_y, grp_groups, anchor_y
        )
        bot_order = self._sort_shelf(list(bnd_bottom.keys()), bnd_bottom,
                                     left_btype=9, right_btype=10)
        top_order = self._sort_shelf(list(bnd_top.keys()), bnd_top,
                                     left_btype=5, right_btype=6)

        # --- 4. 產生候選 W（含 fixed target widths，是滿足 fixed+MIB 的關鍵）---
        W_candidates = self._get_width_candidates(
            sum(areas), pin_xy, p2b_edges, fixed
        )

        # --- 5. 試所有候選 W，選 HPWL + α·bbox_area 最小的 ---
        # preplaced 傳空 dict（不再直接定位）
        empty_preplaced: Dict[int, Tuple[float,float,float,float]] = {}
        best_positions: Optional[List[Tuple[float,float,float,float]]] = None
        best_score = float('inf')

        for W in W_candidates:
            positions = self._assemble_positions(
                block_count, areas, W,
                fixed, empty_preplaced,
                bnd_top, bnd_bottom,
                ordering, bot_order, top_order,
            )
            score = self._estimate_score(positions, b2b_edges, p2b_edges, pin_xy)
            if score < best_score:
                best_score = score
                best_positions = positions

        return best_positions

    # =========================================================================
    # 組裝 positions（對一個固定 W）
    # =========================================================================

    def _assemble_positions(
        self,
        block_count: int,
        areas: List[float],
        W: float,
        fixed: Dict[int, Tuple[float,float]],
        preplaced: Dict[int, Tuple[float,float,float,float]],
        bnd_top: Dict[int, int],
        bnd_bottom: Dict[int, int],
        ordering: List[int],
        bot_order: List[int],
        top_order: List[int],
    ) -> List[Tuple[float,float,float,float]]:
        """用給定的 W 組裝完整 positions（無 overlap，含所有 constraint）。"""

        # 計算各 block 的尺寸
        widths, heights = self._compute_dims(block_count, areas, W, fixed, preplaced)

        # Shelf dims
        def shelf_dims(idxs):
            if not idxs:
                return 0.0, {}
            fixed_in          = {i: fixed[i] for i in idxs if i in fixed}
            non_fixed         = [i for i in idxs if i not in fixed]
            fixed_total_w     = sum(tw for (tw, _) in fixed_in.values())
            non_fixed_total_a = sum(areas[i] for i in non_fixed)
            avail_w           = W - fixed_total_w
            dims: Dict[int, Tuple[float, float]] = {}
            if non_fixed and avail_w > 1e-6:
                H_nf = non_fixed_total_a / avail_w
                for i in non_fixed:
                    dims[i] = (areas[i] / H_nf, H_nf)
            elif non_fixed:
                for i in non_fixed:
                    s = math.sqrt(max(areas[i], 1e-9))
                    dims[i] = (s, s)
            for i, (tw, th) in fixed_in.items():
                dims[i] = (tw, th)
            H_shelf = max(h for (_, h) in dims.values()) if dims else 0.0
            return H_shelf, dims

        H_bot, bot_dims = shelf_dims(list(bnd_bottom.keys()))
        H_top, top_dims = shelf_dims(list(bnd_top.keys()))

        positions: List[Optional[Tuple[float,float,float,float]]] = [None] * block_count

        # 7a. Preplaced
        for i, xywh in preplaced.items():
            positions[i] = xywh

        # 7b. Bottom shelf（y=0）
        x_cur = 0.0
        for i in bot_order:
            wi, hi = bot_dims[i]
            positions[i] = (x_cur, 0.0, wi, hi)
            x_cur += wi

        # 7c. Main column（width=W 對 non-fixed；fixed 用 target w）
        y_cur = H_bot
        for i in ordering:
            positions[i] = (0.0, y_cur, widths[i], heights[i])
            y_cur += heights[i]
        y_top_shelf_start = y_cur

        # 7d. Top shelf
        x_cur = 0.0
        for i in top_order:
            wi, hi = top_dims[i]
            positions[i] = (x_cur, y_top_shelf_start, wi, hi)
            x_cur += wi

        # Fallback
        for i in range(block_count):
            if positions[i] is None:
                positions[i] = (0.0, 0.0, widths[i], heights[i])

        return positions  # type: ignore[return-value]

    # =========================================================================
    # 估算成本：HPWL + α·bbox_area（越小越好）
    # =========================================================================

    def _estimate_score(
        self,
        positions: List[Tuple[float,float,float,float]],
        b2b_edges: List[Tuple[int,int,float]],
        p2b_edges: List[Tuple[int,int,float]],
        pin_xy: List[Tuple[float,float]],
    ) -> float:
        # HPWL（centroid-based，快速估算）
        hpwl = 0.0
        for (i, j, w) in b2b_edges:
            cx_i = positions[i][0] + positions[i][2] * 0.5
            cy_i = positions[i][1] + positions[i][3] * 0.5
            cx_j = positions[j][0] + positions[j][2] * 0.5
            cy_j = positions[j][1] + positions[j][3] * 0.5
            hpwl += w * (abs(cx_i - cx_j) + abs(cy_i - cy_j))
        for (pi, bi, w) in p2b_edges:
            px, py = pin_xy[pi]
            bx = positions[bi][0] + positions[bi][2] * 0.5
            by = positions[bi][1] + positions[bi][3] * 0.5
            hpwl += w * (abs(px - bx) + abs(py - by))

        # Bounding box area
        if positions:
            x_max = max(p[0] + p[2] for p in positions)
            x_min = min(p[0]        for p in positions)
            y_max = max(p[1] + p[3] for p in positions)
            y_min = min(p[1]        for p in positions)
            bbox  = (x_max - x_min) * (y_max - y_min)
        else:
            bbox = 0.0

        # Normalize by block count to balance the two terms
        n = max(len(positions), 1)
        return hpwl / n + 0.5 * bbox / n

    # =========================================================================
    # 解析 constraints
    # =========================================================================

    def _parse_constraints(
        self, n: int, areas: List[float], constraints: torch.Tensor
    ) -> Dict:
        """
        解析 constraints tensor，回傳各類型的 dict。

        Columns: [fixed, preplaced, MIB_group_id, cluster_group_id, boundary_btype]
        """
        preplaced:  Dict[int, Tuple[float,float,float,float]] = {}
        fixed:      Dict[int, Tuple[float,float]] = {}
        mib_groups: Dict[int, List[int]] = {}
        grp_groups: Dict[int, List[int]] = {}
        bnd_top:    Dict[int, int] = {}  # blocks needing top (btype & 4)
        bnd_bottom: Dict[int, int] = {}  # blocks needing bottom (btype & 8)
        # left/right (btype & 1 / & 2) 在 horizontal strip 中自動滿足，不需特殊處理

        tp = self._target_positions  # may be None

        for i in range(n):
            if i >= len(constraints):
                break
            c = constraints[i]

            is_fixed     = float(c[0]) != 0
            is_preplaced = float(c[1]) != 0
            mib_id       = int(float(c[2]))
            grp_id       = int(float(c[3]))
            btype        = int(float(c[4]))

            # fixed-shape: 使用 target (w, h)
            if is_fixed:
                if tp and i < len(tp):
                    _, _, tw, th = tp[i]
                    fixed[i] = (tw, th)
                else:
                    s = math.sqrt(areas[i])
                    fixed[i] = (s, s)  # fallback

            # preplaced: 使用 target (x, y, w, h)
            if is_preplaced:
                if tp and i < len(tp):
                    preplaced[i] = tp[i]
                # 若無 target_positions，接受違反（無法處理）

            if mib_id > 0:
                mib_groups.setdefault(mib_id, []).append(i)
            if grp_id > 0:
                grp_groups.setdefault(grp_id, []).append(i)

            # boundary：只拆出需要 top/bottom 的 blocks
            # left/right 在 horizontal strip 中所有 block 都自動觸及，無需排序
            if btype & 4:   # top
                bnd_top[i] = btype
            if btype & 8:   # bottom
                bnd_bottom[i] = btype

        return dict(
            preplaced=preplaced, fixed=fixed,
            mib_groups=mib_groups, grp_groups=grp_groups,
            bnd_top=bnd_top, bnd_bottom=bnd_bottom,
        )

    # =========================================================================
    # 計算 block 尺寸
    # =========================================================================

    def _compute_dims(
        self, n: int, areas: List[float], W: float,
        fixed: Dict[int, Tuple[float,float]],
        preplaced: Dict[int, Tuple[float,float,float,float]],
    ) -> Tuple[List[float], List[float]]:
        """
        widths[i], heights[i] 依優先順序：
          - preplaced: 使用 target w, h
          - fixed:     使用 target w, h（尺寸固定，位置自由）
          - MIB/other: width=W, height=area/W（自動滿足 MIB）
        """
        widths  = [W]     * n
        heights = [0.0]   * n
        for i in range(n):
            if i in preplaced:
                _, _, w, h = preplaced[i]
                widths[i]  = w
                heights[i] = h
            elif i in fixed:
                w, h = fixed[i]
                widths[i]  = w
                heights[i] = h
            else:
                widths[i]  = W
                heights[i] = areas[i] / W
        return widths, heights

    # =========================================================================
    # 產生候選 W
    # =========================================================================

    def _get_width_candidates(
        self,
        total_area: float,
        pin_xy: List[Tuple[float,float]],
        p2b_edges: List[Tuple[int,int,float]],
        fixed: Dict[int, Tuple[float,float]],
    ) -> List[float]:
        """
        產生多個候選共享寬度 W 供後續試驗。

        關鍵洞察：若 W = fixed block 的 target_w，則：
          - fixed block 的 (W, area/W) = (target_w, target_h) → V_fixed = 0
          - 同 MIB group 的 non-fixed blocks 也得到相同 (W, area/W) → V_mib = 0
          - BBox area ≈ Σ(area_targets)（最小可能值）
        """
        W_sqrt = math.sqrt(max(total_area, 1.0))
        candidates: set = {W_sqrt}

        # 標準倍數候選
        for scale in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]:
            candidates.add(W_sqrt * scale)

        # Pin-based 候選
        if p2b_edges and pin_xy:
            tw = sum(w for (_,_,w) in p2b_edges)
            if tw > 1e-9:
                wx = sum(w * pin_xy[pi][0] for (pi,_,w) in p2b_edges) / tw
                if wx > 1e-6:
                    candidates.add(2.0 * wx)
                    candidates.add(4.0 * wx)

        # Fixed target widths（最重要：同時滿足 V_fixed 和 V_mib）
        if fixed:
            for (tw, th) in fixed.values():
                if tw > 1e-6:
                    candidates.add(tw)

        return sorted(max(c, 1e-6) for c in candidates)

    # =========================================================================
    # Main column 的 1D 排列：Laplacian + Grouping super-node
    # =========================================================================

    def _order_main_column(
        self,
        main_list: List[int],
        b2b_edges: List[Tuple[int,int,float]],
        p2b_edges: List[Tuple[int,int,float]],
        pin_y: List[float],
        grp_groups: Dict[int, List[int]],
        anchor_y: Optional[Dict[int, float]] = None,
    ) -> List[int]:
        """
        用 Analytic 1D Placement 求 main column 的最佳排列順序，
        同時確保 grouping group 的 blocks 連續放置。

        演算法：
          1. 將每個 grouping group 的 members 視為「super-node」
          2. 在 super-node 層級解 Laplacian → 取得各 super-node 和 free block 的最佳 y
          3. 依 y 排序，展開 super-node → 保持 group members 連續
          4. Group 內部順序：依個別的 Laplacian 解排列
        """
        if not main_list:
            return []
        if len(main_list) == 1:
            return main_list

        main_set = set(main_list)

        # --- 建立 block → group 的映射（僅限 main column 內的 blocks）---
        block_to_gid: Dict[int, int] = {}
        for gid, members in grp_groups.items():
            for i in members:
                if i in main_set:
                    block_to_gid[i] = gid

        # --- 建立 super-node 問題 ---
        # 每個 free block 是一個獨立節點；每個 group 是一個 super-node
        # node_id: 0..len(free_blocks)-1 = free blocks
        #          len(free_blocks)..   = super-nodes (gid → node_id)
        free_blocks = [i for i in main_list if i not in block_to_gid]
        active_gids = sorted(set(block_to_gid.values()))

        free_idx  = {b: k for k, b in enumerate(free_blocks)}
        gid_idx   = {gid: len(free_blocks) + k for k, gid in enumerate(active_gids)}
        total_nodes = len(free_blocks) + len(active_gids)

        if total_nodes == 0:
            return main_list

        D   = np.zeros(total_nodes)
        A   = np.zeros((total_nodes, total_nodes))
        rhs = np.zeros(total_nodes)

        def node_of(block_idx):
            if block_idx in free_idx:
                return free_idx[block_idx]
            elif block_idx in block_to_gid:
                return gid_idx[block_to_gid[block_idx]]
            return None  # not in main column

        # b2b edges → 建 super-node Laplacian
        for (i, j, w) in b2b_edges:
            if i not in main_set or j not in main_set:
                continue
            ni = node_of(i)
            nj = node_of(j)
            if ni is None or nj is None or ni == nj:
                continue
            A[ni, nj] += w
            A[nj, ni] += w
            D[ni] += w
            D[nj] += w

        # p2b edges → 固定 pin y 作為邊界條件
        for (pi, bi, w) in p2b_edges:
            if bi not in main_set:
                continue
            ni = node_of(bi)
            if ni is None:
                continue
            if pi >= len(pin_y):
                continue
            py = pin_y[pi]
            D[ni]   += w
            rhs[ni] += w * py

        # anchor_y（preplaced 目標 y）→ 軟約束：偏置排列位置接近目標
        if anchor_y:
            # 計算 connectivity 的平均強度作為基準權重
            avg_conn = float(np.mean(D)) if np.max(np.abs(D)) > 1e-10 else 1.0
            anchor_w = max(avg_conn, 1.0)
            for block_idx, ay in anchor_y.items():
                ni = node_of(block_idx)
                if ni is not None:
                    D[ni]   += anchor_w
                    rhs[ni] += anchor_w * ay

        L = np.diag(D) - A + 1e-8 * np.eye(total_nodes)

        if np.max(np.abs(D)) < 1e-10:
            return main_list

        try:
            y_super = np.linalg.solve(L, rhs)
        except np.linalg.LinAlgError:
            y_super = rhs.copy()

        # --- 依 super-node y 排序 ---
        # free blocks 直接用 y_super[free_idx[b]]
        # group super-nodes 用 y_super[gid_idx[gid]]
        free_sorted  = sorted(free_blocks, key=lambda b: y_super[free_idx[b]])
        gid_y        = {gid: y_super[gid_idx[gid]] for gid in active_gids}

        # --- 展開：group members 按各自在獨立 Laplacian 中的 y 排序（internal order）---
        # 先對每個 group 的 members 做內部排列
        group_orderings: Dict[int, List[int]] = {}
        for gid in active_gids:
            members = [i for i in main_list if block_to_gid.get(i) == gid]
            if len(members) <= 1:
                group_orderings[gid] = members
                continue
            # 用 b2b + p2b 的 y-weighted-sum 作為 internal order heuristic
            member_y = {}
            total_w  = {}
            for i in members:
                member_y[i]  = 0.0
                total_w[i]   = 0.0
            for (bi, bj, w) in b2b_edges:
                for (src, dst) in [(bi, bj), (bj, bi)]:
                    if src in set(members) and dst not in set(members):
                        if node_of(dst) is not None:
                            # use y_super of the other node as an anchor
                            nd = node_of(dst)
                            member_y[src]  += w * y_super[nd]
                            total_w[src]   += w
            for (pi, bi, w) in p2b_edges:
                if bi in set(members) and pi < len(pin_y):
                    member_y[bi]  += w * pin_y[pi]
                    total_w[bi]   += w
            for i in members:
                if total_w[i] > 0:
                    member_y[i] /= total_w[i]
            group_orderings[gid] = sorted(members, key=lambda i: member_y.get(i, 0.0))

        # --- 合併：將 free blocks 和 groups 按 y 混合排列 ---
        # 每個 entry 是 (y_value, type, data)
        entries = []
        for b in free_sorted:
            entries.append((y_super[free_idx[b]], 'free', b))
        for gid in active_gids:
            entries.append((gid_y[gid], 'group', gid))
        entries.sort(key=lambda e: e[0])

        result = []
        for (_, etype, data) in entries:
            if etype == 'free':
                result.append(data)
            else:
                result.extend(group_orderings[data])

        return result

    # =========================================================================
    # Shelf 排列：corner blocks 放對應端點
    # =========================================================================

    def _sort_shelf(
        self,
        idxs: List[int],
        bnd: Dict[int, int],
        left_btype: int,
        right_btype: int,
    ) -> List[int]:
        """
        排列 shelf blocks：
          - left_btype corner → 最左
          - right_btype corner → 最右
          - 其他 → 中間（按 index 排列）
        """
        left  = [i for i in idxs if bnd.get(i, 0) == left_btype]
        right = [i for i in idxs if bnd.get(i, 0) == right_btype]
        mid   = [i for i in idxs if i not in left and i not in right]
        return left + mid + right

    # =========================================================================
    # 解析 connectivity
    # =========================================================================

    def _parse_b2b(self, b2b_conn, n: int) -> List[Tuple[int,int,float]]:
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
        self, p2b_conn, pins_pos, n: int
    ) -> Tuple[List[Tuple[float,float]], List[Tuple[int,int,float]]]:
        pin_xy: List[Tuple[float,float]] = []
        edges: List[Tuple[int,int,float]] = []
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
            pi, bi, w = int(edge[0]), int(edge[1]), float(edge[2])
            if 0 <= pi < n_pins and 0 <= bi < n and w > 0:
                edges.append((pi, bi, w))
        return pin_xy, edges
