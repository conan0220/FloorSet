#!/usr/bin/env python3
"""
Ground Truth Score Computation
依照 contest_evaluate.py 的 spec-compliant 評分，計算 dataset Ground Truth 的分數。

做法：
  1. 從 polygon 頂點原始資料直接還原每個 block 的 (x, y, w, h) 位置。
  2. 用這些 GT 位置重新計算 HPWL（spec bbox 公式）與 BBox area，
     作為 baseline（不使用 dataset 內預存的 metrics[0] / metrics[-2]）。
  3. 將 GT 位置本身作為「優化器輸出」送入 evaluate_solution_spec()。
     - HPWL_gap = 0, Area_gap = 0（GT 自比自）
     - V_rel = GT 對各 soft constraint 的真實違反率
     - Cost = e^{β·V_rel}（≥1.0，越接近 1.0 越好）
  4. 結果存成 JSON，作為未來優化器的比較基準。

Usage:
    python compute_gt_scores.py
    python compute_gt_scores.py --output gt_scores.json
    python compute_gt_scores.py --data-path ../LiteTensorDataTest
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest
from contest_evaluate import (
    ALPHA, BETA, GAMMA, M_PENALTY, RUNTIME_FACTOR, AREA_TOLERANCE,
    evaluate_solution_spec,
    compute_total_score,
    compute_cost,
    calculate_hpwl_int,
    calculate_hpwl_ext,
    calculate_bbox_area,
    check_overlap,
    check_area_tolerance,
    compute_soft_violations,
    TestResult,
)


# =============================================================================
# 從 polygon 頂點還原 GT 位置（不使用預存 metrics）
# =============================================================================

def extract_gt_positions(
    polygons: torch.Tensor,
    block_count: int
) -> List[Tuple[float, float, float, float]]:
    """
    從 polygon 頂點原始資料還原 GT floorplan 的 [(x, y, w, h), ...] 列表。
    polygon 每個 block 有 5 個頂點（閉合矩形），padding 值為 -1。
    """
    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) >= 2:
            x_min = float(valid[:, 0].min())
            y_min = float(valid[:, 1].min())
            x_max = float(valid[:, 0].max())
            y_max = float(valid[:, 1].max())
            positions.append((x_min, y_min, x_max - x_min, y_max - y_min))
        else:
            positions.append((0.0, 0.0, 1.0, 1.0))
    return positions


def compute_fresh_baseline(
    gt_positions: List[Tuple[float, float, float, float]],
    b2b_conn: torch.Tensor,
    p2b_conn: torch.Tensor,
    pins_pos: torch.Tensor,
) -> Dict[str, float]:
    """
    用 GT 位置重新計算 HPWL 與 bbox（spec 公式），不使用預存 metrics。

    因為 baseline 和 solution 都是 GT，所以 HPWL_gap = 0、Area_gap = 0。
    這樣可確保 GT 的 Cost 完全由 soft constraint violations 決定。
    """
    hpwl_int  = calculate_hpwl_int(gt_positions, b2b_conn)
    hpwl_ext  = calculate_hpwl_ext(gt_positions, p2b_conn, pins_pos)
    hpwl_base = hpwl_int + hpwl_ext
    area_base = calculate_bbox_area(gt_positions)
    return {
        'hpwl_int':      hpwl_int,
        'hpwl_ext':      hpwl_ext,
        'hpwl_baseline': hpwl_base,
        'area_baseline': area_base,
    }


# =============================================================================
# 主計算邏輯
# =============================================================================

def compute_all_gt_scores(
    data_path: str = '../LiteTensorDataTest',
    verbose: bool = True,
) -> List[dict]:
    """
    計算所有 test cases 的 GT 分數。
    回傳每個 case 的完整 metrics dict。
    """
    print("Loading dataset...")
    dataset = FloorplanDatasetLiteTest(data_path)
    n_cases = len(dataset)
    print(f"Loaded {n_cases} cases")

    results = []
    iterator = tqdm(range(n_cases), desc="Evaluating GT") if verbose else range(n_cases)

    for idx in iterator:
        sample  = dataset[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        polygons, metrics_stored = labels

        block_count = int((area_target != -1).sum().item())

        # --- 1. 從 polygon 原始頂點還原 GT 位置 ---
        gt_pos = extract_gt_positions(polygons, block_count)

        # --- 2. 重新計算 baseline（不使用 metrics[0] / metrics[-2]）---
        baseline = compute_fresh_baseline(gt_pos, b2b_conn, p2b_conn, pins_pos)

        # --- 3. 以 GT 位置作為「優化器輸出」評分 ---
        #    HPWL_gap 和 Area_gap 都是 0（自比自）
        #    只有 soft violations 決定 Cost
        cm = evaluate_solution_spec(
            positions        = gt_pos,
            target_positions = gt_pos,   # GT 和 baseline 同一份
            baseline         = baseline,
            b2b_connectivity = b2b_conn,
            p2b_connectivity = p2b_conn,
            pins_pos         = pins_pos,
            area_targets     = area_target,
            constraints      = constraints,
        )

        # --- 4. 同時記錄預存 metrics 供對照 ---
        stored_hpwl    = float(metrics_stored[-2]) if metrics_stored is not None and len(metrics_stored) >= 2 else None
        stored_bbox    = float(metrics_stored[0])  if metrics_stored is not None and len(metrics_stored) >= 1 else None

        case_result = {
            'test_id':         idx,
            'block_count':     block_count,
            # Hard constraint check
            'is_feasible':     cm.is_feasible,
            'overlap_viol':    cm.overlap_violations,
            'area_viol':       cm.area_violations,
            # Spec-computed HPWL (from raw positions)
            'hpwl_int':        cm.hpwl_int,
            'hpwl_ext':        cm.hpwl_ext,
            'hpwl_total':      cm.hpwl_total,
            # Spec-computed bbox
            'bbox_area':       cm.bbox_area,
            # Gaps (should be 0 since we self-compare)
            'hpwl_gap':        cm.hpwl_gap,
            'area_gap':        cm.area_gap,
            # Soft violations
            'v_fixed':         cm.v_fixed,
            'v_preplaced':     cm.v_preplaced,
            'v_boundary':      cm.v_boundary,
            'v_grouping':      cm.v_grouping,
            'v_mib':           cm.v_mib,
            'total_violations': cm.total_soft_violations,
            'n_soft':          cm.n_soft,
            'v_rel':           cm.violations_relative,
            # Cost
            'cost':            cm.cost,
            # Pre-stored metrics for comparison
            'stored_hpwl_centroid': stored_hpwl,
            'stored_bbox_area':     stored_bbox,
        }
        results.append(case_result)

        if verbose:
            feasible = "OK" if cm.is_feasible else "INFEAS"
            tqdm.write(
                f"  [{idx:3d}] n={block_count:3d} | {feasible} | "
                f"HPWL={cm.hpwl_total:.1f}  BBox={cm.bbox_area:.1f} | "
                f"V_rel={cm.violations_relative:.3f} "
                f"(fix={cm.v_fixed} pre={cm.v_preplaced} "
                f"bnd={cm.v_boundary} grp={cm.v_grouping} mib={cm.v_mib} / N={cm.n_soft}) | "
                f"Cost={cm.cost:.4f}"
            )

    return results


# =============================================================================
# 彙整統計
# =============================================================================

def summarize(results: List[dict]) -> dict:
    """計算彙整統計。"""
    import math

    costs       = [r['cost']   for r in results]
    feasible    = [r for r in results if r['is_feasible']]
    block_cnts  = [r['block_count'] for r in results]

    # Exponential-weighted total score (spec formula)
    weights     = [math.exp(r['block_count']) for r in results]
    total_w     = sum(weights)
    total_score = sum(r['cost'] * w for r, w in zip(results, weights)) / total_w if total_w > 0 else 0.0

    # Violation breakdown (weighted avg)
    v_types = ['v_fixed', 'v_preplaced', 'v_boundary', 'v_grouping', 'v_mib']
    viol_sums = {k: sum(r[k] for r in results) for k in v_types}
    n_soft_total = sum(r['n_soft'] for r in results)

    return {
        'total_score_exp_weighted': total_score,
        'n_cases':      len(results),
        'n_feasible':   len(feasible),
        'avg_cost':     sum(costs) / len(costs) if costs else 0.0,
        'avg_hpwl':     sum(r['hpwl_total'] for r in results) / len(results),
        'avg_bbox':     sum(r['bbox_area']   for r in results) / len(results),
        'avg_v_rel':    sum(r['v_rel']       for r in results) / len(results),
        'total_violations': {k: viol_sums[k] for k in v_types},
        'n_soft_total': n_soft_total,
        'note': (
            'HPWL_gap = 0 and Area_gap = 0 by construction (GT vs GT). '
            'Cost = exp(beta * V_rel). Violations are real GT constraint violations.'
        ),
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Compute Ground Truth scores')
    parser.add_argument('--data-path', default='../LiteTensorDataTest')
    parser.add_argument('--output',    default='gt_scores.json')
    parser.add_argument('--quiet',     action='store_true')
    args = parser.parse_args()

    results = compute_all_gt_scores(
        data_path=args.data_path,
        verbose=not args.quiet,
    )

    summary = summarize(results)

    output = {
        'metadata': {
            'timestamp':   datetime.now().isoformat(),
            'description': 'Ground Truth scores computed from raw polygon data using spec-compliant formulas',
            'evaluator':   'contest_evaluate.py spec formulas (HPWL_int bbox, HPWL_ext centroid)',
            'baseline':    'Fresh-computed from GT positions (NOT using pre-stored metrics[0] or metrics[-2])',
            'alpha': ALPHA,
            'beta':  BETA,
            'runtime_factor': RUNTIME_FACTOR,
        },
        'summary': summary,
        'cases': results,
    }

    out_path = Path(args.output)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("GROUND TRUTH SCORES (spec-compliant, raw polygon data)")
    print("=" * 60)
    print(f"Total Score (exp-weighted): {summary['total_score_exp_weighted']:.4f}")
    print(f"Cases:    {summary['n_cases']}")
    print(f"Feasible: {summary['n_feasible']}")
    print(f"Avg Cost: {summary['avg_cost']:.4f}")
    print(f"Avg HPWL: {summary['avg_hpwl']:.2f}")
    print(f"Avg BBox: {summary['avg_bbox']:.2f}")
    print(f"Avg V_rel: {summary['avg_v_rel']:.4f}")
    print(f"\nViolation breakdown (total across all cases):")
    for k, v in summary['total_violations'].items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {out_path.resolve()}")
    print("=" * 60)
    print(
        "\n[Note] HPWL_gap = 0, Area_gap = 0 (GT vs GT). "
        "Cost >= 1.0 reflects soft constraint violations only."
    )


if __name__ == '__main__':
    main()
