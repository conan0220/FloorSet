"""
Shared floorplan visualisation utilities.

Used by both solution/train.py and iccad2026contest/my_optimizer.py.
"""

import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    LAMBDA_WIRELENGTH, LAMBDA_AREA,
    LAMBDA_GROUPING, LAMBDA_MIB, LAMBDA_BOUNDARY, LAMBDA_OVERLAP,
)

# Block type → (facecolor, legend label)
BLOCK_TYPE_COLORS = {
    "mib":       ("mediumseagreen", "MIB"),
    "cluster":   ("tomato",         "Cluster"),
    "fixed":     ("violet",         "Fixed"),
    "preplaced": ("slategray",      "Preplaced"),
    "boundary":  ("goldenrod",      "Boundary"),
    "default":   ("lightsteelblue", "Default"),
}


def block_color(cons_row):
    """
    Return (facecolor, legend_label) for one block given its constraint row [5].
    constraints layout: [is_fixed_shape, is_preplaced, is_mib, is_cluster, boundary_code]
    """
    if cons_row[3] > 0:  return BLOCK_TYPE_COLORS["cluster"]
    if cons_row[0] > 0:  return BLOCK_TYPE_COLORS["fixed"]
    if cons_row[1] > 0:  return BLOCK_TYPE_COLORS["preplaced"]
    if cons_row[2] > 0:  return BLOCK_TYPE_COLORS["mib"]
    if cons_row[4] > 1:  return BLOCK_TYPE_COLORS["boundary"]
    return BLOCK_TYPE_COLORS["default"]


def save_floorplan_viz(
    gt_raw,           # [k, 4] tensor or list of (x, y, w, h) tuples
    pred_raw,         # [k, 4] tensor or list of (x, y, w, h) tuples
    block_count: int,
    out_path,         # Path or str — where to save the PNG
    title: str = "",
    loss_parts: dict = None,  # keys: total, coord, wirelength, area, grouping, mib, boundary, overlap
    constraints=None,         # [k, 5] tensor (optional)
    b2b_conn=None,            # [e, 3] tensor (block_i, block_j, weight) — optional
    p2b_conn=None,            # [e, 3] tensor (pin_idx, block_idx, weight) — optional
    pins_pos=None,            # [p, 2] tensor (x, y) — optional
):
    """
    Save a side-by-side GT vs Predicted floorplan figure.

    Blocks are coloured by type when constraints is provided.
    A legend and loss summary are added automatically.
    When b2b_conn / p2b_conn / pins_pos are provided, nets and pins are drawn.
    """
    def _iter(positions):
        """Yield (x, y, w, h) regardless of whether positions is a tensor or list."""
        for i in range(block_count):
            row = positions[i]
            if isinstance(row, torch.Tensor):
                yield tuple(row.tolist())
            else:
                yield tuple(row)

    def _draw_nets(ax, positions_list):
        """Draw b2b wires, p2b wires, and pin dots on ax."""
        # Precompute block centers
        centers = [(x + w / 2, y + h / 2) for x, y, w, h in positions_list]

        # ── b2b ──────────────────────────────────────────────────────
        if b2b_conn is not None:
            conn = b2b_conn
            if isinstance(conn, torch.Tensor):
                conn = conn.tolist()
            # normalise weights for alpha
            weights = [e[2] for e in conn if e[0] >= 0]
            max_w = max(weights) if weights else 1.0
            for bi, bj, w in conn:
                bi, bj = int(bi), int(bj)
                if bi < 0 or bj < 0 or bi >= block_count or bj >= block_count:
                    continue
                alpha = max(0.05, min(0.7, w / max(max_w, 1e-8)))
                cx_i, cy_i = centers[bi]
                cx_j, cy_j = centers[bj]
                ax.plot([cx_i, cx_j], [cy_i, cy_j],
                        color="royalblue", linewidth=0.5, alpha=alpha, zorder=1)

        # ── p2b ──────────────────────────────────────────────────────
        if p2b_conn is not None and pins_pos is not None:
            conn = p2b_conn
            pins = pins_pos
            if isinstance(conn, torch.Tensor):
                conn = conn.tolist()
            if isinstance(pins, torch.Tensor):
                pins = pins.tolist()
            n_pins = len(pins)
            weights = [e[2] for e in conn if e[0] >= 0]
            max_w = max(weights) if weights else 1.0
            for pi, bi, w in conn:
                pi, bi = int(pi), int(bi)
                if pi < 0 or bi < 0 or pi >= n_pins or bi >= block_count:
                    continue
                alpha = max(0.05, min(0.7, w / max(max_w, 1e-8)))
                px, py = pins[pi]
                cx, cy = centers[bi]
                ax.plot([px, cx], [py, cy],
                        color="darkorange", linewidth=0.5, alpha=alpha, zorder=1)

        # ── pins ─────────────────────────────────────────────────────
        if pins_pos is not None:
            pins = pins_pos
            if isinstance(pins, torch.Tensor):
                pins = pins.tolist()
            px = [p[0] for p in pins if p[0] >= 0]
            py = [p[1] for p in pins if p[1] >= 0]
            ax.scatter(px, py, s=8, c="darkorange", zorder=3, marker="x", linewidths=0.8)

    # Build per-block colors
    if constraints is not None:
        colors = [block_color(constraints[i]) for i in range(block_count)]
    else:
        cm = plt.cm.tab20(range(block_count))
        colors = [(cm[i % len(cm)], str(i)) for i in range(block_count)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, positions, subtitle in [
        (axes[0], gt_raw,   f"GT  ({block_count} blocks)"),
        (axes[1], pred_raw, f"Pred ({block_count} blocks)"),
    ]:
        ax.set_title(subtitle)
        for i, (x, y, w, h) in enumerate(_iter(positions)):
            facecolor, _ = colors[i]
            ax.add_patch(mpatches.Rectangle(
                (x, y), w, h,
                linewidth=0.8, edgecolor="black",
                facecolor=facecolor, alpha=0.7,
            ))
            # Label cluster blocks with their group ID
            if constraints is not None:
                group_id = int(constraints[i][3])
                if group_id > 0:
                    ax.text(
                        x + w / 2, y + h / 2, str(group_id),
                        ha="center", va="center",
                        fontsize=6, fontweight="bold", color="black",
                    )
        # Draw connectivity nets and pins (if provided)
        if b2b_conn is not None or p2b_conn is not None or pins_pos is not None:
            _draw_nets(ax, list(_iter(positions)))

        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

    # Legend — only types present in this sample
    if constraints is not None:
        seen = {lbl: col for col, lbl in colors}
        axes[1].legend(
            handles=[mpatches.Patch(facecolor=col, edgecolor="black", label=lbl)
                     for lbl, col in seen.items()],
            loc="upper right", fontsize=7, title="Block Type", title_fontsize=7,
        )

    if title:
        fig.suptitle(title)

    if loss_parts is not None:
        loss_text = (
            f"total={loss_parts['total']:.4f}  "
            f"wl={LAMBDA_WIRELENGTH * loss_parts['wirelength']:.4f}  "
            f"area={LAMBDA_AREA * loss_parts['area']:.4f}  "
            f"grp={LAMBDA_GROUPING * loss_parts['grouping']:.4f}  "
            f"mib={LAMBDA_MIB * loss_parts['mib']:.4f}  "
            f"bnd={LAMBDA_BOUNDARY * loss_parts['boundary']:.4f}  "
            f"ovlp={LAMBDA_OVERLAP * loss_parts['overlap']:.4f}"
        )
        fig.text(0.5, 0.01, loss_text, ha="center", va="bottom",
                 fontsize=8, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        fig.tight_layout(rect=[0, 0.04, 1, 1])
    else:
        fig.tight_layout()

    out_path = Path(out_path)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] saved {out_path.name}")
