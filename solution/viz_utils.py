"""
Shared floorplan visualisation utilities.

Used by both solution/train.py and iccad2026contest/my_optimizer.py.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import torch

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
    loss_parts: dict = None,  # keys: total, coord, wirelength, area, violation
    constraints=None,         # [k, 5] tensor (optional)
):
    """
    Save a side-by-side GT vs Predicted floorplan figure.

    Blocks are coloured by type when constraints is provided.
    A legend and loss summary are added automatically.
    """
    def _iter(positions):
        """Yield (x, y, w, h) regardless of whether positions is a tensor or list."""
        for i in range(block_count):
            row = positions[i]
            if isinstance(row, torch.Tensor):
                yield tuple(row.tolist())
            else:
                yield tuple(row)

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
            f"coord={loss_parts['coord']:.4f}  "
            f"wl={loss_parts['wirelength']:.4f}  "
            f"area={loss_parts['area']:.4f}  "
            f"viol={loss_parts['violation']:.4f}"
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
