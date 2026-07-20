from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "figures" / "adapter_flow_polished.pdf"

NAVY = "#123A7A"
BLUE = "#C7D7EB"
GREEN = "#D8EED7"
AMBER = "#F7D9A0"
INK = "#102B59"


def add_box(ax, x, y, width, height, label, color, fontsize=11):
    box = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.03,rounding_size=0.16",
        linewidth=1.25,
        edgecolor=NAVY,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color=INK,
        zorder=3,
    )


def arrow(ax, start, end, **kwargs):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=NAVY,
            shrinkA=0,
            shrinkB=0,
            zorder=1,
            **kwargs,
        )
    )


def line(ax, points):
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=NAVY, linewidth=1.25, zorder=1)


def main():
    fig, ax = plt.subplots(figsize=(5.4, 5.85))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 9.6)
    ax.axis("off")

    x_main = 4.2
    width_main = 6.4
    height = 0.68
    steps = [
        (9.1, "Selected hidden states", BLUE),
        (7.95, "1. Token-wise layer fusion", GREEN),
        (6.8, "2. Gated residual projection", AMBER),
        (5.65, "3. Token salience gate", GREEN),
        (4.5, "4. EncStack", AMBER),
        (3.35, r"Refined token memory $\tilde{R}$", BLUE),
    ]

    for y, label, color in steps:
        add_box(ax, x_main, y, width_main, height, label, color, 12)
    for (y1, _, _), (y2, _, _) in zip(steps, steps[1:]):
        arrow(ax, (x_main, y1 - height / 2), (x_main, y2 + height / 2))

    global_x, merge_y = 2.0, 1.8
    concat_x = 7.45
    concat_width, concat_height = 1.85, 0.78
    add_box(ax, global_x, merge_y, 3.5, 0.78, r"Global memory $G_m$", GREEN, 10)
    add_box(ax, concat_x, merge_y, concat_width, concat_height, "Concat", AMBER, 11)
    source_x = 7.0
    add_box(ax, source_x, 0.55, 5.8, 0.82, r"Source memory $M = [G_m;\,\tilde{R}]$", BLUE, 11)

    # Keep the two inputs visually distinct: R enters from above, G_m from the left.
    refined_bottom = steps[-1][0] - height / 2
    concat_top = merge_y + concat_height / 2
    line(ax, [(x_main, refined_bottom), (x_main, 2.72), (concat_x, 2.72)])
    arrow(ax, (concat_x, 2.72), (concat_x, concat_top))
    arrow(ax, (global_x + 1.625, merge_y), (concat_x - concat_width / 2, merge_y))
    arrow(ax, (concat_x, merge_y - concat_height / 2), (concat_x, 0.92))

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.04)


if __name__ == "__main__":
    main()
