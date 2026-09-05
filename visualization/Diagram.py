# script to generate system diagram for methodology section of paper courtesy of Anthropic Claude

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

fig, ax = plt.subplots(figsize=(12, 13))
ax.set_xlim(0, 12.5)
ax.set_ylim(0, 13)
ax.axis("off")

BOX_STYLE = "round,pad=0.3,rounding_size=0.15"


def draw_box(
    x,
    y,
    w,
    h,
    text,
    facecolor="#E8EEF7",
    edgecolor="#2C3E50",
    fontsize=9.5,
    fontweight="normal",
    textcolor="#1a1a1a",
):
    box = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle=BOX_STYLE,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=1.4,
    )
    ax.add_patch(box)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=fontweight,
        color=textcolor,
        wrap=True,
    )
    return (x, y, w, h)


def draw_arrow(
    start, end, color="#2C3E50", style="-|>", connectionstyle="arc3,rad=0.0"
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=14,
        color=color,
        linewidth=1.3,
        connectionstyle=connectionstyle,
    )
    ax.add_patch(arrow)


gen_box = draw_box(
    6.0,
    12.3,
    6.4,
    0.8,
    "Chatette Template Generation",
    facecolor="#FCE9D8",
    fontweight="bold",
    fontsize=11,
)

closed_box = draw_box(
    3.0,
    10.9,
    4.6,
    1.1,
    "Closed-Vocabulary Test Set\n(44,000 sentences)",
    facecolor="#D9EAD3",
    fontweight="bold",
)
open_box = draw_box(
    9.0,
    10.9,
    4.6,
    1.1,
    "Open-Vocabulary Test Set\n(44,000 sentences,\ndisjoint train/test vocab)",
    facecolor="#D9EAD3",
    fontweight="bold",
)

draw_arrow((6.0, 11.9), (3.0, 11.45))
draw_arrow((6.0, 11.9), (9.0, 11.45))

closed_simple = draw_box(
    1.9, 9.5, 2.1, 0.75, "Simple\n(22,000)", facecolor="#F3F6EC", fontsize=9
)
closed_complex = draw_box(
    4.1, 9.5, 2.1, 0.75, "Complex\n(22,000)", facecolor="#F3F6EC", fontsize=9
)
open_simple = draw_box(
    7.9, 9.5, 2.1, 0.75, "Simple\n(22,000)", facecolor="#F3F6EC", fontsize=9
)
open_complex = draw_box(
    10.1, 9.5, 2.1, 0.75, "Complex\n(22,000)", facecolor="#F3F6EC", fontsize=9
)

draw_arrow((3.0, 10.35), (1.9, 9.875))
draw_arrow((3.0, 10.35), (4.1, 9.875))
draw_arrow((9.0, 10.35), (7.9, 9.875))
draw_arrow((9.0, 10.35), (10.1, 9.875))

sentence_box = draw_box(
    6.0, 8.0, 3.2, 0.7, "Test Sentence", facecolor="#FFF2CC", fontweight="bold"
)

for x in (1.9, 4.1, 7.9, 10.1):
    draw_arrow((x, 9.12), (6.0, 8.35))

gamebert_box = draw_box(
    2.1,
    6.4,
    2.9,
    1.3,
    "GAMEBERT\n\nFine-tuned BERT-Tiny;\nlearned entity\nclassification",
    facecolor="#CFE2F3",
    fontweight="bold",
    fontsize=9,
)
rake_box = draw_box(
    6.0,
    6.4,
    2.9,
    1.3,
    "RAKE\n\nCo-occurrence phrase\nranking + closed-\nvocabulary mapping",
    facecolor="#D0E0E3",
    fontweight="bold",
    fontsize=9,
)
gazetteer_box = draw_box(
    9.9,
    6.4,
    2.9,
    1.3,
    "Gazetteer\n\nDirect vocabulary\nlookup; no ranking\nstep",
    facecolor="#D0E0E3",
    fontweight="bold",
    fontsize=9,
)

draw_arrow((5.0, 7.65), (2.1, 7.05), connectionstyle="arc3,rad=-0.15")
draw_arrow((6.0, 7.65), (6.0, 7.05))
draw_arrow((7.0, 7.65), (9.9, 7.05), connectionstyle="arc3,rad=0.15")

pred_box = draw_box(
    6.0,
    4.6,
    8.4,
    0.85,
    "Predicted Labels: action, target, correction_connector",
    facecolor="#FFE5E5",
    fontweight="bold",
    fontsize=9.5,
)

draw_arrow((2.1, 5.75), (3.6, 5.0), connectionstyle="arc3,rad=-0.15")
draw_arrow((6.0, 5.75), (6.0, 5.025))
draw_arrow((9.9, 5.75), (8.4, 5.0), connectionstyle="arc3,rad=0.15")

gt_box = draw_box(
    10.1,
    3.1,
    2.8,
    1.3,
    "Ground Truth Labels\n(Chatette entity\nannotations;\nlast-occurrence-wins)",
    facecolor="#EAD1DC",
    fontsize=8.5,
)

eval_box = draw_box(
    5.2,
    3.1,
    5.6,
    1.0,
    "Evaluation Module",
    facecolor="#FCE5CD",
    fontweight="bold",
    fontsize=11,
)

draw_arrow((6.0, 4.175), (5.2, 3.6))
draw_arrow((10.1, 2.45), (8.0, 3.1), connectionstyle="arc3,rad=-0.1")

metrics_box = draw_box(
    5.2,
    1.4,
    9.0,
    1.3,
    "Metrics: Accuracy, Precision, Recall, F1, Latency\n"
    "Sliced by: Overall / Simple / Complex / Filler Present / Filler Absent /\n"
    "Closed Vocabulary / Open Vocabulary",
    facecolor="#D9D2E9",
    fontweight="bold",
    fontsize=9.5,
)

draw_arrow((5.2, 2.6), (5.2, 2.05))

ax.set_title(
    "Figure 3: GAMEBERT vs. RAKE vs. Gazetteer Evaluation Pipeline",
    fontsize=12,
    pad=20,
)

plt.tight_layout()
plt.savefig("figure3_system_diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved figure3_system_diagram.png")
