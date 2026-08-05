import matplotlib.pyplot as plt
import numpy as np

ENTITIES = ["Action", "Target", "Correction"]
METRICS = ["Accuracy", "Precision", "Recall", "F1"]

RAKE_DATA = {
    "Table 2": {
        "Action": [0.899, 0.871, 0.899, 0.878],
        "Target": [0.875, 0.971, 0.875, 0.912],
        "Correction": [1.000, 1.000, 1.000, 1.000],
    },
    "Table 3": {
        "Action": [0.494, 0.476, 0.494, 0.475],
        "Target": [0.529, 0.680, 0.529, 0.586],
        "Correction": [0.300, 0.267, 0.300, 0.282],
    },
}

GAMEBERT_DATA = {
    "Table 2": {
        "Action": [0.858, 0.988, 0.858, 0.917],
        "Target": [0.904, 0.946, 0.904, 0.918],
        "Correction": [1.000, 1.000, 1.000, 1.000],
    },
    "Table 3": {
        "Action": [0.884, 0.987, 0.884, 0.931],
        "Target": [0.927, 0.949, 0.927, 0.933],
        "Correction": [0.926, 0.999, 0.926, 0.947],
    },
}

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

conditions = ["Simple", "Complex"]
x = np.arange(len(conditions))
bar_width = 0.35

for ax, entity in zip(axes, ENTITIES):
    gamebert_values = [
        GAMEBERT_DATA["Table 2"][entity][0],
        GAMEBERT_DATA["Table 3"][entity][0],
    ]
    rake_values = [
        RAKE_DATA["Table 2"][entity][0],
        RAKE_DATA["Table 3"][entity][0],
    ]

    ax.bar(
        x - bar_width / 2,
        gamebert_values,
        width=bar_width,
        label="GAMEBERT",
    )
    ax.bar(
        x + bar_width / 2,
        rake_values,
        width=bar_width,
        label="RAKE",
    )
    ax.set_title(entity, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)

    # method courtesy of Google Gemini
    for xi, (g_val, r_val) in enumerate(zip(gamebert_values, rake_values)):
        ax.text(
            xi - bar_width / 2,
            g_val + 0.02,
            f"{g_val:.3f}",
            ha="center",
            fontsize=8,
        )

        ax.text(
            xi + bar_width / 2,
            r_val + 0.02,
            f"{r_val:.3f}",
            ha="center",
            fontsize=8,
        )

axes[0].set_ylabel("Accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels)

fig.supxlabel("Figure 2: Accuracy for Simple and Complex Commands")

plt.tight_layout(rect=[0, 0.005, 1, 1])
plt.savefig("complexity.png")
plt.close()
