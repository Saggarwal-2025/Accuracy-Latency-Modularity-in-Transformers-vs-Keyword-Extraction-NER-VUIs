import matplotlib.pyplot as plt
import numpy as np

ENTITIES = ["Action", "Target", "Correction"]
METRICS = ["Accuracy", "Precision", "Recall", "F1"]

RAKE_DATA = {
    "Table 2": {
        "Action": [0.899, 0.871, 0.899, 0.878],
        "Target": [0.875, 0.971, 0.875, 0.912],
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
    },
    "Table 3": {
        "Action": [0.884, 0.987, 0.884, 0.931],
        "Target": [0.927, 0.949, 0.927, 0.933],
        "Correction": [0.926, 0.999, 0.926, 0.947],
    },
}

GAZETTEER_DATA = {
    "Table 2": {
        "Action": [0.912, 0.902, 0.912, 0.901],
        "Target": [0.985, 0.979, 0.985, 0.981],
    },
    "Table 3": {
        "Action": [0.910, 0.899, 0.910, 0.898],
        "Target": [0.987, 0.980, 0.987, 0.982],
        "Correction": [0.999, 1.000, 0.999, 0.999],
    },
}


def get_metric(data: dict, table: str, entity: str, idx: int = 0) -> float:

    values = data.get(table, {}).get(entity)
    if not values:
        return np.nan
    return values[idx]


fig, axes = plt.subplots(1, 3, figsize=(12, 6))

bar_width = 0.25

for ax, entity in zip(axes, ENTITIES):
    # Correction has no simple-command data by construction, so that
    # subplot only shows "Complex" -- no empty "Simple" slot.
    conditions = ["Complex"] if entity == "Correction" else ["Simple", "Complex"]
    x = np.arange(len(conditions))

    if entity == "Correction":
        gamebert_values = [get_metric(GAMEBERT_DATA, "Table 3", entity)]
        rake_values = [get_metric(RAKE_DATA, "Table 3", entity)]
        gazetteer_values = [get_metric(GAZETTEER_DATA, "Table 3", entity)]
    else:
        gamebert_values = [
            get_metric(GAMEBERT_DATA, "Table 2", entity),
            get_metric(GAMEBERT_DATA, "Table 3", entity),
        ]
        rake_values = [
            get_metric(RAKE_DATA, "Table 2", entity),
            get_metric(RAKE_DATA, "Table 3", entity),
        ]
        gazetteer_values = [
            get_metric(GAZETTEER_DATA, "Table 2", entity),
            get_metric(GAZETTEER_DATA, "Table 3", entity),
        ]

    ax.bar(
        x - bar_width,
        gamebert_values,
        width=bar_width,
        label="GAMEBERT",
    )
    ax.bar(
        x,
        rake_values,
        width=bar_width,
        label="RAKE",
    )
    ax.bar(
        x + bar_width,
        gazetteer_values,
        width=bar_width,
        label="Gazetteer",
    )

    ax.set_title(entity, fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 1.05)

    # method courtesy of Google Gemini
    for xi, (g_val, r_val, z_val) in enumerate(
        zip(gamebert_values, rake_values, gazetteer_values)
    ):
        if not np.isnan(g_val):
            ax.text(
                xi - bar_width,
                g_val + 0.02,
                f"{g_val:.3f}",
                ha="center",
                fontsize=8,
            )

        if not np.isnan(r_val):
            ax.text(
                xi,
                r_val + 0.02,
                f"{r_val:.3f}",
                ha="center",
                fontsize=8,
            )

        if not np.isnan(z_val):
            ax.text(
                xi + bar_width,
                z_val + 0.02,
                f"{z_val:.3f}",
                ha="center",
                fontsize=8,
            )

axes[0].set_ylabel("Accuracy")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels)

fig.supxlabel("Figure 3: Accuracy for Simple and Complex Commands in Closed Vocabulary")

plt.tight_layout(rect=[0, 0.005, 1, 1])
plt.savefig("3_complexity.png")
plt.close()
