import matplotlib.pyplot as plt

metrics = [0.99, 0.9692, 0.97, 0.9697]
bar_colors = ["#FF6B6B", "#4D96FF", "#6BCB77", "#FFD93D"]
labels = ["Accuracy", "Precision", "Recall", "F1 Score"]

bars = plt.bar(labels, metrics, color=bar_colors)
plt.bar_label(bars, padding=3, fmt="%.4g")
plt.figtext(
    0.5,
    0.02,
    "",
    ha="center",
    fontsize=9,
    color="black",
)
plt.ylim(0.9, 1)

plt.savefig("metrics.png")
plt.close()
