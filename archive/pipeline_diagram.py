import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Create figure and axis with a nice style
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(14, 12))
ax.set_xlim(0, 12)
ax.set_ylim(0, 12)
ax.axis("off")

# Add a subtle background
ax.add_patch(
    patches.Rectangle((0, 0), 12, 12, facecolor="#f0f8ff", edgecolor="none", zorder=-1)
)

# Define stages (removed Visualizations, with multi-line for long names)
stages = [
    ("Load\nData", 1, 7),
    ("Feature\nEngineering", 3, 7),
    ("Chronological\nSplit", 5, 7),
    ("Modeling", 7, 7),
    ("Evaluation", 9, 7),
]

# Draw rounded boxes with gradient-like color (larger boxes)
colors = ["#4e79a7", "#f28e2c", "#e15759", "#76b7b2", "#59a14f"]
for i, (stage, x, y) in enumerate(stages):
    # Use FancyBboxPatch for rounded corners, larger height
    bbox = FancyBboxPatch(
        (x - 0.5, y - 0.5),
        1,
        1.0,
        boxstyle="round,pad=0.05",
        linewidth=2,
        edgecolor="white",
        facecolor=colors[i % len(colors)],
        alpha=0.9,
    )
    ax.add_patch(bbox)
    # Text inside the box, centered
    ax.text(
        x,
        y,
        stage,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color="white",
    )

# Draw arrows with better style
for i in range(len(stages) - 1):
    x1, y1 = stages[i][1], stages[i][2]
    x2, y2 = stages[i + 1][1], stages[i + 1][2]
    arrow = FancyArrowPatch(
        (x1 + 0.5, y1),
        (x2 - 0.5, y2),
        arrowstyle="-|>",
        mutation_scale=20,
        color="#333333",
        linewidth=2,
    )
    ax.add_patch(arrow)

# Title with better font
ax.text(
    6,
    11,
    "End-to-End Pipeline: PowerCo Customer Churn Prediction",
    ha="center",
    va="center",
    fontsize=16,
    fontweight="bold",
    color="#2c3e50",
)

# Save with higher DPI
plt.savefig("Outputs/End_to_End_Pipeline_Diagram.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved updated pipeline diagram to Outputs/End_to_End_Pipeline_Diagram.png")
