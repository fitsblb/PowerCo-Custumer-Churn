import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# Create figure and axis with a nice style
plt.style.use("seaborn-v0_8")
fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis("off")

# Add a subtle background
ax.add_patch(
    patches.Rectangle((0, 0), 16, 10, facecolor="#f0f8ff", edgecolor="none", zorder=-1)
)

# Define phases and sub-stages
phase1 = {
    "title": "Phase 1: Data & Modeling",
    "stages": ["Load Data", "Feature Engineering", "Chronological Split", "Modeling"],
    "x": 2,
    "y": 5,
}
phase2 = {
    "title": "Phase 2: Evaluation & Insights",
    "stages": ["Evaluation", "Visualizations"],
    "x": 10,
    "y": 5,
}

# Colors
phase_colors = ["#4e79a7", "#f28e2c"]

# Draw Phase 1
bbox1 = FancyBboxPatch(
    (phase1["x"] - 1.5, phase1["y"] - 1.5),
    3,
    3,
    boxstyle="round,pad=0.1",
    linewidth=3,
    edgecolor="white",
    facecolor=phase_colors[0],
    alpha=0.9,
)
ax.add_patch(bbox1)
ax.text(
    phase1["x"],
    phase1["y"] + 1.2,
    phase1["title"],
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="white",
)

# Sub-stages for Phase 1
for i, stage in enumerate(phase1["stages"]):
    y_sub = phase1["y"] - 0.8 + i * 0.4
    ax.text(
        phase1["x"],
        y_sub,
        f"• {stage}",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

# Draw Phase 2
bbox2 = FancyBboxPatch(
    (phase2["x"] - 1.5, phase2["y"] - 1.5),
    3,
    3,
    boxstyle="round,pad=0.1",
    linewidth=3,
    edgecolor="white",
    facecolor=phase_colors[1],
    alpha=0.9,
)
ax.add_patch(bbox2)
ax.text(
    phase2["x"],
    phase2["y"] + 1.2,
    phase2["title"],
    ha="center",
    va="center",
    fontsize=14,
    fontweight="bold",
    color="white",
)

# Sub-stages for Phase 2
for i, stage in enumerate(phase2["stages"]):
    y_sub = phase2["y"] - 0.4 + i * 0.4
    ax.text(
        phase2["x"],
        y_sub,
        f"• {stage}",
        ha="center",
        va="center",
        fontsize=10,
        color="white",
    )

# Arrow from Phase 1 to Phase 2
arrow = FancyArrowPatch(
    (phase1["x"] + 1.5, phase1["y"]),
    (phase2["x"] - 1.5, phase2["y"]),
    arrowstyle="-|>",
    mutation_scale=25,
    color="#333333",
    linewidth=3,
)
ax.add_patch(arrow)

# Title
ax.text(
    8,
    9,
    "Project Roadmap: PowerCo Customer Churn Prediction",
    ha="center",
    va="center",
    fontsize=18,
    fontweight="bold",
    color="#2c3e50",
)

# Save
plt.savefig("Outputs/Project_Roadmap_Phase1_Phase2.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved project roadmap diagram to Outputs/Project_Roadmap_Phase1_Phase2.png")
