from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

# Clean academic palette
NAVY = "#172033"
BLUE = "#2F5D8C"
SOFT_BLUE = "#EAF2F8"
LIGHT_BLUE = "#D7E8F5"
GRAY = "#64748B"
LIGHT_GRAY = "#F3F4F6"
MID_GRAY = "#CBD5E1"
DARK_GRAY = "#334155"
GREEN = "#2E7D32"
RED = "#B91C1C"
AMBER = "#B45309"
WHITE = "#FFFFFF"

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 11


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05, dpi=300)
    plt.close(fig)


# -------------------------------------------------
# 1. Simple title background, optional
# Better: do not use this in Beamer title slide.
# -------------------------------------------------
def minimal_background(name):
    fig, ax = plt.subplots(figsize=(16, 9))
    clean(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(WHITE)

    # subtle top band
    ax.add_patch(Rectangle((0, 0.86), 1, 0.14, fc=NAVY, ec="none"))

    # subtle DNA-like line, very faint
    t = np.linspace(0.05, 0.95, 400)
    y1 = 0.42 + 0.04 * np.sin(2 * np.pi * 3 * t)
    y2 = 0.42 - 0.04 * np.sin(2 * np.pi * 3 * t)
    ax.plot(t, y1, color=LIGHT_BLUE, lw=2.2, alpha=0.75)
    ax.plot(t, y2, color=LIGHT_BLUE, lw=2.2, alpha=0.75)

    for i in np.linspace(0.08, 0.92, 20):
        yy1 = 0.42 + 0.04 * np.sin(2 * np.pi * 3 * i)
        yy2 = 0.42 - 0.04 * np.sin(2 * np.pi * 3 * i)
        ax.plot([i, i], [yy1, yy2], color=MID_GRAY, lw=0.8, alpha=0.6)

    save(fig, name)


minimal_background("title_background.pdf")
minimal_background("final_title_background.pdf")


# -------------------------------------------------
# 2. Impostor reads, clean academic version
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.set_facecolor(WHITE)
clean(ax)
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.6)

ax.text(
    0.5,
    5.05,
    "Opening question: which read appears artificially joined?",
    color=NAVY,
    fontsize=17,
    weight="bold",
)

# Read A
ax.text(0.75, 4.05, "Read A", color=NAVY, fontsize=13, weight="bold")
ax.add_patch(
    FancyBboxPatch(
        (2.0, 3.82),
        6.5,
        0.30,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        fc=BLUE,
        ec="none",
        alpha=0.95,
    )
)
ax.text(3.65, 3.35, "continuous alignment pattern", color=GRAY, fontsize=11)

# Read B
ax.text(0.75, 2.45, "Read B", color=NAVY, fontsize=13, weight="bold")
ax.add_patch(
    FancyBboxPatch(
        (2.0, 2.22),
        3.0,
        0.30,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        fc=BLUE,
        ec="none",
        alpha=0.95,
    )
)
ax.add_patch(
    FancyBboxPatch(
        (5.38, 2.22),
        3.12,
        0.30,
        boxstyle="round,pad=0.04,rounding_size=0.08",
        fc=LIGHT_BLUE,
        ec=BLUE,
        lw=1.2,
        alpha=0.95,
    )
)

ax.plot([5.18, 5.18], [2.02, 2.72], color=RED, lw=1.5, ls="--")
ax.text(4.68, 1.72, "possible junction", color=RED, fontsize=11)

ax.text(
    1.1,
    0.8,
    "The task of MitoChime is to automate this decision at read scale.",
    color=DARK_GRAY,
    fontsize=12,
)

save(fig, "impostor_reads.pdf")


# -------------------------------------------------
# 3. PCR chimera diagram
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.2))
ax.set_facecolor(WHITE)
clean(ax)
ax.set_xlim(0, 10)
ax.set_ylim(0, 6.2)

ax.text(0.5, 5.65, "PCR-induced chimera formation", fontsize=17, color=NAVY, weight="bold")

# Template 1
ax.add_patch(
    FancyBboxPatch((0.9, 4.65), 6.7, 0.28, boxstyle="round,pad=0.04", fc=BLUE, ec="none")
)
ax.text(7.85, 4.61, "Template 1", fontsize=11, color=GRAY)

# Template 2
ax.add_patch(
    FancyBboxPatch((0.9, 3.85), 6.7, 0.28, boxstyle="round,pad=0.04", fc=LIGHT_BLUE, ec=BLUE, lw=1)
)
ax.text(7.85, 3.81, "Template 2", fontsize=11, color=GRAY)

# Incomplete extension
ax.add_patch(
    FancyBboxPatch((0.9, 2.95), 3.0, 0.28, boxstyle="round,pad=0.04", fc=BLUE, ec="none")
)
ax.text(4.25, 2.92, "Incomplete extension", fontsize=11, color=GRAY)

# Arrow
ax.add_patch(
    FancyArrowPatch((4.7, 2.65), (4.7, 2.08), arrowstyle="-|>", mutation_scale=15, lw=1.6, color=GRAY)
)
ax.text(5.0, 2.32, "template switching", fontsize=11, color=RED)

# Final chimera
ax.add_patch(
    FancyBboxPatch((0.9, 1.25), 3.1, 0.34, boxstyle="round,pad=0.04", fc=BLUE, ec="none")
)
ax.add_patch(
    FancyBboxPatch((4.25, 1.25), 3.35, 0.34, boxstyle="round,pad=0.04", fc=LIGHT_BLUE, ec=BLUE, lw=1)
)
ax.plot([4.12, 4.12], [1.05, 1.8], color=RED, lw=1.5, ls="--")

ax.text(
    2.9,
    0.65,
    "Chimeric read: sequence from one region joined to another",
    fontsize=12,
    color=DARK_GRAY,
    ha="center",
)

save(fig, "pcr_chimera.pdf")


# -------------------------------------------------
# 4. Tool gap matrix
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6.3))
clean(ax)
ax.set_xlim(0, 12)
ax.set_ylim(0, 6.3)

ax.text(0.3, 5.85, "Existing tools and the study gap", fontsize=17, weight="bold", color=NAVY)

cols = ["Amplicon", "ASV/OTU", "Read-level", "Mito PCR\nchimera"]
rows = ["UCHIME", "DADA2", "CATCh", "ChimPipe", "MitoChime"]

x0, y0 = 2.4, 5.0
cw, rh = 2.1, 0.68

for j, c in enumerate(cols):
    ax.add_patch(Rectangle((x0 + j * cw, y0), cw, rh, fc=NAVY, ec=WHITE, lw=1))
    ax.text(
        x0 + j * cw + cw / 2,
        y0 + rh / 2,
        c,
        ha="center",
        va="center",
        color=WHITE,
        fontsize=10.5,
        weight="bold",
    )

values_dict = {
    "UCHIME": ["yes", "yes", "limited", "no"],
    "DADA2": ["yes", "yes", "no", "no"],
    "CATCh": ["yes", "yes", "no", "no"],
    "ChimPipe": ["no", "no", "yes", "different"],
    "MitoChime": ["no", "no", "yes", "yes"],
}

for i, r in enumerate(rows):
    y = y0 - (i + 1) * rh
    row_fc = SOFT_BLUE if r == "MitoChime" else LIGHT_GRAY
    ax.add_patch(Rectangle((0.3, y), 2.1, rh, fc=row_fc, ec=WHITE, lw=1))
    ax.text(
        1.35,
        y + rh / 2,
        r,
        ha="center",
        va="center",
        color=NAVY,
        fontsize=11,
        weight="bold" if r == "MitoChime" else "normal",
    )

    for j, v in enumerate(values_dict[r]):
        fc = "#F8FAFC" if r != "MitoChime" else "#EEF6FC"
        ax.add_patch(Rectangle((x0 + j * cw, y), cw, rh, fc=fc, ec=WHITE, lw=1))

        if v == "yes":
            text, color = "Yes", GREEN
        elif v == "no":
            text, color = "No", RED
        elif v == "limited":
            text, color = "Limited", AMBER
        else:
            text, color = "Different", AMBER

        ax.text(
            x0 + j * cw + cw / 2,
            y + rh / 2,
            text,
            ha="center",
            va="center",
            color=color,
            fontsize=10.5,
            weight="bold",
        )

ax.text(
    0.3,
    0.42,
    "Gap: direct read-level detection of PCR-induced mitochondrial chimeras",
    fontsize=12.5,
    color=BLUE,
    weight="bold",
)

save(fig, "tool_gap_matrix.pdf")


# -------------------------------------------------
# 5. Methodology workflow
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 6.4))
clean(ax)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6.4)

ax.text(0.4, 5.85, "MitoChime workflow", fontsize=17, weight="bold", color=NAVY)

steps = [
    "Reference\ngenome",
    "Simulate\nreads",
    "Align\nreads",
    "Extract\nfeatures",
    "Train\nmodels",
    "Filter\nreads",
    "Validate\nassembly",
]

xs = np.linspace(1.15, 12.85, len(steps))

for idx, label in enumerate(steps):
    x = xs[idx]
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.78, 3.05),
            1.56,
            0.95,
            boxstyle="round,pad=0.08,rounding_size=0.10",
            fc=SOFT_BLUE,
            ec=BLUE,
            lw=1.4,
        )
    )
    ax.text(x, 3.53, label, ha="center", va="center", fontsize=10.5, color=NAVY, weight="bold")

    if idx < len(steps) - 1:
        ax.add_patch(
            FancyArrowPatch(
                (x + 0.82, 3.53),
                (xs[idx + 1] - 0.82, 3.53),
                arrowstyle="-|>",
                mutation_scale=13,
                lw=1.4,
                color=GRAY,
            )
        )

ax.text(1.15, 2.35, "NCBI reference", ha="center", fontsize=10.5, color=GRAY)
ax.text(3.1, 2.35, "Clean + chimeric", ha="center", fontsize=10.5, color=GRAY)
ax.text(6.95, 2.35, "GB / CNN / BiGRU", ha="center", fontsize=10.5, color=GRAY)
ax.text(11.7, 2.35, "SPAdes + Bandage", ha="center", fontsize=10.5, color=GRAY)

save(fig, "methodology.png")


# -------------------------------------------------
# 6. Feature families
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.2))
clean(ax)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.2)

ax.text(0.3, 4.75, "Evidence extracted from each read", fontsize=17, weight="bold", color=NAVY)

ax.add_patch(
    FancyBboxPatch((0.55, 2.35), 1.45, 0.62, boxstyle="round,pad=0.08", fc=NAVY, ec="none")
)
ax.text(1.275, 2.66, "Read", ha="center", va="center", color=WHITE, fontsize=12, weight="bold")

labels = ["SA structure", "Soft clipping", "K-mer shift", "Microhomology", "MAPQ"]
for i, lab in enumerate(labels):
    x = 2.75 + i * 1.72
    ax.add_patch(
        FancyArrowPatch(
            (2.08, 2.66),
            (x - 0.63, 2.66),
            arrowstyle="-|>",
            mutation_scale=10,
            lw=1.0,
            color=MID_GRAY,
            alpha=0.7,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x - 0.61, 2.30),
            1.22,
            0.72,
            boxstyle="round,pad=0.08",
            fc=SOFT_BLUE,
            ec=BLUE,
            lw=1.2,
        )
    )
    ax.text(x, 2.66, lab, ha="center", va="center", color=NAVY, fontsize=10.2, weight="bold")

ax.add_patch(
    FancyBboxPatch(
        (4.15, 0.72),
        3.75,
        0.62,
        boxstyle="round,pad=0.10",
        fc=LIGHT_GRAY,
        ec=MID_GRAY,
        lw=1.2,
    )
)
ax.text(6.03, 1.03, "23 numeric predictors", ha="center", va="center", fontsize=12.5, color=NAVY, weight="bold")

for x in [4.5, 6.3, 8.1]:
    ax.add_patch(
        FancyArrowPatch((x, 2.18), (x, 1.45), arrowstyle="-|>", mutation_scale=11, lw=1.1, color=MID_GRAY)
    )

save(fig, "feature_families.pdf")


# -------------------------------------------------
# 7. Model architecture
# -------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5.3))
clean(ax)
ax.set_xlim(0, 12)
ax.set_ylim(0, 5.3)

ax.text(0.3, 4.85, "Three model representations", fontsize=17, weight="bold", color=NAVY)

cols = [
    ("Engineered features", "Gradient\nBoosting"),
    ("One-hot sequence", "1D CNN"),
    ("4-mer tokens", "Embedding\n+ BiGRU"),
]

for i, (inp, model) in enumerate(cols):
    x = 1.0 + i * 3.7

    ax.add_patch(
        FancyBboxPatch(
            (x, 3.15),
            2.55,
            0.62,
            boxstyle="round,pad=0.08",
            fc=LIGHT_GRAY,
            ec=MID_GRAY,
            lw=1.1,
        )
    )
    ax.text(x + 1.275, 3.46, inp, ha="center", va="center", fontsize=11.5, color=NAVY, weight="bold")

    ax.add_patch(
        FancyArrowPatch(
            (x + 1.275, 3.05),
            (x + 1.275, 2.45),
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.3,
            color=GRAY,
        )
    )

    ax.add_patch(
        FancyBboxPatch(
            (x, 1.42),
            2.55,
            0.76,
            boxstyle="round,pad=0.08",
            fc=SOFT_BLUE,
            ec=BLUE,
            lw=1.3,
        )
    )
    ax.text(x + 1.275, 1.80, model, ha="center", va="center", fontsize=12, color=NAVY, weight="bold")

    ax.add_patch(
        FancyArrowPatch(
            (x + 1.275, 1.30),
            (x + 1.275, 0.78),
            arrowstyle="-|>",
            mutation_scale=12,
            lw=1.2,
            color=GRAY,
        )
    )

    ax.text(x + 1.275, 0.50, "Clean or chimeric", ha="center", va="center", fontsize=10.5, color=GRAY)

save(fig, "model_architecture.pdf")


# -------------------------------------------------
# 8. Performance comparison
# -------------------------------------------------
models = ["GB", "CNN", "BiGRU"]
f1 = [0.7765, 0.9135, 0.9523]
auc = [0.8459, 0.9607, 0.9849]

x = np.arange(len(models))
width = 0.34

fig, ax = plt.subplots(figsize=(8.8, 5.0))
ax.bar(x - width / 2, f1, width, label="F1-score", color=BLUE)
ax.bar(x + width / 2, auc, width, label="ROC-AUC", color=LIGHT_BLUE, edgecolor=BLUE)

ax.set_ylim(0, 1.08)
ax.set_xticks(x, models)
ax.set_ylabel("Score")
ax.set_title("Held-out test performance", color=NAVY, weight="bold")
ax.legend(frameon=False, loc="lower right")

for i, v in enumerate(f1):
    ax.text(i - width / 2, v + 0.025, f"{v:.3f}", ha="center", fontsize=9, color=DARK_GRAY)
for i, v in enumerate(auc):
    ax.text(i + width / 2, v + 0.025, f"{v:.3f}", ha="center", fontsize=9, color=DARK_GRAY)

ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.18)

fig.tight_layout()
save(fig, "gb_cnn_rnn_compare.png")


# -------------------------------------------------
# 9. Confusion matrices
# -------------------------------------------------
cms = {
    "cm_gradient_boosting.png": ("Gradient Boosting", np.array([[3819, 181], [1352, 2648]])),
    "cnn_confusion.png": ("1D CNN", np.array([[3440, 560], [165, 3835]])),
    "rnn_confusion.png": ("BiGRU", np.array([[3751, 249], [138, 3862]])),
}

for name, (title, cm) in cms.items():
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks([0, 1], ["Clean", "Chimeric"])
    ax.set_yticks([0, 1], ["Clean", "Chimeric"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, color=NAVY, weight="bold")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color=WHITE if cm[i, j] > cm.max() / 2 else NAVY,
                fontsize=13,
                weight="bold",
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, name)


# -------------------------------------------------
# 10. Feature importance
# -------------------------------------------------
features = [
    "total clipped\nbases",
    "k-mer JS\ndivergence",
    "k-mer cosine\ndifference",
    "left soft\nclipping",
    "right soft\nclipping",
    "microhomology\nlength",
    "microhomology\nGC",
]
vals = [0.117, 0.074, 0.022, 0.017, 0.010, 0.002, 0.001]

fig, ax = plt.subplots(figsize=(8.3, 4.8))
y = np.arange(len(features))[::-1]

colors = [BLUE, BLUE, BLUE, LIGHT_BLUE, LIGHT_BLUE, MID_GRAY, MID_GRAY]
ax.barh(y, vals, color=colors, edgecolor="none")

ax.set_yticks(y, features)
ax.set_xlabel("Permutation importance")
ax.set_title("Gradient Boosting: strongest predictive evidence", color=NAVY, weight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", alpha=0.16)

for yi, v in zip(y, vals):
    ax.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9, color=DARK_GRAY)

fig.tight_layout()
save(fig, "gradient_imp_pair.png")


# -------------------------------------------------
# 11. Assembly placeholders
# Use actual Bandage exports if you have them.
# These are only clean placeholders.
# -------------------------------------------------
def assembly_placeholder(name, title, nodes):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    labels = ["Unfiltered", "GB", "CNN", "BiGRU"]

    for ax, label, n in zip(axes.ravel(), labels, nodes):
        ax.set_facecolor("#F8FAFC")
        clean(ax)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_title(label, fontsize=12, weight="bold", color=NAVY)

        rng = np.random.default_rng(abs(hash(label + name)) % 10000)

        if n <= 3:
            pts = []
            for k in range(n):
                angle = 2 * np.pi * k / max(n, 1)
                px = 0.36 * np.cos(angle)
                py = 0.36 * np.sin(angle)
                pts.append((px, py))
                ax.add_patch(Circle((px, py), 0.16, fc=SOFT_BLUE, ec=BLUE, lw=1.4))

            if n > 1:
                for a, b in zip(pts, pts[1:] + pts[:1]):
                    ax.plot([a[0], b[0]], [a[1], b[1]], color=GRAY, lw=1.5, alpha=0.85)

        else:
            pts = rng.normal(0, 0.35, (min(n, 45), 2))
            for p in pts:
                ax.add_patch(Circle((p[0], p[1]), 0.035, fc=BLUE, ec="none", alpha=0.75))
            for _ in range(60):
                a, b = rng.integers(0, len(pts), 2)
                ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], color=GRAY, lw=0.5, alpha=0.35)

        label_text = f"{n} node" if n == 1 else f"{n} nodes"
        ax.text(0, -0.84, label_text, ha="center", fontsize=11, color=GRAY)

    fig.suptitle(title, fontsize=15.5, weight="bold", color=NAVY)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, name)


assembly_placeholder(
    "assembly_20k_50.pdf",
    "Assembly graphs: 20K reads, 50% chimeras",
    [966, 3, 1, 2],
)

assembly_placeholder(
    "assembly_3200_50.pdf",
    "Assembly graphs: 3,200 reads, 50% chimeras",
    [2, 1, 2, 8],
)

print("Clean figures created in:", OUT)