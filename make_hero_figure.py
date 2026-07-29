"""
Journal-grade "hero" figure for RACE-MCBM.

Composition (single-figure panel, 4 blocks arranged left->right, top+bottom):
    (A) Problem  : STATS19 admin table mixing REPORTING artefacts vs. PHYSICAL
                   crash context, illustrated as two colour-coded feature stacks.
    (B) Signal   : label z-score + local-density (kNN in PCA space) gives a
                   continuous supervision signal, drawn as a 2-D PCA scatter
                   with a highlighted neighbourhood.
    (C) Extractor: iterative deflation PLS with (i) artefact soft-threshold
                   penalty and (ii) Gram-Schmidt orthogonalisation, shown as
                   three orthogonal concept directions in 3-D.
    (D) Outcome  : concept bottleneck X -> C -> y and the temporal-split
                   headline: RACE-MCBM matches LightGBM while cutting the
                   artefact score by ~4x.

Aesthetic: Nature-style, colour-blind-safe Okabe-Ito palette, soft shadows,
rounded panels, no gratuitous 3D.  All numbers taken from the README results
table so the figure is faithful to the reported experiments.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Palette (Okabe-Ito, colour-blind safe) + a few tinted neutrals.
# --------------------------------------------------------------------------- #
C = {
    "bg":        "#FBFBFD",
    "panel":     "#FFFFFF",
    "ink":       "#1A1F2B",
    "ink_soft":  "#4A5060",
    "grid":      "#E4E7EE",
    "blue":      "#0072B2",   # physical / concept
    "orange":    "#E69F00",   # artefact / reporting
    "green":     "#009E73",   # positive result
    "red":       "#D55E00",   # KSI / failure
    "purple":    "#CC79A7",
    "sky":       "#56B4E9",
    "yellow":    "#F0E442",
}

mpl.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           9,
    "axes.titlesize":      10.5,
    "axes.titleweight":    "bold",
    "axes.labelsize":      9,
    "axes.edgecolor":      C["ink_soft"],
    "axes.linewidth":      0.8,
    "xtick.color":         C["ink_soft"],
    "ytick.color":         C["ink_soft"],
    "legend.frameon":      False,
    "figure.dpi":          160,
    "savefig.dpi":         400,
    "pdf.fonttype":        42,
    "ps.fonttype":         42,
})

rng = np.random.default_rng(7)

# --------------------------------------------------------------------------- #
# Figure skeleton                                                             #
# --------------------------------------------------------------------------- #
fig = plt.figure(figsize=(14.0, 8.4))
fig.patch.set_facecolor(C["bg"])

# 2x2 grid with small gaps.  We'll place each panel manually to allow custom
# framing and inter-panel arrows.
gs = fig.add_gridspec(
    2, 2,
    left=0.035, right=0.985,
    top=0.905, bottom=0.055,
    wspace=0.20, hspace=0.32,
)

ax_A = fig.add_subplot(gs[0, 0])
ax_B = fig.add_subplot(gs[0, 1])
ax_C = fig.add_subplot(gs[1, 0])
ax_D = fig.add_subplot(gs[1, 1])

for ax in (ax_A, ax_B, ax_C, ax_D):
    ax.set_facecolor(C["panel"])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

# Rounded panel frames (drawn as background patches on the figure canvas).
def draw_panel_frame(ax, letter, title):
    bbox = ax.get_position()
    frame = FancyBboxPatch(
        (bbox.x0 - 0.006, bbox.y0 - 0.010),
        bbox.width + 0.012, bbox.height + 0.028,
        boxstyle="round,pad=0.0,rounding_size=0.012",
        linewidth=0.9, edgecolor="#D6DAE3",
        facecolor=C["panel"], zorder=0,
    )
    frame.set_transform(fig.transFigure)
    fig.patches.append(frame)
    # Letter badge
    fig.text(bbox.x0 + 0.006, bbox.y1 + 0.014, letter,
             fontsize=15, fontweight="bold", color=C["ink"],
             ha="left", va="bottom")
    fig.text(bbox.x0 + 0.028, bbox.y1 + 0.017, title,
             fontsize=11, fontweight="bold", color=C["ink"],
             ha="left", va="bottom")

# Master title & subtitle
fig.text(0.5, 0.965,
         "RACE-MCBM: Reporting-Aware Concept Bottleneck for Temporally Robust "
         "Crash Severity Prediction",
         ha="center", va="center",
         fontsize=14.5, fontweight="bold", color=C["ink"])
fig.text(0.5, 0.937,
         "Suppressing bureaucratic reporting shortcuts inside concept "
         "learning — not after it.",
         ha="center", va="center",
         fontsize=10.5, color=C["ink_soft"], style="italic")

# =========================================================================== #
# Panel A -- The problem: reporting artefacts entangled with physical signal  #
# =========================================================================== #
ax = ax_A
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# STATS19 "table" mock-up on the left
tbl_x, tbl_y, tbl_w, tbl_h = 0.4, 1.1, 4.6, 7.8
ax.add_patch(FancyBboxPatch((tbl_x, tbl_y), tbl_w, tbl_h,
                            boxstyle="round,pad=0.02,rounding_size=0.12",
                            linewidth=0.9, edgecolor="#C9CFDA",
                            facecolor="#F6F8FC", zorder=1))
ax.text(tbl_x + tbl_w/2, tbl_y + tbl_h + 0.28,
        "STATS19 casualty record",
        ha="center", va="bottom", fontsize=9.5,
        color=C["ink"], fontweight="bold")

# Column headers
ax.text(tbl_x + 0.5, tbl_y + tbl_h - 0.35, "feature",
        fontsize=8, color=C["ink_soft"], ha="left")
ax.text(tbl_x + 3.2, tbl_y + tbl_h - 0.35, "artefact $a_j$",
        fontsize=8, color=C["ink_soft"], ha="left")

rows = [
    ("police_attended",         0.92, "orange"),
    ("self_completion_form",    0.88, "orange"),
    ("age_band = Unknown",      0.75, "orange"),
    ("did_police_officer_attend", 0.95, "orange"),
    ("speed_limit_mph",         0.05, "blue"),
    ("light_conditions",        0.10, "blue"),
    ("vehicle_type = HGV",      0.08, "blue"),
    ("road_surface = wet",      0.06, "blue"),
    ("vulnerable_user",         0.04, "blue"),
]
row_h = 0.62
top_row_y = tbl_y + tbl_h - 1.05
for i, (name, a, tag) in enumerate(rows):
    y = top_row_y - i * row_h
    # Alternating row band
    if i % 2 == 0:
        ax.add_patch(Rectangle((tbl_x + 0.1, y - 0.24), tbl_w - 0.2, 0.48,
                               facecolor="#EEF2F8", edgecolor="none", zorder=1.2))
    ax.text(tbl_x + 0.35, y, name, fontsize=8.2, color=C["ink"],
            va="center", zorder=2)
    # artefact bar
    bar_x0 = tbl_x + 3.1; bar_w = 1.35
    ax.add_patch(Rectangle((bar_x0, y - 0.14), bar_w, 0.28,
                           facecolor="#E4E7EE", edgecolor="none", zorder=2))
    ax.add_patch(Rectangle((bar_x0, y - 0.14), bar_w * a, 0.28,
                           facecolor=C[tag], edgecolor="none", zorder=3))
    ax.text(bar_x0 + bar_w + 0.08, y, f"{a:.2f}",
            fontsize=7.6, color=C["ink_soft"], va="center", zorder=3)

# Right-hand "two worlds" schematic
world_x = 6.0
# Reporting world
ax.add_patch(FancyBboxPatch((world_x, 5.55), 3.7, 3.15,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=0.9, edgecolor=C["orange"],
                            facecolor="#FFF3DF", zorder=1))
ax.text(world_x + 1.85, 8.45, "REPORTING PROCESS",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#8A5A00")
ax.text(world_x + 1.85, 7.05,
        "how the crash entered\nthe database",
        ha="center", va="center", fontsize=8.5, color=C["ink"], linespacing=1.3)
ax.text(world_x + 1.85, 5.95,
        "officer attendance · unknown codes\nself-report form · missingness",
        ha="center", va="center", fontsize=7.6, color=C["ink_soft"],
        style="italic", linespacing=1.3)

# Physical world
ax.add_patch(FancyBboxPatch((world_x, 1.55), 3.7, 3.15,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=0.9, edgecolor=C["blue"],
                            facecolor="#E6F1FA", zorder=1))
ax.text(world_x + 1.85, 4.45, "PHYSICAL CONTEXT",
        ha="center", va="center", fontsize=9, fontweight="bold",
        color="#003F63")
ax.text(world_x + 1.85, 3.05,
        "why the crash was\nsevere",
        ha="center", va="center", fontsize=8.5, color=C["ink"], linespacing=1.3)
ax.text(world_x + 1.85, 1.95,
        "speed · vulnerable users · light\nvehicle mix · road surface",
        ha="center", va="center", fontsize=7.6, color=C["ink_soft"],
        style="italic", linespacing=1.3)

# Dashed "shortcut" arrow from artefact rows to reporting box
ax.annotate("", xy=(world_x - 0.05, 7.15),
            xytext=(tbl_x + tbl_w - 0.15, top_row_y - 0.9),
            arrowprops=dict(arrowstyle="-|>",
                            color=C["orange"], lw=1.4,
                            linestyle=(0, (4, 3)),
                            shrinkA=6, shrinkB=6))
ax.text(6.35, 7.85, "shortcut", fontsize=8, color=C["orange"],
        rotation=-8, fontweight="bold")

# Solid arrow from physical rows to physical box
ax.annotate("", xy=(world_x - 0.05, 3.15),
            xytext=(tbl_x + tbl_w - 0.15, top_row_y - 5.7),
            arrowprops=dict(arrowstyle="-|>",
                            color=C["blue"], lw=1.4,
                            shrinkA=6, shrinkB=6))
ax.text(6.35, 3.65, "signal we want", fontsize=8, color=C["blue"],
        rotation=6, fontweight="bold")

draw_panel_frame(ax_A, "A", "Two entangled signals in one administrative record")

# =========================================================================== #
# Panel B -- Signal: label + local-density (kNN in PCA space)                 #
# =========================================================================== #
ax = ax_B
ax.set_xlim(-4.4, 4.6); ax.set_ylim(-3.4, 4.0)

# Simulated PCA scatter with two intertwined "moons" so structure is clear.
n = 900
theta = rng.uniform(0, np.pi, n)
r = 2.6 + rng.normal(0, 0.28, n)
xs0 = r * np.cos(theta) + rng.normal(0, 0.32, n)
ys0 = r * np.sin(theta) - 0.7 + rng.normal(0, 0.32, n)

theta2 = rng.uniform(0, np.pi, n)
r2 = 2.6 + rng.normal(0, 0.28, n)
xs1 =  r2 * np.cos(theta2) + 1.3 + rng.normal(0, 0.32, n)
ys1 = -r2 * np.sin(theta2) + 1.0 + rng.normal(0, 0.32, n)

# Local KSI density -> colour intensity (fake but structurally sensible).
def local_density(xy, k=25):
    from scipy.spatial import cKDTree
    tree = cKDTree(xy)
    _, idx = tree.query(xy, k=k+1)
    return idx

X0 = np.column_stack([xs0, ys0])
X1 = np.column_stack([xs1, ys1])
Xall = np.vstack([X0, X1])
yall = np.concatenate([np.zeros(n), np.ones(n)])

try:
    from scipy.spatial import cKDTree
    tree = cKDTree(Xall)
    _, idx = tree.query(Xall, k=26)
    dens = yall[idx].mean(axis=1)
except Exception:
    # Fallback: distance-to-KSI centroid
    c = Xall[yall == 1].mean(axis=0)
    dens = 1.0 - np.linalg.norm(Xall - c, axis=1) / 6.0
    dens = np.clip(dens, 0.05, 0.95)

sc = ax.scatter(Xall[:, 0], Xall[:, 1], c=dens, s=8.5,
                cmap="RdYlBu_r", vmin=0.05, vmax=0.85,
                alpha=0.82, edgecolors="none", zorder=2)

# Highlighted query point + its k neighbours
q = np.array([1.3, 0.2])
tree = cKDTree(Xall)
_, nbr = tree.query(q, k=25)
ax.scatter(Xall[nbr, 0], Xall[nbr, 1],
           s=24, facecolor="none", edgecolor=C["ink"], linewidth=0.7, zorder=3)
ax.scatter([q[0]], [q[1]], s=140, marker="*",
           facecolor=C["yellow"], edgecolor=C["ink"], linewidth=0.9, zorder=5)

# Neighbourhood boundary
r_nbr = np.linalg.norm(Xall[nbr] - q, axis=1).max()
ax.add_patch(Circle(q, r_nbr,
                    facecolor="none", edgecolor=C["ink"],
                    linewidth=1.0, linestyle=(0, (3, 2)), zorder=4))
ax.text(q[0] + 0.15, q[1] + r_nbr + 0.18,
        "$k$-NN in PCA space",
        fontsize=8.4, color=C["ink"], fontweight="bold")

# Axes labels
ax.text(-4.1, 3.7, "PC$_2$", fontsize=9, color=C["ink_soft"],
        rotation=90, va="top")
ax.text(4.3, -3.15, "PC$_1$", fontsize=9, color=C["ink_soft"], ha="right")
ax.plot([-4.1, 4.3], [-3.15, -3.15], color=C["grid"], lw=0.7, zorder=1)
ax.plot([-4.1, -4.1], [-3.15, 3.6], color=C["grid"], lw=0.7, zorder=1)

# Signal equation ribbon
ribbon = FancyBboxPatch((-4.1, -3.1), 8.55, 0.85,
                        boxstyle="round,pad=0.02,rounding_size=0.08",
                        linewidth=0.7, edgecolor="#D6DAE3",
                        facecolor="#F6F8FC", zorder=1)
ax.add_patch(ribbon)
ax.text(0.15, -2.68,
        r"$s_i \;=\; z(y_i)\;+\;\beta\cdot z\!\left(\mathrm{logit}\;\hat{p}_{\mathrm{KSI}}^{(k\mathrm{-NN})}(x_i)\right)$",
        ha="center", va="center", fontsize=10.5, color=C["ink"])

# Colourbar
cax = fig.add_axes([ax.get_position().x1 - 0.055,
                    ax.get_position().y0 + 0.06,
                    0.012,
                    ax.get_position().height * 0.55])
cb = plt.colorbar(sc, cax=cax)
cb.set_label("local KSI rate", fontsize=7.8, color=C["ink_soft"])
cb.ax.tick_params(labelsize=7, colors=C["ink_soft"])
cb.outline.set_edgecolor("none")

# Legend chips
ax.text(-4.05, 3.55, "teacher-free geometric guidance",
        fontsize=8.5, color=C["ink"], fontweight="bold")
ax.text(-4.05, 3.15,
        "continuous, non-linear, no black-box teacher",
        fontsize=7.8, color=C["ink_soft"], style="italic")

draw_panel_frame(ax_B, "B", "Local-density supervision signal")

# =========================================================================== #
# Panel C -- Concept extractor: soft-threshold penalty + Gram-Schmidt         #
# =========================================================================== #
ax = ax_C
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# 1) Soft-threshold penalty curve
sub1 = fig.add_axes([ax.get_position().x0 + 0.028,
                     ax.get_position().y0 + 0.055,
                     ax.get_position().width * 0.30,
                     ax.get_position().height * 0.78])
w = np.linspace(-1.2, 1.2, 400)
eta = 0.35
raw = w
soft = np.sign(w) * np.maximum(0, np.abs(w) - eta)
sub1.plot(w, raw, color=C["ink_soft"], lw=1.1, linestyle=(0, (4, 3)),
          label="raw loading")
sub1.plot(w, soft, color=C["orange"], lw=2.0,
          label=r"after $\eta\cdot a_j$ penalty")
sub1.axhline(0, color=C["grid"], lw=0.6)
sub1.axvline(0, color=C["grid"], lw=0.6)
# Shaded dead-zone
sub1.axvspan(-eta, eta, color=C["orange"], alpha=0.12)
sub1.set_xlim(-1.15, 1.15); sub1.set_ylim(-1.15, 1.15)
sub1.set_xticks([-1, 0, 1]); sub1.set_yticks([-1, 0, 1])
sub1.tick_params(labelsize=7, colors=C["ink_soft"])
for s in ("top", "right"):
    sub1.spines[s].set_visible(False)
for s in ("bottom", "left"):
    sub1.spines[s].set_color(C["ink_soft"])
sub1.set_xlabel("raw loading $w_j$", fontsize=8, color=C["ink_soft"])
sub1.set_ylabel("penalised loading", fontsize=8, color=C["ink_soft"])
sub1.set_title("(i) Soft-threshold\nartefact penalty",
               fontsize=8.6, color=C["ink"], pad=4)
sub1.legend(loc="upper left", fontsize=6.8, handlelength=1.6,
            borderpad=0.3, labelspacing=0.35)

# 2) Gram-Schmidt orthogonal concepts (schematic 3-vector diagram)
sub2 = fig.add_axes([ax.get_position().x0 + ax.get_position().width * 0.36,
                     ax.get_position().y0 + 0.055,
                     ax.get_position().width * 0.33,
                     ax.get_position().height * 0.78])
sub2.set_xlim(-1.25, 1.25); sub2.set_ylim(-1.25, 1.25)
sub2.set_aspect("equal")
sub2.axhline(0, color=C["grid"], lw=0.6)
sub2.axvline(0, color=C["grid"], lw=0.6)
# Feature cloud
pts = rng.normal(0, 0.35, (200, 2))
sub2.scatter(pts[:, 0], pts[:, 1], s=6, color="#B7BEC9", alpha=0.6,
             edgecolors="none")
# Three orthogonal-ish concept axes
concept_dirs = np.array([[0.95, 0.15], [-0.20, 0.98], [0.65, -0.75]])
concept_dirs /= np.linalg.norm(concept_dirs, axis=1, keepdims=True)
colors3 = [C["blue"], C["green"], C["purple"]]
labels3 = ["$c_1$ vulnerable-user",
           "$c_2$ speed / road",
           "$c_3$ visibility"]
for v, col, lab in zip(concept_dirs, colors3, labels3):
    sub2.annotate("", xy=(v[0], v[1]), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0,
                                  mutation_scale=14))
    sub2.text(v[0] * 1.14, v[1] * 1.14, lab, fontsize=7.5,
              color=col, ha="center", va="center", fontweight="bold")
# Right-angle glyphs (visual hint of orthogonality)
sub2.text(0.05, 0.05, "$\\perp$", fontsize=13, color=C["ink_soft"])
sub2.set_xticks([]); sub2.set_yticks([])
for s in sub2.spines.values():
    s.set_color(C["ink_soft"])
sub2.set_title("(ii) Gram-Schmidt\northogonal concepts",
               fontsize=8.6, color=C["ink"], pad=4)

# 3) Iterative deflation loop schematic
sub3 = fig.add_axes([ax.get_position().x0 + ax.get_position().width * 0.72,
                     ax.get_position().y0 + 0.055,
                     ax.get_position().width * 0.26,
                     ax.get_position().height * 0.78])
sub3.set_xlim(0, 10); sub3.set_ylim(0, 10)
sub3.axis("off")

# Steps as pill boxes
steps = [
    (r"$w = X^{\!\top} s / n$", C["blue"]),
    (r"penalise by $a_j$",       C["orange"]),
    (r"orthogonalise",           C["purple"]),
    (r"deflate  $X, s$",         C["green"]),
]
step_h = 1.2; step_w = 8.4
for i, (txt, col) in enumerate(steps):
    y0 = 8.5 - i * 2.0
    sub3.add_patch(FancyBboxPatch((0.8, y0 - step_h/2), step_w, step_h,
                                  boxstyle="round,pad=0.02,rounding_size=0.20",
                                  facecolor="white", edgecolor=col, lw=1.2))
    sub3.text(0.8 + step_w/2, y0, txt, ha="center", va="center",
              fontsize=8.4, color=C["ink"])
    if i < len(steps) - 1:
        sub3.annotate("", xy=(5, y0 - step_h/2 - 0.05),
                      xytext=(5, y0 - step_h/2 - 0.70),
                      arrowprops=dict(arrowstyle="<|-", color=C["ink_soft"],
                                      lw=1.1, mutation_scale=10))
# Loop-back arrow
sub3.annotate("", xy=(0.35, 8.5), xytext=(0.35, 0.7),
              arrowprops=dict(arrowstyle="-|>", color=C["ink_soft"], lw=1.1,
                              mutation_scale=11,
                              connectionstyle="arc3,rad=-0.55"))
sub3.text(0.15, 4.6, "$k = 1 \\dots K$", fontsize=7.8, color=C["ink_soft"],
          rotation=90, ha="center", va="center")
sub3.set_title("(iii) Deflation PLS loop",
               fontsize=8.6, color=C["ink"], pad=4)

draw_panel_frame(ax_C, "C", "Artefact-regularised concept extractor")

# =========================================================================== #
# Panel D -- Bottleneck + temporal-split headline result                      #
# =========================================================================== #
ax = ax_D
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# --- Bottleneck schematic (top half of panel) ------------------------------
# X block
def block(ax, x, y, w, h, label, sublabel, colour, fill="#F6F8FC"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.10",
                                linewidth=1.1, edgecolor=colour,
                                facecolor=fill))
    ax.text(x + w/2, y + h - 0.25, label, ha="center", va="top",
            fontsize=9.5, color=colour, fontweight="bold")
    ax.text(x + w/2, y + 0.28, sublabel, ha="center", va="bottom",
            fontsize=7.6, color=C["ink_soft"], style="italic")

# X
block(ax, 0.35, 6.9, 2.0, 2.3, "$X$", "293 features", C["ink_soft"],
      fill="#EDEFF4")
# concept boxes (K=8 small circles inside a bigger frame)
cx0, cy0, cw, ch = 3.5, 6.9, 3.2, 2.3
ax.add_patch(FancyBboxPatch((cx0, cy0), cw, ch,
                            boxstyle="round,pad=0.02,rounding_size=0.10",
                            linewidth=1.1, edgecolor=C["blue"],
                            facecolor="#EAF3FB"))
ax.text(cx0 + cw/2, cy0 + ch - 0.25, "$C$   (8 concepts)",
        ha="center", va="top", fontsize=9.5, color=C["blue"],
        fontweight="bold")
concept_names = ["vulnerable", "speed", "visibility",
                 "urban mix", "vehicle", "surface", "time", "manoeuvre"]
concept_cols = [C["blue"], C["green"], C["purple"], C["sky"],
                C["orange"], C["red"], "#8065A6", "#B36F2E"]
for i, (nm, col) in enumerate(zip(concept_names, concept_cols)):
    r, c = i // 4, i % 4
    ccx = cx0 + 0.55 + c * 0.72
    ccy = cy0 + 1.30 - r * 0.60
    ax.add_patch(Circle((ccx, ccy), 0.20, facecolor=col,
                        edgecolor="white", linewidth=1.1, zorder=3))
    ax.text(ccx, ccy - 0.42, nm, ha="center", va="top",
            fontsize=6.6, color=C["ink_soft"])

# y block
block(ax, 8.0, 6.9, 1.65, 2.3, "$\\hat y$",
      "P(KSI)", C["red"], fill="#FDECE1")

# Arrows X -> C -> y
ax.annotate("", xy=(3.45, 8.05), xytext=(2.4, 8.05),
            arrowprops=dict(arrowstyle="-|>", color=C["ink_soft"],
                            lw=1.6, mutation_scale=14))
ax.text(2.92, 8.35, "concept\nextractor",
        ha="center", va="center", fontsize=7.6, color=C["ink_soft"],
        style="italic", linespacing=1.2)

ax.annotate("", xy=(7.95, 8.05), xytext=(6.75, 8.05),
            arrowprops=dict(arrowstyle="-|>", color=C["ink_soft"],
                            lw=1.6, mutation_scale=14))
ax.text(7.35, 8.35, "logistic +\ncalibration",
        ha="center", va="center", fontsize=7.6, color=C["ink_soft"],
        style="italic", linespacing=1.2)

# --- Temporal-split bar chart (bottom half) --------------------------------
sub = fig.add_axes([ax.get_position().x0 + 0.030,
                    ax.get_position().y0 + 0.055,
                    ax.get_position().width * 0.94,
                    ax.get_position().height * 0.48])

methods = ["LR\n(raw)", "Plain\nCBM", "SHAP\ntop-48", "RACE-MCBM\n(reporting-aware)",
           "RACE-MCBM\n(local density)", "LightGBM\n(black-box)"]
f1     = [0.1418, 0.2777, 0.4700, 0.4695, 0.4740, 0.4748]
artif  = [0.157, 0.207, 0.200, 0.102, 0.053, np.nan]

x = np.arange(len(methods))
bar_w = 0.36

bars1 = sub.bar(x - bar_w/2, f1, width=bar_w,
                color=[C["red"], C["red"], C["orange"],
                       C["blue"], C["green"], C["ink_soft"]],
                edgecolor="white", linewidth=1.0, label="Temporal F1")
sub2r = sub.twinx()
mask = ~np.isnan(artif)
bars2 = sub2r.bar(x[mask] + bar_w/2, np.array(artif)[mask], width=bar_w,
                  color="#F0E0C4", edgecolor=C["orange"], linewidth=1.0,
                  label="Artefact score")

# Value labels
for xi, v in zip(x, f1):
    sub.text(xi - bar_w/2, v + 0.012, f"{v:.3f}",
             ha="center", va="bottom", fontsize=7.4, color=C["ink"])
for xi, v in zip(x[mask], np.array(artif)[mask]):
    sub2r.text(xi + bar_w/2, v + 0.006, f"{v:.3f}",
               ha="center", va="bottom", fontsize=7.0, color="#8A5A00")

sub.set_ylim(0, 0.60); sub2r.set_ylim(0, 0.32)
sub.set_ylabel("Temporal F1", fontsize=8.6, color=C["ink"])
sub2r.set_ylabel("Artefact score (↓ better)",
                 fontsize=8.6, color=C["orange"])
sub.tick_params(labelsize=7.6, colors=C["ink_soft"])
sub2r.tick_params(labelsize=7.6, colors=C["orange"])
sub.set_xticks(x)
sub.set_xticklabels(methods, fontsize=7.5, color=C["ink"])
for s in ("top",):
    sub.spines[s].set_visible(False)
    sub2r.spines[s].set_visible(False)
sub.spines["right"].set_visible(False)
sub2r.spines["left"].set_visible(False)
sub.spines["left"].set_color(C["ink_soft"])
sub2r.spines["right"].set_color(C["orange"])
sub.grid(True, axis="y", color=C["grid"], lw=0.6, zorder=0)
sub.set_axisbelow(True)

# Highlight the winning bar
sub.annotate("matches LightGBM,\n4× less artefact",
             xy=(4 - bar_w/2, 0.474), xytext=(3.35, 0.575),
             fontsize=8, color=C["green"], fontweight="bold",
             ha="center", linespacing=1.2,
             arrowprops=dict(arrowstyle="-|>", color=C["green"], lw=1.2,
                             mutation_scale=11))

# Custom legend
handles = [
    mpatches.Patch(facecolor=C["green"], edgecolor="white", label="RACE-MCBM"),
    mpatches.Patch(facecolor=C["blue"], edgecolor="white", label="reporting-aware"),
    mpatches.Patch(facecolor=C["red"], edgecolor="white", label="unconstrained baselines"),
    mpatches.Patch(facecolor="#F0E0C4", edgecolor=C["orange"], label="artefact score"),
]
sub.legend(handles=handles, loc="upper left", fontsize=7.2,
           ncol=2, columnspacing=0.8, handlelength=1.2)

sub.set_title("Temporal split (train 2020-22 → test 2024)",
              fontsize=9, color=C["ink"], loc="left", pad=2)

draw_panel_frame(ax_D, "D", "Bottleneck & temporal-split headline")

# =========================================================================== #
# Inter-panel flow arrows (A -> B -> C -> D)                                  #
# =========================================================================== #
def flow_arrow(x0, y0, x1, y1, label=None, side="above"):
    arr = FancyArrowPatch((x0, y0), (x1, y1),
                          transform=fig.transFigure,
                          arrowstyle="-|>", mutation_scale=16,
                          color="#8A93A3", lw=1.6,
                          connectionstyle="arc3,rad=0.0", zorder=5)
    fig.patches.append(arr)
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.017 if side == "above" else -0.020
        fig.text(mx, my + dy, label, ha="center", va="center",
                 fontsize=8.5, color="#5A6070", fontweight="bold",
                 style="italic")

# A -> B (horizontal at top row)
flow_arrow(0.492, 0.72, 0.516, 0.72, "signal")
# B -> C (vertical between rows on right/left)
flow_arrow(0.75, 0.505, 0.75, 0.478, "concepts", side="below")
# C -> D (horizontal at bottom row)
flow_arrow(0.492, 0.28, 0.516, 0.28, "bottleneck")

# Save
out_png = OUT / "hero_overview.png"
out_pdf = OUT / "hero_overview.pdf"
fig.savefig(out_png, dpi=400, facecolor=C["bg"], bbox_inches=None)
fig.savefig(out_pdf, facecolor=C["bg"], bbox_inches=None)
print(f"Saved: {out_png}")
print(f"Saved: {out_pdf}")
