"""Plot composition and processing rankings."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd

from stage2.scoring import MODE_WEIGHTS
from .config import ELEMENT_COLS

COLORS = {
    "blue": "#4C78A8", "teal": "#2A9D8F", "orange": "#E76F51",
    "gold": "#F2B134", "purple": "#7A6FA6", "dark": "#27313A",
    "grid": "#E8ECEF",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 16, "axes.labelsize": 16, "axes.titlesize": 18,
        "axes.titleweight": "bold", "xtick.labelsize": 16, "ytick.labelsize": 16,
        "legend.fontsize": 14, "axes.linewidth": 1.1,
        "xtick.major.width": 1.1, "ytick.major.width": 1.1,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "mathtext.fontset": "dejavusans",
    })


def clean_axis(ax, grid_axis="y"):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=.8)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=False, right=False, length=6, pad=7)


def panel_label(ax, label):
    ax.text(-.10, 1.12, label, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="top", color=COLORS["dark"])


def short_id(value):
    return value.rsplit("-", 1)[-1]


def save_figure(fig, directory, name):
    fig.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def ranked_bars(ax, top, title, xlabel, ylabel, minimum_limit):
    selected = top.iloc[0].candidate_id
    rows = top.iloc[::-1]
    colors = [COLORS["orange"] if cid == selected else COLORS["blue"] for cid in rows.candidate_id]
    bars = ax.barh([short_id(cid) for cid in rows.candidate_id], rows.final_score,
                   color=colors, height=.62)
    maximum = max(minimum_limit, float(top.final_score.max()) * 1.10)
    ax.set(xlim=(0, maximum), xlabel=xlabel, ylabel=ylabel)
    ax.set_title(title, pad=22)
    for bar, score in zip(bars, rows.final_score):
        ax.text(score + maximum * .009, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=14)
    clean_axis(ax, "x")


def plot_s6(results, directory):
    ranked = pd.read_csv(results / "04_composition/composition_ranked.csv")
    top = pd.read_csv(results / "04_composition/composition_topk.csv").sort_values("rank")
    selected = top.iloc[0]
    background = ranked[~ranked.candidate_id.isin(top.candidate_id)]
    others = top.iloc[1:]
    fig, axes = plt.subplots(1, 2, figsize=(19.2, 7.8),
                             gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    ax.scatter(background.mean_UTS, background.mean_EL, s=13, color="#CBD1D6",
               alpha=.35, edgecolors="none", label="Candidates")
    ax.scatter(others.mean_UTS, others.mean_EL, s=48, color=COLORS["teal"],
               edgecolors="white", linewidth=.8, label=f"Top {len(top)}", zorder=3)
    ax.scatter(selected.mean_UTS, selected.mean_EL, marker="*", s=260,
               color=COLORS["gold"], edgecolors=COLORS["dark"], linewidth=1,
               zorder=5, label=f"Selected {short_id(selected.candidate_id)}")
    ax.annotate(f"{selected.candidate_id}\n{selected.mean_UTS:.0f} MPa, {selected.mean_EL:.1f}%",
                (selected.mean_UTS, selected.mean_EL), xytext=(-160, 38),
                textcoords="offset points", fontsize=14, color=COLORS["dark"],
                arrowprops={"arrowstyle": "-", "color": COLORS["dark"], "lw": .8},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 3}, zorder=8)
    ax.set(xlabel="Predicted UTS (MPa)", ylabel="Predicted EL (%)")
    ax.set_title("Stage I: composition screening", pad=22)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.19),
               ncol=3, columnspacing=1.2, handletextpad=.45)
    clean_axis(ax, "both")
    panel_label(ax, "a")
    ranked_bars(axes[1], top, f"Top {len(top)} composition candidates",
                "Composite composition score", "Candidate ID", 1.0)
    panel_label(axes[1], "b")
    fig.subplots_adjust(wspace=.34, left=.075, right=.985, bottom=.25, top=.85)
    save_figure(fig, directory, "Fig_S6")


def plot_s7(results, directory):
    ranked = pd.read_csv(results / "05_processing/process_ranked.csv")
    top = pd.read_csv(results / "05_processing/process_topk.csv").sort_values("rank")
    selected = ranked.set_index("candidate_id").loc[top.iloc[0].candidate_id]
    selected_id = top.iloc[0].candidate_id
    design = json.loads((results / "05_processing/design_settings.json").read_text())
    target_uts, target_el = design["targets"]["UTS"], design["targets"]["EL"]
    fig, axes = plt.subplots(1, 3, figsize=(22.8, 7.8),
                             gridspec_kw={"width_ratios": [1.23, 1, .92]})
    ax = axes[0]
    background = ranked[~ranked.candidate_id.isin(top.candidate_id)]
    others = ranked[ranked.candidate_id.isin(top.candidate_id.iloc[1:])]
    x_span = max(float(ranked.mean_UTS.max()) - min(float(ranked.mean_UTS.min()), target_uts),
                 abs(float(ranked.mean_UTS.min()) - target_uts), 1.0)
    lower_x = min(float(ranked.mean_UTS.min()), target_uts) - .06*x_span
    upper_x = max(float(ranked.mean_UTS.max()), target_uts) + .10*x_span
    lower_y = min(target_el - .5, float(ranked.mean_EL.min()) - .5)
    upper_y = max(target_el + 8.5, float(ranked.mean_EL.max()) + .5)
    ax.set(xlim=(lower_x, upper_x), ylim=(lower_y, upper_y))
    ax.add_patch(Rectangle((target_uts, target_el), upper_x-target_uts, upper_y-target_el,
                           facecolor="#DDEFEA", alpha=.65, zorder=0))
    ax.scatter(background.mean_UTS, background.mean_EL, s=22, color="#BBC3CA",
               alpha=.58, edgecolors="none", label="Other routes")
    ax.scatter(others.mean_UTS, others.mean_EL, s=46, color=COLORS["teal"],
               edgecolors="white", linewidth=.7, zorder=3, label=f"Top {len(top)} by score")
    ax.scatter(selected.mean_UTS, selected.mean_EL, marker="*", s=360,
               color="white", edgecolors="white", linewidth=2, zorder=4)
    ax.scatter(selected.mean_UTS, selected.mean_EL, marker="*", s=270,
               color=COLORS["gold"], edgecolors=COLORS["dark"], linewidth=1, zorder=5)
    ax.axvline(target_uts, color=COLORS["orange"], linestyle=(0, (4, 3)), linewidth=1)
    ax.axhline(target_el, color=COLORS["orange"], linestyle=(0, (4, 3)), linewidth=1)
    ax.annotate(f"{selected_id}\n{selected.mean_UTS:.0f} MPa, {selected.mean_EL:.1f}%",
                (selected.mean_UTS, selected.mean_EL), xytext=(8, -62), textcoords="offset points",
                fontsize=14, arrowprops={"arrowstyle": "-", "color": COLORS["dark"], "lw": .8},
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .95, "pad": 3}, zorder=8)
    ax.set(xlabel="Predicted UTS (MPa)", ylabel="Predicted EL (%)")
    ax.set_title("Stage II: process screening", pad=22)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.19),
               ncol=2, columnspacing=1, handletextpad=.4)
    clean_axis(ax, "both")
    panel_label(ax, "a")
    ranked_bars(axes[1], top, f"Top {len(top)} process routes",
                "Composite process score", "Process route", .70)
    panel_label(axes[1], "b")

    ax = axes[2]
    compare = ranked.set_index("candidate_id").loc[top.candidate_id.head(3)]
    positions = np.arange(len(compare))
    bottom = np.zeros(len(compare))
    weights = MODE_WEIGHTS["balanced"]
    for name, weight, color, label in zip(
        ["uncertainty_score", "performance_score", "diversity_score"], weights[:3],
        ["purple", "teal", "gold"], [r"$S_U$", r"$S_P$", r"$S_D$"]
    ):
        values = weight * compare[name].to_numpy()
        ax.bar(positions, values, bottom=bottom, width=.62, color=COLORS[color],
                edgecolor="white", linewidth=.6)
        for x, (base, value) in enumerate(zip(bottom, values)):
            if value >= .045:
                ax.text(x, base+value/2, label, ha="center", va="center", fontsize=14,
                        fontweight="bold", color=COLORS["dark"] if color == "gold" else "white")
        bottom += values
    penalty = -weights[3] * compare.constraint_penalty.to_numpy()
    if np.any(penalty):
        ax.bar(positions, penalty, width=.62, color=COLORS["orange"], label=r"$-0.10S_C$")
        ax.legend(frameon=False, loc="lower right", fontsize=12)
    for x, total in enumerate(compare.final_score):
        ax.text(x, max(bottom[x], total)+.016, f"{total:.3f}", ha="center", fontsize=14, fontweight="bold")
    ax.set_xticks(positions, [f"PROC-{short_id(cid)}" + ("\n(selected)" if cid == selected_id else "")
                             for cid in compare.index])
    ax.set(xlim=(-.7, len(compare)-.3), ylabel="Weighted score contribution",
           ylim=(min(0.0, float(penalty.min())-.02) if np.any(penalty) else 0,
                 max(.72, float(bottom.max())+.06)))
    ax.set_title("Score decomposition", pad=22)
    clean_axis(ax)
    panel_label(ax, "c")
    fig.subplots_adjust(wspace=.40, left=.06, right=.99, bottom=.25, top=.85)
    save_figure(fig, directory, "Fig_S7")


def export_tables(results, tables):
    top = pd.read_csv(results / "04_composition/composition_topk.csv")
    composition = top.apply(lambda row: "-".join(f"{name.removesuffix(' (wt%)')}{row[name]:.4f}"
                                                for name in ELEMENT_COLS if row[name] > 0), axis=1)
    table = top[["rank", "candidate_id", "mean_UTS", "mean_EL", "final_score"]].copy()
    table.insert(2, "composition_wt_pct", composition)
    table.to_csv(tables / "Table_S5.csv", index=False)
    top = pd.read_csv(results / "05_processing/process_topk.csv")
    ranked = pd.read_csv(results / "05_processing/process_ranked.csv").set_index("candidate_id")
    table = ranked.loc[top.candidate_id, ["composition_id", "mean_UTS", "mean_EL", "final_score", "initial_temp", "initial_time",
                                        "initial_deformation", "secondary_temp", "secondary_time", "deformation_per_pass", "passes",
                                        "total_cyclic_deformation", "final_total_deformation", "cooling_type"]].reset_index()
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    table.to_csv(tables / "Table_S6.csv", index=False)



def export_figures_and_tables(results, output):
    results, output = Path(results), Path(output)
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    setup_style()
    export_tables(results, tables)
    plot_s6(results, figures)
    plot_s7(results, figures)
