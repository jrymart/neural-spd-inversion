#!/usr/bin/env python3
"""Render available plot-notebook figures to PNG for PR review.

Run from the project root:
    python notebooks/org/plots/render_figures.py

Figures that require data not present in the repo (shuffle_viz, shuffle_perf,
network_viz) are skipped with a note.
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import seaborn as sns
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[3] / "src"))
from neural_spd.config import PROJECT_ROOT, PLOTS_PATH
from neural_spd import plot_styles

FIG_DIR = PLOTS_PATH / "preview_pngs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SAVE_KW = dict(dpi=150, bbox_inches="tight", facecolor="white")


# ── performance.org ────────────────────────────────────────────────────────────
def render_performance():
    plot_styles.apply()
    plot_styles.single_column()

    PERF_METRICS_PATH = PROJECT_ROOT / "analysis/overall_performance.csv"
    RAW_PERF_PATH     = PROJECT_ROOT / "analysis/all_test_performance.csv"

    perf_metrics_df = pd.read_csv(PERF_METRICS_PATH)
    best_perf_row = perf_metrics_df[
        (perf_metrics_df["target"] == "DoK") &
        (perf_metrics_df["data"]   == "elevation") &
        (perf_metrics_df["noise"]  == 0)
    ].sort_values("nrmse").iloc[0]
    best_seed = best_perf_row["seed"]

    raw_perf_df = pd.read_csv(RAW_PERF_PATH)
    best_raw_perf_df = raw_perf_df[
        (raw_perf_df["seed"]   == best_seed) &
        (raw_perf_df["target"] == "DoK") &
        (raw_perf_df["data"]   == "elevation") &
        (raw_perf_df["noise"]  == 0)
    ]

    def plot_raw_perf(ax):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("True $(D/K)$")
        ax.set_ylabel("Inferred $(D/K)$")
        sns.kdeplot(
            data=best_raw_perf_df,
            x="true_labels", y="predictions",
            ax=ax, fill=True, alpha=1, levels=5,
            cmap="pub_greens",
        )
        ax2 = ax.twinx()
        ax2.set_ylabel("Count", color=plot_styles.COLORS["blue"],
                       fontsize=plt.rcParams["axes.labelsize"])
        ax2.tick_params(axis="y", labelcolor=plot_styles.COLORS["blue"])
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color(plot_styles.COLORS["blue"])
        sns.histplot(
            data=best_raw_perf_df, x="true_labels",
            color=plot_styles.COLORS["blue"],
            ax=ax2, alpha=0.25, edgecolor="none",
        )
        lims = [best_raw_perf_df["true_labels"].min(),
                best_raw_perf_df["true_labels"].max()]
        ax.plot(lims, lims, color=plot_styles.COLORS["gray_mid"],
                linestyle="--", linewidth=0.8, zorder=3)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=plot_styles.COLORS["green_dark"],
                  label="Predicted vs. True density"),
            Patch(facecolor=plot_styles.COLORS["blue"], alpha=0.4,
                  label="True $D/K$ distribution"),
            Line2D([0], [0], color=plot_styles.COLORS["gray_mid"],
                   linestyle="--", linewidth=0.8, label="1:1 line"),
        ]
        ax.legend(handles=legend_elements, loc="upper left")

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    plot_raw_perf(ax)
    fig.savefig(FIG_DIR / "performance_raw.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓  performance_raw.png")


# ── perf_compare.org ───────────────────────────────────────────────────────────
def render_perf_compare():
    plot_styles.apply()
    plot_styles.double_column()

    PERF_METRICS_PATH = PROJECT_ROOT / "analysis/overall_performance.csv"
    target_names = {
        'DoK':    "$D/K$",
        'KoD':    "$K/D$",
        'logDoK': r"$\log_{10}(D/K)$",
        'logKoD': r"$\log_{10}(K/D)$",
    }
    data_names = {
        'elevation':    "Elevation",
        'flowacc':      "Flow Acc",
        'log10flowacc': r"$\log_{10}(\mathrm{Flow\ Acc})$",
        'slope':        "Slope",
        'curvature':    "Curvature",
    }

    perf_metrics_df = pd.read_csv(PERF_METRICS_PATH)
    perf_metrics_df["target"] = perf_metrics_df["target"].map(target_names)
    perf_metrics_df["data"]   = perf_metrics_df["data"].map(data_names)

    hue_order = ["$D/K$", r"$\log_{10}(D/K)$", "$K/D$", r"$\log_{10}(K/D)$"]
    palette   = [
        plot_styles.COLORS["green_dark"],  # D/K      — dark green
        "#82bb8e",                         # log(D/K) — light green
        "#1d5f8a",                         # K/D      — dark blue
        "#7aadcc",                         # log(K/D) — light blue
    ]

    def plot_perf_comp(ax):
        sns.barplot(
            data=perf_metrics_df,
            x="data", y="nrmse", hue="target",
            hue_order=hue_order,
            ax=ax,
            palette=palette,
            edgecolor="none",
            capsize=0.04,
            err_kws={"linewidth": 0.8},
        )
        ax.set_ylabel("NRMSE")
        ax.set_xlabel("Input Data Type")
        ax.legend(title="Target", loc="upper right")
        ax.tick_params(axis="x", labelrotation=15)
        for label in ax.get_xticklabels():
            label.set_ha("right")

    fig, ax = plt.subplots(figsize=(6.75, 3.0))
    plot_perf_comp(ax)
    fig.savefig(FIG_DIR / "perf_compare.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓  perf_compare.png")


# ── target_distributions.org ──────────────────────────────────────────────────
def render_target_distributions():
    plot_styles.apply()
    plot_styles.double_column()

    target_dist_path = PROJECT_ROOT / "analysis/all_test_performance.csv"
    target_names = {
        'DoK':    "$D/K$",
        'KoD':    "$K/D$",
        'logDoK': r"$\log_{10}(D/K)$",
        'logKoD': r"$\log_{10}(K/D)$",
    }

    target_dist_df = pd.read_csv(target_dist_path)
    target_dist_df = target_dist_df[
        (target_dist_df["seed"]  == 0) &
        (target_dist_df["noise"] == 0)
    ]
    target_dfs = {
        t: target_dist_df[target_dist_df["target"] == t]
        for t in target_dist_df["target"].unique()
    }

    def plot_target_dists(axs):
        for ax, (target, df) in zip(axs.flat, target_dfs.items()):
            ax.hist(df["true_labels"], bins=50,
                    color=plot_styles.COLORS["green_dark"],
                    edgecolor="none", alpha=0.85)
            ax.set_xlabel(target_names[target])
            ax.set_ylabel("Count")
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
            )

    fig, axs = plt.subplots(2, 2, figsize=(6.75, 4.5))
    plot_target_dists(axs)
    fig.savefig(FIG_DIR / "target_distributions.png", **SAVE_KW)
    plt.close(fig)
    print("  ✓  target_distributions.png")


if __name__ == "__main__":
    print(f"Writing preview PNGs to {FIG_DIR}\n")
    render_performance()
    render_perf_compare()
    render_target_distributions()
    print("\nSkipped (data not present in repo):")
    print("  –  shuffle_viz.png    (needs model_runs.db + elevation .npy files)")
    print("  –  shuffle_perf.png   (needs analysis/all_mod_performance.csv)")
    print("  –  network_viz.png    (needs pygraphviz system library)")
