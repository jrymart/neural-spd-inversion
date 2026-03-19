#!/usr/bin/env python3

"""
Publication-quality matplotlib/seaborn style configuration.

Colorblind-friendly palette built around the green tones in existing figures,
extended with accessible complementary colors (tested against deuteranopia,
protanopia, and tritanopia via Coblis simulator recommendations).

Usage:
    import pub_style
    pub_style.apply()

    # Or just grab the palette:
    from pub_style import COLORS, PALETTE

    # For seaborn:
    import seaborn as sns
    sns.set_palette(pub_style.PALETTE)
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# COLOR DEFINITIONS — edit these to propagate everywhere
# ============================================================================

# Primary palette (colorblind-friendly, ordered for categorical data)
# Hex values chosen to be distinguishable under all three common
# color vision deficiencies (deuteranopia, protanopia, tritanopia).
COLORS = {
    # Greens — anchored to your existing figure aesthetic
    "green_dark":   "#2a7f3f",   # primary accent, dense regions
    "green_mid":    "#4caf50",   # secondary accent, medium density
    "green_light":  "#a5d6a7",   # fills, low density, backgrounds

    # Complementary colors — accessible contrast with the greens
    "blue":         "#3a7ca5",   # cool complement
    "orange":       "#e8850c",   # warm complement (safe for deuteranopia)
    "purple":       "#8e6fbf",   # neutral complement
    "gold":         "#d4a017",   # highlight / annotation
    "teal":         "#1a9988",   # secondary cool
    "rose":         "#c45b84",   # secondary warm

    # Grays — axes, text, gridlines
    "black":        "#1a1a1a",   # text, spines
    "gray_dark":    "#4a4a4a",   # tick labels, secondary text
    "gray_mid":     "#8c8c8c",   # gridlines
    "gray_light":   "#d9d9d9",   # minor gridlines, panel borders
    "white":        "#ffffff",   # background
}

# Ordered categorical palette (use for line plots, scatter, bar, etc.)
PALETTE = [
    COLORS["green_dark"],
    COLORS["blue"],
    COLORS["orange"],
    COLORS["purple"],
    COLORS["teal"],
    COLORS["rose"],
    COLORS["gold"],
    COLORS["green_mid"],
]

# Sequential colormaps for density / heatmap plots
# (matches your KDE green gradient)
GREEN_CMAP_COLORS = [COLORS["white"], COLORS["green_light"],
                     COLORS["green_mid"], COLORS["green_dark"], "#0d3f1a"]

# ============================================================================
# RCPARAMS
# ============================================================================

_rcparams = {
    # --- Figure ---
    "figure.figsize":        (3.5, 3.0),     # single-column width (inches)
    "figure.dpi":            300,
    "figure.facecolor":      COLORS["white"],
    "figure.edgecolor":      "none",
    "figure.constrained_layout.use": True,

    # --- Font ---
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":             8,
    "mathtext.fontset":      "dejavuserif",

    # --- Axes ---
    "axes.facecolor":        COLORS["white"],
    "axes.edgecolor":        COLORS["gray_dark"],
    "axes.linewidth":        0.6,
    "axes.grid":             False,
    "axes.titlesize":        9,
    "axes.titleweight":      "bold",
    "axes.titlepad":         6,
    "axes.labelsize":        8,
    "axes.labelpad":         4,
    "axes.labelcolor":       COLORS["black"],
    "axes.prop_cycle":       plt.cycler("color", PALETTE),
    "axes.spines.top":       False,
    "axes.spines.right":     False,

    # --- Ticks ---
    "xtick.labelsize":       7,
    "ytick.labelsize":       7,
    "xtick.color":           COLORS["gray_dark"],
    "ytick.color":           COLORS["gray_dark"],
    "xtick.direction":       "out",
    "ytick.direction":       "out",
    "xtick.major.width":     0.5,
    "ytick.major.width":     0.5,
    "xtick.major.size":      3,
    "ytick.major.size":      3,
    "xtick.minor.visible":   False,
    "ytick.minor.visible":   False,

    # --- Grid ---
    "grid.color":            COLORS["gray_light"],
    "grid.linewidth":        0.4,
    "grid.alpha":            0.7,

    # --- Legend ---
    "legend.fontsize":       7,
    "legend.frameon":        True,
    "legend.borderpad":      0.3,
    "legend.handlelength":   1.2,
    "legend.handletextpad":  0.4,
    "legend.labelspacing":   0.3,

    # --- Lines / Scatter ---
    "lines.linewidth":       1.2,
    "lines.markersize":      4,
    "scatter.marker":        "o",

    # --- Patches (bars, histograms) ---
    "patch.edgecolor":       COLORS["white"],
    "patch.linewidth":       0.4,

    # --- Histogram ---
    "hist.bins":             30,

    # --- Savefig ---
    "savefig.dpi":           600,
    "savefig.bbox":          "tight",
    "savefig.pad_inches":    0.02,
    "savefig.transparent":   False,
    "savefig.facecolor":     COLORS["white"],

    # --- PDF / SVG backend ---
    "pdf.fonttype":          42,     # TrueType — editable in Inkscape
    "ps.fonttype":           42,
    "svg.fonttype":          "none", # text as text, not paths
}

# ============================================================================
# COLORMAPS
# ============================================================================

def _make_green_cmap():
    """Sequential green colormap matching the density plots."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "pub_greens", GREEN_CMAP_COLORS, N=256
    )

green_cmap = _make_green_cmap()

# ============================================================================
# APPLY
# ============================================================================

def apply():
    """Apply publication rcParams globally."""
    mpl.rcParams.update(_rcparams)
    # Register custom cmap so it's available by name
    try:
        mpl.colormaps.register(green_cmap, name="pub_greens")
    except (ValueError, AttributeError):
        # Already registered or older mpl version
        pass

def double_column():
    """Switch to two-column figure width (7 in)."""
    mpl.rcParams.update({
        "figure.figsize": (7.0, 3.5),
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    })

def single_column():
    """Switch back to single-column width (3.5 in)."""
    mpl.rcParams.update({
        "figure.figsize": (3.5, 3.0),
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })


# ============================================================================
# QUICK REFERENCE (printed when run directly)
# ============================================================================

if __name__ == "__main__":
    print("=== Publication Style — Color Reference ===\n")
    for name, hexval in COLORS.items():
        print(f"  {name:<16s}  {hexval}")
    print(f"\n  Categorical palette order:")
    for i, c in enumerate(PALETTE):
        print(f"    [{i}] {c}")
    print(f"\n  Custom cmap: 'pub_greens'")
    print(f"\n  Inkscape-compatible: pdf.fonttype=42, svg.fonttype='none'")
    print(f"  Figure widths: single_column() → 3.5in, double_column() → 7.0in")
