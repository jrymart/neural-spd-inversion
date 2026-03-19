import seaborn as sns
import matplotlib.pyplot as plt

def setup_poster_style():
    colors = {
        'primary': '#2C5F2D',      # Forest green
        'secondary': '#97BC62',     # Light green
        'accent': '#4A7C59',        # Sage green
        'highlight': '#E57F00',     # Burnt orange (for emphasis)
        'neutral_dark': '#2F3E46',  # Charcoal
        'neutral_light': '#F4F4F9', # Off-white
        'gradient_1': '#1B3A4B',    # Deep blue (for elevation)
        'gradient_2': '#8B4513',    # Saddle brown (for terrain)
    }

    # Create custom colormap for terrain/elevation data
    terrain_cmap = plt.cm.get_cmap('terrain')
    flow_cmap = plt.cm.get_cmap('Blues')

    # Set seaborn style
    sns.set_style("whitegrid", {
        'axes.edgecolor': colors['neutral_dark'],
        'axes.linewidth': 1.5,
        'grid.color': '#E5E5E5',
        'grid.linewidth': 0.8,
    })

    # Matplotlib rcParams for consistent styling
    plt.rcParams.update({
        # Font settings - professional and readable from distance
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 14,  # Base font size

        # Figure settings
        'figure.facecolor': 'white',
        'figure.edgecolor': colors['neutral_dark'],
        'figure.dpi': 150,  # High DPI for poster quality

        # Axes settings
        'axes.labelsize': 16,
        'axes.titlesize': 18,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'medium',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': plt.cycler('color', [
            colors['primary'],
            colors['secondary'],
            colors['accent'],
            colors['highlight'],
            colors['gradient_1'],
            colors['gradient_2']
        ]),

        # Legend settings
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.edgecolor': colors['neutral_dark'],

        # Grid settings
        'axes.grid': True,
        'axes.axisbelow': True,

        # Tick settings
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'xtick.major.size': 6,
        'ytick.major.size': 6,

        # Line settings
        'lines.linewidth': 2.5,
        'lines.markersize': 8,

        # Save settings for high quality
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.facecolor': 'white',
    })
