"""
Plotting styles for different output formats.
Ensures consistency across paper, slides, and posters.
Works with both matplotlib and seaborn.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# Try to import seaborn
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Color scheme - define once, use everywhere
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'control': '#666666',
}

# Seaborn palette derived from our colors (for seaborn plots)
SNS_PALETTE = [
    COLORS['primary'],
    COLORS['secondary'],
    COLORS['accent'],
    '#C73E1D',  # Additional colors if needed
    '#6A994E',
    '#BC4749',
]

def set_style(output_type='paper', use_seaborn=True):
    """
    Set matplotlib style for different output types.
    
    Parameters
    ----------
    output_type : str, one of 'paper', 'slides', 'poster'
    use_seaborn : bool
        If True and seaborn is available, apply seaborn styling on top
        of matplotlib settings. If False, use pure matplotlib.
    """
    
    # First, set matplotlib parameters
    if output_type == 'paper':
        # High resolution, small fonts, suitable for Nature/Science style journals
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 8,
            'axes.labelsize': 9,
            'axes.titlesize': 10,
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'legend.fontsize': 8,
            'figure.titlesize': 10,
            'lines.linewidth': 1.0,
            'lines.markersize': 4,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        })
        context_scale = 0.8  # For seaborn context
        
    elif output_type == 'slides':
        # Lower resolution, larger fonts, high contrast
        plt.rcParams.update({
            'figure.dpi': 150,
            'savefig.dpi': 150,
            'font.size': 14,
            'axes.labelsize': 16,
            'axes.titlesize': 18,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.titlesize': 20,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        })
        context_scale = 1.3  # For seaborn context
        
    elif output_type == 'poster':
        # Very high resolution, very large fonts
        plt.rcParams.update({
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'font.size': 20,
            'axes.labelsize': 30,
            'axes.titlesize': 60,
            'xtick.labelsize': 20,
            'ytick.labelsize': 20,
            'legend.fontsize': 20,
            'figure.titlesize': 32,
            'lines.linewidth': 3.0,
            'lines.markersize': 10,
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        })
        context_scale = 1.8  # For seaborn context
    
    # Common settings for all output types
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,  # TrueType fonts for PDFs (better for editing)
        'ps.fonttype': 42,
    })
    
    # Apply seaborn styling if requested and available
    if use_seaborn and HAS_SEABORN:
        # Use whitegrid style (clean, professional)
        sns.set_style("whitegrid", {
            'axes.edgecolor': '.8',
            'grid.color': '.9',
        })
        
        # Set color palette to match our colors
        sns.set_palette(SNS_PALETTE)
        
        # Adjust seaborn's context scaling to match our font sizes
        # Note: calling set_context will override some rcParams, so we
        # may need to reapply certain settings
        sns.set_context("notebook", font_scale=context_scale)
        
        # Re-apply our specific font sizes (seaborn may have changed them)
        plt.rcParams.update({
            'font.size': plt.rcParams['font.size'],  # Keep what we set
        })

def get_figsize(output_type='paper', aspect_ratio=1.5):
    """
    Get appropriate figure size for output type.
    
    Parameters
    ----------
    output_type : str
    aspect_ratio : float, width/height ratio
    
    Returns
    -------
    tuple of (width, height) in inches
    """
    
    if output_type == 'paper':
        # Typical single-column width in inches
        width = 3.5
    elif output_type == 'slides':
        # Larger for visibility
        width = 8
    elif output_type == 'poster':
        # Very large for poster
        width = 12
    
    height = width / aspect_ratio
    return (width, height)

def get_seaborn_style(output_type='paper'):
    """
    Get appropriate seaborn style settings for output type.
    
    Returns a dict that can be used with sns.set_style() or as context manager.
    
    Example
    -------
    >>> with sns.axes_style(**get_seaborn_style('paper')):
    ...     sns.scatterplot(data=df, x='x', y='y')
    """
    if not HAS_SEABORN:
        return {}
    
    base_style = {
        'axes.edgecolor': '.8',
        'grid.color': '.9',
        'axes.spines.top': False,
        'axes.spines.right': False,
    }
    
    if output_type == 'paper':
        base_style.update({
            'grid.linewidth': 0.5,
            'axes.linewidth': 0.8,
        })
    elif output_type in ['slides', 'poster']:
        base_style.update({
            'grid.linewidth': 1.0,
            'axes.linewidth': 1.2,
        })
    
    return base_style

def apply_seaborn_palette():
    """
    Apply our custom color palette to seaborn.
    Call this after importing seaborn if you want to use our colors.
    """
    if HAS_SEABORN:
        sns.set_palette(SNS_PALETTE)

# Convenience function for common seaborn plots
def seaborn_plot_wrapper(plot_func, *args, output_type='paper', **kwargs):
    """
    Wrapper for seaborn plotting functions that ensures consistent styling.
    
    Parameters
    ----------
    plot_func : callable
        A seaborn plotting function (e.g., sns.scatterplot, sns.violinplot)
    output_type : str
        'paper', 'slides', or 'poster'
    *args, **kwargs
        Arguments passed to plot_func
        
    Returns
    -------
    The return value of plot_func
    
    Example
    -------
    >>> ax = seaborn_plot_wrapper(sns.scatterplot, data=df, x='x', y='y', 
    ...                           output_type='slides')
    """
    set_style(output_type, use_seaborn=True)
    return plot_func(*args, **kwargs)

def setup_poster_style():
    """
    Set up consistent matplotlib/seaborn styling for poster figures.
    This creates a professional, cohesive look across all plots.
    """

    # Define color palette - earthy tones that work well for geomorphology
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

    return colors
