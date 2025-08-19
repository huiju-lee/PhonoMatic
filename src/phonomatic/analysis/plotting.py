import matplotlib.pyplot as plt
import numpy as np
from phonomatic.utils.io import (
    load_band_yaml,
    get_method_and_material, 
    create_output_file, 
    get_grouped_paths, 
    save_figure, 
    save_figures_to_pdf
)
from itertools import accumulate, cycle
from pathlib import Path
    
        
def plot_phonon_dispersion(
        yaml_files, 
        output_path=None,
        labels=None, 
        line_styles=None,
        axis_kwargs=None,
        figure_kwargs=None, 
        legend_kwargs=None, 
        postprocess=None, 
        ax=None
    ):
    """
    Plot phonon dispersion curves from a list of phonopy band.yaml files.

    If `ax` is provided, plots into that axis and does NOT save a file.
    If `ax` is None, creates a new figure and saves to `output_path` if given.

    Args:
        yaml_files (list of Path or str): Paths to band.yaml files.
        output_path (Path or str): Path to save the output plot (default
            bandplot.png).
        labels (list of str): Legend labels for each dataset. If None, 
            the computation methods are used.
        line_styles (list of dict): Matplotlib style kwargs for each dataset.
        axis_kwargs (dict): Axis customization options (xlabel, ylabel, 
            tick_params, etc.).
        figure_kwargs (dict): Figure customization options (e.g. figsize).
        legend_kwargs (dict): Legend customization options.
        postprocess (callable): Function taking `ax` for additional 
            customization.
        ax (matplotlib.axes.Axes): Optional axis to plot into.
    """
    # Prepare labels and styles
    if labels is None:
        labels = [get_method_and_material(f)[0] for f in yaml_files]
    if line_styles is None:
        line_styles = [{'color': 'C' + str(i)} 
                       for i in range(len(yaml_files))]
    # Cycle through line styles if there are fewer styles than datasets
    style_cycler = cycle(line_styles)

    # Load band yaml data
    band_data = [load_band_yaml(f) for f in yaml_files]

    # Load first dataset for x-ticks and metadata
    dist, _, xtick_labels, seg_nqpoint, npath = band_data[0]

    # Decide whether to create our own figure - we do if none was passed
    own_fig = False
    if ax is None:
        own_fig = True
        output_path = create_output_file(output_path, "band_plot.png")
        default_figure_kwargs = {"figsize": (4, 4)}
        # Merge user-supplied figure kwargs with defaults
        figure_kwargs = {**default_figure_kwargs, **(figure_kwargs or {})}
        fig, ax = plt.subplots(**figure_kwargs)

    axis_kwargs = axis_kwargs or {}
    tick_params = axis_kwargs.get(
        "tick_params",
        dict(axis='both', direction='in', top=True, right=True)
    )

    # Set x-ticks
    xticks = list(accumulate(seg_nqpoint))
    xtick_positions = ([0] 
                       + [dist[xticks[j]] for j in range(npath - 1)] 
                       + [dist[xticks[-1] - 1]])
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels)
    ax.tick_params(**tick_params)

    # Plot datasets
    for (dist, freqs, _, _, _), label in zip(band_data, labels):
        style = next(style_cycler)
        for j in range(freqs.shape[0]):
            ax.plot(dist, freqs[j], label=label if j == 0 else "", **style)

    # Annotate with axis labels
    ax.set_xlabel(axis_kwargs.get("xlabel", "Wave vector"), 
                  fontdict=axis_kwargs.get("xlabel_fontdict", {}))
    ax.set_ylabel(axis_kwargs.get("ylabel", "Frequency (THz)"), 
                  fontdict=axis_kwargs.get("ylabel_fontdict", {}))
    # Default title is material ID
    ax.set_title(axis_kwargs.get("title", 
                                 get_method_and_material(yaml_files[0])[1]), 
                 fontdict=axis_kwargs.get("title_fontdict", {}))

    # Set x-axis limits, plot 0-frequency line
    ax.set_xlim(min(dist), max(dist))
    ax.axhline(0, linestyle='--', color='lightcoral', lw=0.3)

    # Let the user make additional stylistic changes beyond the parameters
    # included in this function
    if postprocess:
        postprocess(ax)

    # Save as a single file if we are generating one figure in isolation
    if own_fig:
        # Add legend - if part of a larger PDF we would use a common legend
        ax.legend(**(legend_kwargs or {}))
        plt.tight_layout()
        save_figure(fig, output_path, file_name="band_plot.png", dpi=300)


def _get_page_layout(plots_this_page):
    """
    Helper function to determine the arrangement of plots on a single page.
    
    NOTE: I am choosing to keep this in the plotting module for now because 
    it has to do with the visual layout of our plots. It does not read or
    write data from/to a file. 

    Args:
        plots_this_page (int): Number of figures to be placed on the page. 
    
    Returns:
        n_row (int): Number of rows of figures. 
        n_col (int): Number of columns of figures. 
    """
    n_col = int(np.sqrt(plots_this_page))
    n_row = np.ceil(plots_this_page / n_col)

    # Since we will be placing figures into a standard, portrait-oriented 
    # document, we want the number of rows to be larger than the number
    # of columns to create a rectangular layout
    if n_col == n_row and n_col != 1:  # Single plot -> 1 row, 
                                       # 1 col is only option
        n_col -= 1
        n_row = np.ceil(plots_this_page / n_col)
    return int(n_row), n_col


def _make_cycler(param):
    """
    Create a callable that yields values from a parameter specification.

    This helper standardizes how parameters are applied across multiple
    subplots. It handles three cases:

    - If ``param`` is ``None``: the returned function always returns ``None``.
    - If ``param`` is not a list/tuple: the returned function always returns
      the same object (``param``).
    - If ``param`` is a list/tuple: the returned function cycles through its
      elements indefinitely, yielding one on each call.

    Args:
        param (Any | list | tuple | None): The parameter to wrap.

    Returns:
        cycle_func(Callable[[], Any]): A function that returns the next 
            parameter value when called.
    """
    if param is None:
        cycle_func = lambda: None
    elif isinstance(param, (list, tuple)):
        c = cycle(param)
        cycle_func = lambda: next(c)
    else:
        cycle_func = lambda: param
    return cycle_func


def _make_dispersion_page(
    batch, 
    line_styles, 
    common_legend,
    legend_params, 
    axis_kwargs,
    figure_kwargs,
    postprocess_subplot, 
    postprocess_page
):
    """
    Creates a single page of phonon dispersion plots.
    
    Args:
        batch (list of list of Path): Paths to band.yaml datasets for 
            plotting.
        line_styles (list of dict): Matplotlib style kwargs for the datasets 
            used to make a single plot.
        common_legend (bool): Whether to add a common legend on the page.
        legend_params (dict): Optional keyword arguments passed to 
            `fig.legend()`if common_legend is True.
        axis_kwargs (dict): Axis customization options for each plot 
            (xlabel, ylabel, tick_params, title, etc.).
        figure_kwargs (dict): Figure customization options passed to 
            plt.subplots. 
        postprocess_subplot (callable or list of callable): Function or list of
            functions taking an `ax` for additional customization of each 
            subplot.
        postprocess_page (callable): Function taking a `fig` and `axs` as
            arguments for customization of an entire PDF page. 
    
    Returns:
        fig (matplotlib.figure.Figure): The page of phonon dispersion curves.
    """

    # Define layout for this page
    n_row, n_col = _get_page_layout(len(batch))
    default_fig_kwargs = {"figsize": (8.5, 11), "constrained_layout": True}
    # Merge defaults with user-provided
    figure_kwargs = {**default_fig_kwargs, **(figure_kwargs or {})}
    fig, axs = plt.subplots(n_row, n_col, **figure_kwargs)
    
    # Flatten to 1D array for easier indexing
    axs = np.atleast_1d(axs).flatten()

    # Cycle through user-supplied plotting and stylistic 
    # Next set of line styles for a given submplot
    line_styles_next = _make_cycler(line_styles)         
    axis_kwargs_next = _make_cycler(axis_kwargs)
    postprocess_next = _make_cycler(postprocess_subplot)

    # Plot each phonon dispersion curve in the batch
    for ax, yaml_group in zip(axs, batch):
        plot_phonon_dispersion(
            yaml_group,
            ax=ax,
            line_styles=line_styles_next(),
            axis_kwargs=axis_kwargs_next(),
            postprocess=postprocess_next(),
        )

    # Turn off unused subplots
    for ax in axs[len(batch):]:
        ax.axis('off')

    # Add legend
    if common_legend:
        handles, labels = axs[0].get_legend_handles_labels()
        default_args = dict(loc="lower center", bbox_to_anchor=(0.5, -0.05), 
                            fontsize=10, ncol=1)
        # Merge user-provided legend parameters with the defaults
        legend_args = {**default_args, **(legend_params or {})}
        fig.legend(handles, labels, **legend_args)
    
    # Allow user to pass additional function to modify page if they
    # weren't able to do so with the hard-coded parameters
    if postprocess_page:
        postprocess_page(fig, axs)

    return fig
            
        
def plot_all_dispersion_curves(results_dir,
                               output_pdf=None, 
                               line_styles=None,
                               max_plots_per_page=40,
                               common_legend=True,
                               legend_params=None, 
                               axis_kwargs=None, 
                               figure_kwargs=None,
                               postprocess_subplot=None, 
                               postprocess_page=None):
    """
    Plots phonon dispersion curves for all materials contained in a
    provided results directory. Organizes the figures in a PDF document. 

    Args:
        results_dir (Path): Directory containing outputs of Phonopy 
            computations.
        output_pdf (Path): Path to the desired output PDF file.
        line_styles (list of dict OR list of list of dict): Matplotlib style  
            kwargs for plotting the dispersion curves. The user can either
            supply a list of dictionaries (one dictionary for each dataset 
            used to make a single plot), in which case all subplots will 
            share the same line style. Or, they can pass a list containing 
            plotting parameters for each subplot. 
        max_plots_per_page (int): Maximum number of plots to place on a 
            single page of the PDF.
        common_legend (bool): Whether to add a common legend on each page.
        legend_params (dict): Optional keyword arguments passed to `fig.legend()`
            if common_legend is True.
        axis_kwargs (dict or list of dict): Axis customization options for 
            each plot (xlabel, ylabel, tick_params, title, etc.). The user can
            either provide a single dict, in which case all subplots will 
            share the same axes configurations. Or, they can pass a list that
            will by cycled to give the subplots a variety of axes styles. 
        figure_kwargs (dict): Subplot customization options passed to 
            plt.subplots. 
        postprocess_subplot (callable or list of callable): Function or list of
            functions taking an `ax` for additional customization of each 
            subplot.
        postprocess_page (callable): Function taking a `fig` and `axs` as
            arguments for customization of an entire PDF page. 
    """
    results_dir = Path(results_dir)
    
    # Make sure path to output_pdf exists
    output_pdf = create_output_file(output_pdf, 'band_plots.pdf')

    # Get output directories for unique materials
    material_dirs = [d.resolve() for d in results_dir.iterdir() 
                     if 'POSCAR' in str(d)]
    
    # For materials that share the same configuration (chemical formula
    # and ICSD), plot them together in the same figure 
    grouped_paths = get_grouped_paths(material_dirs, 
                                      file_of_interest='band.yaml')
    n_plots = len(grouped_paths)
    if n_plots == 0:
        print("No materials found in directory.")
        return
    
    # Create batches from path lists - we can only plot max_plots_per_page 
    # per page
    batched_paths = [
        grouped_paths[i:i+max_plots_per_page] 
        for i in range(0, len(grouped_paths), max_plots_per_page)
    ]

    # Create each page
    figures = [
        _make_dispersion_page(batch,
                              line_styles=line_styles,
                              common_legend=common_legend,
                              legend_params=legend_params,
                              axis_kwargs=axis_kwargs,
                              figure_kwargs=figure_kwargs,
                              postprocess_subplot=postprocess_subplot, 
                              postprocess_page=postprocess_page)
        for batch in batched_paths
    ]

    # Combine and save to pdf
    save_figures_to_pdf(figures, output_pdf, file_name='band_plots.pdf')
