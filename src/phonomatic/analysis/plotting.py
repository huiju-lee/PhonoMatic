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
from itertools import accumulate
from pathlib import Path
    
        
def plot_phonon_dispersion(yaml_files, output_path=None,
        labels=None, styles=None, ax=None, **plotting_args):
    """
    Plot phonon dispersion curves from a list of phonopy band.yaml files.

    If `ax` is provided, plots into that axis and does NOT save a file.
    If `ax` is None, creates a new figure and saves to `output_path` if given.

    Args:
        yaml_files (list of Path or str): Paths to band.yaml files.
        output_path (Path or str): Path to save the output plot (default 
            bandplot.png).
        labels (list of str): Legend labels for each dataset. If None, 
            filenames are used.
        styles (list of dict): Matplotlib style kwargs for each dataset.
        ax (matplotlib.axes.Axes): Optional axis to plot into.
        **plotting_args: Additional kwargs for figure customization (figsize,
            xlabel, ylabel, etc.).
    """
    # Prepare labels and styles
    if labels is None:
        labels = [get_method_and_material(f)[0] for f in yaml_files]
    if styles is None:
        styles = [{'color': 'C' + str(i)} for i in range(len(yaml_files))]

    # Load first dataset for x-ticks and metadata
    dist, freqs, xtick_labels, seg_nqpoint, npath = (
        load_band_yaml(yaml_files[0])
    )

    # Decide whether to create our own figure
    own_fig = False
    if ax is None:
        own_fig = True
        output_path = create_output_file(output_path, "band_plot.png")

        fig, ax = plt.subplots(figsize=plotting_args.get("figsize", (4, 4)))

    xlabel = plotting_args.get("xlabel", "Wave vector")
    ylabel = plotting_args.get("ylabel", "Frequency (THz)")
    title = plotting_args.get("title", get_method_and_material(yaml_files[0])[1])
    tick_params = plotting_args.get(
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
    for yaml_file, label, style in zip(yaml_files, labels, styles):
        dist, freqs, _, _, _ = load_band_yaml(yaml_file)
        for j in range(freqs.shape[0]):
            ax.plot(dist, freqs[j], label=label if j == 0 else "", **style)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(min(dist), max(dist))
    ax.axhline(0, linestyle='--', color='lightcoral', lw=0.3)

    # Save as a single file if we are generating one figure in isolation
    if own_fig:
        ax.legend()
        plt.tight_layout()
        save_figure(fig, output_path, file_name="band_plot.png", dpi=300)


def _get_page_layout(plots_this_page):
    """
    Helper function to determine the arrangment of plots on a single page.
    
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


def _make_dispersion_page(batch, styles, common_legend,
                          legend_params, **plotting_args):
    """
    Creates a single page of phonon dispersion plots. 

    Args:
        batch (list of list of Path): Paths to band.yaml datasets for 
            plotting.
        styles (list of dict): Matplotlib style kwargs for each dataset used
            to make a single plot. 
        common_legend (Bool): Whether to add a common legend on the page. 
        legend_params (dict):  Optional keyword arguments passed to 
            `fig.legend()` if common_legend is True.
        **plotting_args: Additional kwargs for figure customization (figsize,
            xlabel, ylabel, etc.).

    Returns:
        fig (matplotlib.figure.Figure): The page of phonon dispersion curves.
    """
    # Define layout for this page
    n_row, n_col = _get_page_layout(len(batch))
    fig, axs = plt.subplots(n_row, n_col, 
                            figsize=(8.5, 11),
                            constrained_layout=True)
    # Flatten to 1D array for easier indexing
    
    axs = np.atleast_1d(axs).flatten()

    # Plot each phonon dispersion curve in batch
    for ax, yaml_group in zip(axs, batch):
        plot_phonon_dispersion(yaml_group, ax=ax, 
                               styles=styles, **plotting_args)

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

    return fig
            
        
def plot_all_dispersion_curves(results_dir, output_pdf=None, styles=None,
                               max_plots_per_page=40,
                               common_legend=True,
                               legend_params=None, 
                               **plotting_args):
    """
    Plots phonon dispersion curves for all materials contained in a
    provided results directory. Organizes the figures in a PDF document. 

    Args: 
        results_dir (Path): Directories containing outputs of Phonopy 
            computations.
        output_pdf (Path): Path to the desired output PDF file. 
        styles (list of dict): Matplotlib style kwargs for the datasets used
            to make a single plot. 
        max_plots_per_page (int): The maximum number of plots to be placed on
            a single page of the output PDF document.
        common_legend (Bool): Whether to add a common legend on each page of 
            the PDF output file. 
        legend_params (dict):  Optional keyword arguments passed to 
            `fig.legend()` if common_legend is True.
        **plotting_args: Additional kwargs for figure customization (figsize,
            xlabel, ylabel, etc.).
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
        _make_dispersion_page(batch, styles, common_legend, 
                              legend_params, **plotting_args)
        for batch in batched_paths
    ]

    # Combine and save to pdf
    save_figures_to_pdf(figures, output_pdf, file_name='band_plots.pdf')