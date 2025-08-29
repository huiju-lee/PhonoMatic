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

from collections import defaultdict


def _validate_dos_augment(dos_augment):
    valid_augs = {None, "pdos", "total_dos", "both"}             
    if dos_augment not in valid_augs:
        raise ValueError(
            f"Invalid dos_augment '{dos_augment}'. "
            f"Must be one of {valid_augs}."
        )


def _setup_dispersion_fig(
    dos_augment=None, 
    figure_kwargs=None, 
    num_plots=1
):
    default_figure_kwargs = {"figsize": (6, 4)}
    figure_kwargs = {**default_figure_kwargs, **(figure_kwargs or {})}
    rows, cols = _get_page_layout(num_plots)

    if (dos_augment == 'pdos' or dos_augment == 'total_dos'):
        panels_per_plot = 2  
        width_ratios =  [2, 1]
    elif dos_augment == 'both':
        panels_per_plot = 3
        width_ratios = [2, 1, 1]
    else:
        panels_per_plot = 1
        width_ratios = [1]
        
    fig = plt.figure(**figure_kwargs)
    all_axes = []

    # One grid for entire page or image
    outer_gs = fig.add_gridspec(
        nrows=rows,
        ncols=cols,
    )

    for idx in range(num_plots):
        # Get row and col corresponding to that index
        r, c = divmod(idx, cols)
        # Sub-gridspec for this plot group
        sub_gs = outer_gs[r, c].subgridspec(
            nrows=1,
            ncols=panels_per_plot,
            width_ratios=width_ratios
        )

        group_axes = []
        # For each panel, add a subplot
        for j in range(panels_per_plot):
            if j == 0:
                ax = fig.add_subplot(sub_gs[j])
            else:
                # Extra plots after first (the phonon dispersion plot)
                # should share y-axis with phonon dispersion plot
                ax = fig.add_subplot(sub_gs[j], sharey=group_axes[0])
            group_axes.append(ax)

        all_axes.append(group_axes)

    return fig, all_axes


def _get_dos_paths(band_yaml_paths, dos_augment):
    # To be moved to IO later...
    if dos_augment == 'pdos':
        return [Path(p).parent / 'projected_dos.dat' for p in band_yaml_paths]
    elif dos_augment == 'total_dos':
        return [Path(p).parent / 'total_dos.dat' for p in band_yaml_paths]
    else:
        pdos_paths = [Path(p).parent / 'projected_dos.dat' for p in band_yaml_paths]
        total_paths = [Path(p).parent / 'total_dos.dat' for p in band_yaml_paths]
        return pdos_paths, total_paths # Separate for ease of access


def _add_dos_augment(aug, axs, dos_data_paths):
    if aug == 'pdos':
        plot_projected_dos(dos_data_paths, axs[0])
    elif aug == 'total_dos':
        plot_total_dos(dos_data_paths, axs[0])
    else:
        pdos_paths, total_paths = dos_data_paths
        plot_projected_dos(pdos_paths, axs[0])
        plot_total_dos(total_paths, axs[1])

        
def plot_phonon_dispersion(
        yaml_files, 
        output_path=None,
        labels=None, 
        line_styles=None,
        axis_kwargs=None,
        figure_kwargs=None, 
        legend_kwargs=None, 
        postprocess=None, 
        dos_augment=None,
        axes=None    
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
    # If single dict was passed, convert to list for cycling
    elif isinstance(line_styles, dict):
        line_styles = [line_styles]

    # Cycle through line styles if there are fewer styles than datasets
    style_cycler = cycle(line_styles)

    # Load band yaml data
    band_data = [load_band_yaml(f) for f in yaml_files]

    # Load first dataset for x-ticks and metadata
    dist, _, xtick_labels, seg_nqpoint, npath = band_data[0]

    # Decide whether to create our own figure - we do if none was passed
    own_fig = False
    if axes is None:
        own_fig = True
        output_path = create_output_file(output_path, "band_plot.png")
        _validate_dos_augment(dos_augment)
        # In the case where we create a PDF, _setup_dispersion_fig is called
        # in _make_dispersion_page
        fig, axes = _setup_dispersion_fig(dos_augment, figure_kwargs)
        # _setup_dispersion_fig returns a list of lists - unpack to get
        # list of axes for this singular plot
        axes = axes[0]  

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
    dispersion_ax = axes[0]
    dispersion_ax.set_xticks(xtick_positions)
    dispersion_ax.set_xticklabels(xtick_labels)
    dispersion_ax.tick_params(**tick_params)

    # Plot datasets
    for (dist, freqs, _, _, _), label in zip(band_data, labels):
        style = next(style_cycler)
        for j in range(freqs.shape[0]):
            print("style type:", type(style), style)
            dispersion_ax.plot(dist, freqs[j],
                               label=label if j == 0 else "", **style)

    # Annotate with axis labels
    dispersion_ax.set_xlabel(axis_kwargs.get("xlabel", "Wave vector"), 
                             fontdict=axis_kwargs.get("xlabel_fontdict", {}))
    dispersion_ax.set_ylabel(axis_kwargs.get("ylabel", "Frequency (THz)"), 
                             fontdict=axis_kwargs.get("ylabel_fontdict", {}))
    # Default title is material ID
    dispersion_ax.set_title(axis_kwargs.get("title", 
                                get_method_and_material(yaml_files[0])[1]), 
                            fontdict=axis_kwargs.get("title_fontdict", {}))

    # Set x-axis limits, plot 0-frequency line
    dispersion_ax.set_xlim(min(dist), max(dist))
    dispersion_ax.axhline(0, linestyle='--', color='lightcoral', lw=0.3)

    # Add subplots for DOS / pDOS
    dos_axs = axes[1:]
    if len(dos_axs) > 0:
        dos_paths = _get_dos_paths(yaml_files, dos_augment)
        _add_dos_augment(dos_augment, dos_axs, dos_paths)
        
    # Let the user make additional stylistic changes beyond the parameters
    # included in this function
    # TO-DO: Allow post-processing of DOS augmentations
    if postprocess:
        postprocess(dispersion_ax)

    # Save as a single file if we are generating one figure in isolation
    if own_fig:
        # Add legend - if part of a larger PDF we would use a common legend
        dispersion_ax.legend(**(legend_kwargs or {}))
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
    postprocess_page, 
    dos_augment
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

    # Define layout of page
    fig_kwargs = {"figsize": (8.5, 11), "constrained_layout": True}
    fig, group_axs = _setup_dispersion_fig(dos_augment, fig_kwargs, 
                                     num_plots=len(batch))

    # Cycle through user-supplied plotting and stylistic 
    # Next set of line styles for a given submplot
    line_styles_next = _make_cycler(line_styles)         
    axis_kwargs_next = _make_cycler(axis_kwargs)
    postprocess_next = _make_cycler(postprocess_subplot)

    # Plot each phonon dispersion curve in the batch
    for group_ax, yaml_group in zip(group_axs, batch):
        plot_phonon_dispersion(
            yaml_group,
            axes=group_ax,
            line_styles=line_styles_next(),
            axis_kwargs=axis_kwargs_next(),
            postprocess=postprocess_next(),
            dos_augment=dos_augment
        )

    # Turn off unused subplots
    for group_ax in group_axs[len(batch):]:
        for ax in group_ax:
            ax.axis('off')

    # Add legend
    if common_legend:
        handles, labels = group_axs[0][0].get_legend_handles_labels()
        default_args = dict(loc="lower center", bbox_to_anchor=(0.5, -0.05), 
                            fontsize=10, ncol=1)
        # Merge user-provided legend parameters with the defaults
        legend_args = {**default_args, **(legend_params or {})}
        fig.legend(handles, labels, **legend_args)
    
    # Allow user to pass additional function to modify page if they
    # weren't able to do so with the hard-coded parameters
    if postprocess_page:
        postprocess_page(fig, group_axs)

    return fig
            
        
def plot_all_dispersion_curves(
        results_dir,
        output_pdf=None, 
        line_styles=None,
        max_plots_per_page=40,
        common_legend=True,
        legend_params=None, 
        axis_kwargs=None, 
        figure_kwargs=None,
        postprocess_subplot=None, 
        postprocess_page=None, 
        dos_augment=None
):
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
        legend_params (dict): Optional keyword arguments passed to 
            `fig.legend()` if common_legend is True.
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
                              postprocess_page=postprocess_page, 
                              dos_augment=dos_augment)
        for batch in batched_paths
    ]

    # Combine and save to pdf
    save_figures_to_pdf(figures, output_pdf, file_name='band_plots.pdf')


#============================= Plot DOS / pDOS =============================#
def plot_total_dos(total_dos_paths, ax=None):
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        own_fig = True

    for path in total_dos_paths:
        method, _ = get_method_and_material(path)
        dos_data = np.loadtxt(path, comments="#")
        freq = dos_data[:, 0]
        dos = dos_data[:, 1]
        ax.plot(dos, freq)
    ax.set_xlabel("DOS (states/THz)")
    if own_fig:
        ax.set_ylabel("Frequency (THz)")
        plt.show()


def plot_projected_dos(projected_dos_paths, ax=None):
    """
    Plot projected DOS for one material across multiple methods (e.g., mlip vs
    dft). Can plot standalone or into an existing axis (e.g., as a subplot
    next to a dispersion plot).

    Args:
        projected_dos_paths (list[str]): One or more file paths to
            projected DOS files.
        ax (matplotlib.axes.Axes, optional): Axis to plot into. If None, a new
            figure/axis is created.
    """
    own_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        own_fig = True

    # Predefine a consistent color map for species
    color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    # Add new key (element) -> value is next color in cycle
    species_colors = defaultdict(lambda: next(color_cycle))

    for path in projected_dos_paths:
        method, _ = get_method_and_material(path)

        # Linestyle by method - use dashed for mlip by default
        linestyle = "--" if method.lower() == "mlip" else "-"

        # Load data
        pdos_data = np.loadtxt(path, comments="#")

        # Read symbols in correct order from top of file
        with open(path, "r") as f:
            symbols = f.readline().strip()  
        symbols = symbols.lstrip("#").split()  
        
        # Extract frequency and density arrays 
        # Set up dictionary to keep cumulative total of density by element
        freq = pdos_data[:, 0]
        densities = pdos_data[:, 1:]
        sym_to_pdos = defaultdict(lambda: np.zeros_like(freq))

        # Aggregate per species
        for sym, density in zip(symbols, densities.T):
            sym_to_pdos[sym] += density

        # Plot into axis
        for species, pdos in sym_to_pdos.items():
            ax.plot(
                pdos,
                freq,
                label=f"{species} ({method})",
                color=species_colors[species],
                linestyle=linestyle,
            )

    ax.set_xlabel("pDOS (states/THz)")
    ax.legend()
    if own_fig:
        # If we are augmenting with a phonon dispersion plot, 
        # we'll already have the y-label
        ax.set_ylabel("Frequency (THz)")
        plt.show()


