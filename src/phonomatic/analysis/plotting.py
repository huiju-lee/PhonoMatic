import matplotlib.pyplot as plt
import re
import yaml
import numpy as np
from itertools import accumulate
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages
from collections import defaultdict


# This should probably be moved to io.py
def _load_band_yaml(band_yaml_path):
    """
    Load phonon band structure data from a phonopy band.yaml file.

    Args:
        band_yaml_path (Path): Path to yaml file containing phonon band 
            structure data.

    Returns: 
        distance (np.ndarray): Shape (n_qpoints,), cumulative distance along
            the path for each q-point.
        frequencies (np.ndarray): Shape (n_bands, n_qpoints), phonon
            frequencies in THz.
        labels (list of str): High-symmetry point labels for x-axis ticks.
        seg_nqpoint (list of int): Number of q-points sampled along each
            segment of reciprocal space. 
        npath (int): Number of distinct path segments.  
    """
    with open(band_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    phonon = data["phonon"]
    npath = data["npath"]
    seg_nqpoint = data["segment_nqpoint"]
    # Convert to array to flatten, then to list for 
    # easier deletion / appending
    labels = np.array(data["labels"]).flatten().tolist()

    distance = []
    freqs = []
    xtick_positions = []

    for qpt in phonon:
        # Get distance and bands
        dist = qpt["distance"]
        distance.append(dist)
        freqs.append([b["frequency"] for b in qpt["band"]])

    # Convert to arrays
    distance = np.array(distance)
    frequencies = np.array(freqs).T  # shape (n_bands, n_qpoints)

    # Merge consecutive duplicate labels
    for j in range(1, npath):
        if labels[j] == labels[j+1]:
            del labels[j]
        else:
            labels[j] = labels[j] + "/" + labels[j+1]
            del labels[j+1]

    return distance, frequencies, labels, seg_nqpoint, npath


def _get_method_and_material(band_yaml_path):
    """
    NOTE: This function may need to be changed to account for different 
    interatmic potentials. Currently, we are only working with DFT and 
    the MACE model. 

    Extracts material label ({chemical formula}-{index}) and 
    method of computation (mlip or dft) for a provided Phonopy band yaml
    file. This information is used for the label and title in 
    plot_phonon_dispersion.

    Args:
        band_yaml_path (Path): Path to the band yaml file output by Phonopy.

    Returns:
        material_id (str): The material label ({chemical_formula}-{index}). 
        method (str): Describes the method by which the frequencies in the 
            band taml file were computed (either mlip or dft). 
    """
    parent_folder = Path(band_yaml_path).parent.name
    
    # We want to capture material-index as title and method as label
    pattern = r"POSCAR-([A-Za-z0-9]+-[0-9]+)-(mlip|dft)$"
    match = re.match(pattern, parent_folder)
    if match:
        material_id = match.group(1)  
        method = match.group(2)   # "mlip" or "dft"
    else:
        # fallback values if pattern doesn't match
        material_id = "MaterialUnknown"
        method = "MethodUnknown"

    return method, material_id


def _create_output_file(path, file_name):
    """
    Creates an output file by checking the validity of a user-specified file
    path. 

    Args: 
        path (Path or str): The user-specified path.
        file_name (str): An output file name (in case the user only specified
            a directory). 
    """
    if path is None:
        path = Path.cwd() / "visualizations" / file_name
    else:
        path = Path(path)

    # Ensure directories exist
    if path.suffix == "" or path.is_dir():
        opath.mkdir(parents=True, exist_ok=True)
        path = path / file_name
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
    
        
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
        labels = [_get_method_and_material(f)[0] for f in yaml_files]
    if styles is None:
        styles = [{'color': 'C' + str(i)} for i in range(len(yaml_files))]

    # Load first dataset for x-ticks and metadata
    dist, freqs, xtick_labels, seg_nqpoint, npath = (
        _load_band_yaml(yaml_files[0])
    )

    # Decide whether to create our own figure
    own_fig = False
    if ax is None:
        own_fig = True
        output_path = _create_output_file(output_path, "band_plot.png")

        fig, ax = plt.subplots(figsize=plotting_args.get("figsize", (4, 4)))

    xlabel = plotting_args.get("xlabel", "Wave vector")
    ylabel = plotting_args.get("ylabel", "Frequency (THz)")
    title = plotting_args.get("title", _get_method_and_material(yaml_files[0])[1])
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
        dist, freqs, _, _, _ = _load_band_yaml(yaml_file)
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
        plt.savefig(output_path, dpi=300)
        plt.close()


def _get_grouped_paths(dirs, file_of_interest):
    """
    Helper function to group a list of file directories by material
    configuration. Used to plot dispersion bands for the same material
    together in plot_all_dispersion_curves.

    Args:
        dirs (list of Path): List containing paths to computation results for
            input POSCAR files ('POSCAR' should be included in path names). 
        file_of_interest (str): Phonopy output file name to be extracted from 
            material directories. 

    Returns: 
        grouped_dirs_list (list of list of Path): Output paths grouped 
            by material configuration. 
        **plotting_args: Additional kwargs for figure customization (figsize,
            xlabel, ylabel, etc.).
    """
    grouped_dirs = defaultdict(list)
    for d in dirs:
        stem = d.stem
        match = re.match(r"^POSCAR-([A-Za-z0-9]+-\d+)-.*$", stem)
        if match:
            # Material ID is formula + ICSD
            material_id = match.group(1)
        else:
            # Fallback: use the whole stem as its own ID
            material_id = stem
        grouped_dirs[material_id].append(d / file_of_interest)
    grouped_dirs_list = list(grouped_dirs.values())
    return grouped_dirs_list


def _get_page_layout(plots_this_page):
    """
    Helper function to determine the arrangment of plots on a single page.

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
    output_pdf = _create_output_file(output_pdf, 'band_plots.pdf')

    # Get output directories for unique materials
    material_dirs = [d.resolve() for d in results_dir.iterdir() 
                     if 'POSCAR' in str(d)]
    
    # For materials that share the same configuration (chemical formula
    # and ICSD), plot them together in the same figure 
    grouped_paths = _get_grouped_paths(material_dirs, 
                                       file_of_interest='band.yaml')
    n_plots = len(grouped_paths)
    if n_plots == 0:
        print("No materials found in directory.")
        return
    n_pages = np.ceil(n_plots / max_plots_per_page)
    # Create batches from path lists - we can only plot max_plots_per_page 
    # per page
    batched_paths = [grouped_paths[i:max_plots_per_page] 
                     for i in range(0, n_plots, max_plots_per_page)]
    with PdfPages(output_pdf) as pdf:
        for batch in batched_paths:
            # For now, always plot rectangular layout
            # TO-DO: Allow user to customize this
            n_row, n_col = _get_page_layout(len(batch))
            fig, axs = plt.subplots(n_row, n_col,
                                    figsize=(8.5, 11),
                                    constrained_layout=True)
            # Flatten to 1D array for easier indexing
            axs = np.atleast_1d(axs).flatten()  

            # Plot each material into its subplot
            for ax, yaml_group in zip(axs, batch):
                plot_phonon_dispersion(yaml_group, ax=ax, 
                                       styles=styles, **plotting_args)

            # Turn off unused subplots
            for ax in axs[len(batch):]:
                ax.axis('off')

            if common_legend:
                handles, labels = axs[0].get_legend_handles_labels()
                default_legend_args = dict(
                    loc="lower center",
                    bbox_to_anchor=(0.5, -0.05),
                    fontsize=10,
                    ncol=1
                )
    
                # Merge user-provided args
                legend_args = {**default_legend_args, **(legend_params or {})}
                
                # Add legend
                fig.legend(handles, labels, **legend_args)

            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)