import matplotlib.pyplot as plt
import re
import yaml
import numpy as np
from itertools import accumulate
from pathlib import Path


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
        xtick_positions (list of float): Distances at each high-symmetry
                point for plotting.
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
    # Expect pattern like POSCAR-<material>-<index>-<method>
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


def plot_phonon_dispersion(yaml_files, output_path=None,
        labels=None, styles=None, **plotting_args):
    """
    Plot phonon dispersion curves from a list of phonopy band.yaml files.

    Args:
        yaml_files (list of Path or str): Paths to band.yaml files.
        output_path (Path or str): Path to save the output plot (e.g. 
            'plot.png').
        labels (list of str): Legend labels for each dataset. If None, 
            filenames are used.
        styles (list of dict): Matplotlib style kwargs for each dataset. If 
            None, defaults are used.
        **plotting_args: Additional kwargs for figure customization 
            (figsize, xlabel, ylabel, etc.).
    """
    # Define output path directory, make it if it does not exist
    if output_path is None:
        output_path = Path.cwd() / "visualizations"
        output_path.mkdir(parents=True, exist_ok=True)  
    else:
        output_path = Path(output_path)

    # If output_path is a directory, define a default filename inside it
    if output_path.is_dir():
        output_path = output_path / "bandplot.png"

    # Get method labels if none are passed
    if labels is None:
        labels = [_get_method_and_material(f)[0] for f in yaml_files]
    # Use default if no style dict is passed
    # Use a unique color for each dataset 
    if styles is None:
        styles = [{'color':'C' + str(i)} for i in range(len(yaml_files))]

    # Load first dataset to get xtick info
    dist, freqs, xtick_labels, seg_nqpoint, npath = (
            _load_band_yaml(yaml_files[0])
    )

    # Figure and axis labels
    figsize = plotting_args.get("figsize", (4, 4))
    xlabel = plotting_args.get("xlabel", "Wave vector")
    ylabel = plotting_args.get("ylabel", "Frequency (THz)")
    title = plotting_args.get("title", 
            _get_method_and_material(yaml_files[0])[1]) 
    tick_params = plotting_args.get(
        "tick_params",
        dict(
            axis='both',
            direction='in',
            top=True,
            right=True
        )
    )

    plt.figure(figsize=figsize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Set xticks based on segment_nqpoint from first file
    xticks = list(accumulate(seg_nqpoint))
    xtick_positions = (
        [0] 
        + [dist[xticks[j]] for j in range(npath - 1)] 
        + [dist[xticks[-1] - 1]]
    )
    plt.xticks(xtick_positions, xtick_labels)
    plt.tick_params(**tick_params)

    # Plot each dataset
    for yaml_file, label, style in zip(yaml_files, labels, styles):
        dist, freqs, _, _, _ = _load_band_yaml(yaml_file)
        num_frequencies = freqs.shape[0]
        for j in range(num_frequencies):
            plt.plot(dist, freqs[j], label=label if j == 0 else "", **style)

    plt.title(title)
    plt.legend()
    plt.xlim(min(dist), max(dist))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
