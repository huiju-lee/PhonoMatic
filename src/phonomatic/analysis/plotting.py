import matplotlib.pyplot as plt
import yaml
import numpy as np


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

    return distance, frequencies, labels, seg_nqpoint

