import yaml
import re
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from pathlib import Path
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages


def load_structure(structure_path):
    """
    Loads a crystal structure given an input file path.

    Args:
        structure_path (Path): The path to the input file (e.g., POSCAR, 
            cif, etc.)

    Returns:
        pymatgen.core.Structure: The loaded crystal structure. 
    """
    return Structure.from_file(structure_path)


#================== Functions for Writing Phonopy Outputs ==================#
def write_force_constants(phonon, output_path):
    """
    Saves the force constants to a file.

    Args:
        phonon (Phonopy): The Phonopy object with force constants.
        output_path (Path): Directory where FORCE_CONSTANTS file will be saved.
    """
    try:
        phonopy_ifc = phonon.force_constants
        # Get shape of IFC tensor
        (natom1, natom2, dim1, dim2) = phonopy_ifc.shape
        ifc_str = str(natom1) + "\n"
        for i in range(natom1):
            for j in range(natom2):
                # For each pair of atoms i and j...
                ifc_str += str(i + 1) + " " + str(j + 1) + "\n"
                # Get 3 by 3 matrix describing effect of atom i on j 
                # upon being displaced in each spatial (xyz) direction
                mat_ifc = phonopy_ifc[i][j]
                for line in mat_ifc:
                    formatted_line = "  ".join(f"{val:.15f}" for val in line)
                    ifc_str += f"{formatted_line}\n"
        with open(output_path, "w") as file:
            file.write(ifc_str)
    except Exception as e:
        raise IOError(f"Error writing FORCE_CONSTANTS to {self.FC_file}: {e}")
        

def write_band_structure(phonon, output_path):
    """
    Saves the phonon band structure to a YAML file.

    Args:
        phonon (Phonopy): The Phonopy object.
        output_prefix (Path): Directory where band.yaml will be saved.
    """
    try:
        phonon.auto_band_structure(write_yaml=True, filename=output_path)
    except Exception as e:
        print(f"[WARNING] Could not write band structure YAML: {e}")


def write_thermal_properties(phonon, output_path, mesh):
    """
    Runs thermal properties calculation and saves to a YAML file.

    Args:
        phonon (Phonopy): The Phonopy object.
        output_path (Path): Directory where thermal.yaml will be saved.
        mesh (int): Mesh density for thermal properties.
    """
    try:
        phonon.run_mesh(mesh)
        phonon.run_thermal_properties(t_step=100, t_max=2000, t_min=100, 
                                      cutoff_frequency=0.1)
        phonon.write_yaml_thermal_properties(filename=output_path)
    except Exception as e:
        print(f"[WARNING] Could not write thermal properties YAML: {e}")


def write_total_dos(phonon, output_path):
    """
    Runs and saves the total phonon density of states (DOS).

    Args:
        phonon (Phonopy): The Phonopy object.
        output_path (Path): Directory where total_dos.dat will be saved.
    """
    try:
        # Running the mesh again is not necessary as long as we previously ran 
        # write_thermal_properties
        phonon.run_total_dos()
        phonon.write_total_dos(filename=output_path)
    except Exception as e:
        print(f"[WARNING] Could not write total DOS: {e}")


def save_phonopy_results(phonon, output_prefix, mesh):
    """
    Saves various phonon calculation results to files.

    Args:
        phonon (Phonopy): The Phonopy object containing calculated results.
        output_prefix (Path): Path to the directory for output files.
        mesh (int): Mesh density for thermal and DOS calculations.
    """
    output_paths = []
    for file_to_create in ['FORCE_CONSTANTS', 'band.yaml', 
                           'thermal.yaml', 'total_dos.dat']:
        path = Path(output_prefix / file_to_create)
        path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.append(path)
    write_force_constants(phonon, output_paths[0])
    write_band_structure(phonon, output_paths[1])
    write_thermal_properties(phonon, output_paths[2], mesh)
    write_total_dos(phonon, output_paths[3])


#====================== Helper Functions for Plotting ======================#
def load_band_yaml(band_yaml_path):
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


def get_method_and_material(band_yaml_path):
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


def create_output_file(path, file_name):
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
        path.mkdir(parents=True, exist_ok=True)
        path = path / file_name
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(figure, output_path, file_name, dpi=300):
    """
    Save a Matplotlib figure to a PNG file.

    Args:
        fig (matplotlib.figure.Figure): The figure to save.
         output_path (Path): Figure output path. 
        file_name (str): A filename used to create a complete save path if 
            the user only specifies an output directory. 
    """
    output_path = create_output_file(output_path, file_name)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def get_grouped_paths(dirs, file_of_interest):
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


def save_figures_to_pdf(figures, output_path, file_name):
    """
    Saves a list of figures to an output PDF file. 

    Args:
        figures (list of matplotlib.figure.Figure): Figures to be saved to 
            the PDF.
        output_path (Path): PDF output path. 
        file_name (str): A filename used to create a complete save path if 
            the user only specifies an output directory. 
    """
    output_path = create_output_file(output_path, file_name)
    with PdfPages(output_path) as pdf:
        for fig in figures:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
