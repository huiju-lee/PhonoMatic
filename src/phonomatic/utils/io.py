from pymatgen.core import Structure
from pathlib import Path


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
        phonon (Phonopy): The Phonopy object with band structure.
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
        phonon.run_thermal_properties(t_step=100, t_max=2000, t_min=100, cutoff_frequency=0.1)
        phonon.write_yaml_thermal_properties(filename=output_path)
    except Exception as e:
        print(f"[WARNING] Could not write thermal properties YAML: {e}")


def write_total_dos(phonon, output_path):
    """
    Runs and saves the total phonon density of states (DOS).

    Args:
        phonon (Phonopy): The Phonopy object.
        output_path (Path): Directory where total_dos.dat will be saved.
        mesh (int): Mesh density for DOS calculation.
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
    for file_to_create in ['FORCE_CONSTANTS', 'band.yaml', 'thermal.yaml', 'total_dos.dat']:
        path = Path(output_prefix / file_to_create)
        path.parent.mkdir(parents=True, exist_ok=True)
        output_paths.append(path)
    write_force_constants(phonon, output_paths[0])
    write_band_structure(phonon, output_paths[1])
    write_thermal_properties(phonon, output_paths[2], mesh)
    write_total_dos(phonon, output_paths[3])

