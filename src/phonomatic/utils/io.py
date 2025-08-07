from pymatgen.core import Structure


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
    
