import phonopy
from phonopy import Phonopy
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

class MlipPhononCalculator:
    """Calculates phonon properties using an MLIP calculator."""

    def __init__(self, structure: Structure, supercell_matrix, distance=0.01):
        """Initializes with a pymatgen structure and phonopy settings."""
        unitcell = AseAtomsAdaptor.get_atoms(structure)
        self.phonon = Phonopy(unitcell, supercell_matrix=supercell_matrix)
        self.phonon.generate_displacements(distance=distance)
        self.supercells_with_displacements = self.phonon.supercells_with_displacements

    def calculate_force_constants(self, mlip_calculator):
        """Calculates forces and produces force constants."""
        forces_set = []
        for sc in self.supercells_with_displacements:
            sc_atoms = AseAtomsAdaptor.get_atoms(sc)
            sc_atoms.calc = mlip_calculator
            forces = sc_atoms.get_forces()
            forces_set.append(forces)
        
        self.phonon.forces = forces_set
        self.phonon.produce_force_constants()
        return self.phonon
