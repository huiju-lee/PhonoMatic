import phonopy
from phonopy import Phonopy
from phonopy.structure.atoms import Atoms as PhonopyAtoms

from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms as ASEAtoms

def phonopy_atoms_to_ase_atoms(phonopy_atoms):
    return ASEAtoms(
        symbols=phonopy_atoms.symbols,
        scaled_positions=phonopy_atoms.scaled_positions,
        cell=phonopy_atoms.cell,
        pbc=True
    )

class MlipPhononCalculator:
    """Calculates phonon properties using an MLIP calculator."""

    def __init__(self, structure: Structure, supercell_matrix, distance=0.01):
        """Initializes with a pymatgen structure and phonopy settings."""
       # Convert pymatgen Structure → ASE Atoms
        unitcell_ase = AseAtomsAdaptor.get_atoms(structure, msonable=False)

        # Convert ASE Atoms → Phonopy Atoms
        unitcell_phonopy = PhonopyAtoms(
            symbols=unitcell_ase.get_chemical_symbols(),
            scaled_positions=unitcell_ase.get_scaled_positions(),
            cell=unitcell_ase.get_cell(),
            masses=unitcell_ase.get_masses()
        )

        # Pass PhonopyAtoms into Phonopy
        self.phonon = Phonopy(unitcell_phonopy, supercell_matrix=supercell_matrix)
        self.phonon.generate_displacements(distance=distance)
        self.supercells_with_displacements = self.phonon.supercells_with_displacements

    def calculate_force_constants(self, mlip_calculator):
        """Calculates forces and produces force constants."""
        forces_set = []
        for sc in self.supercells_with_displacements:
            sc_atoms = phonopy_atoms_to_ase_atoms(sc)
            sc_atoms.calc = mlip_calculator
            forces = sc_atoms.get_forces()
            forces_set.append(forces)
        
        self.phonon.forces = forces_set
        self.phonon.produce_force_constants()
        return self.phonon
