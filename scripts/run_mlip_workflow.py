import yaml
import os
from pathlib import Path

# --- Import tools from your library ---
# Note the 'from phonomatic...' which works because of the src layout
from phonomatic.mlip_flow.phonopy import MlipPhononCalculator
from phonomatic.utils.io import load_structure, save_phonopy_results # You'd create these helpers

# --- Main workflow logic ---
def run_workflow():
    """
    Main function to run the MLIP phonon calculation workflow.
    """
    # 1. Load Configuration
    config_path = '../configs/config.yaml' # Path relative to the script location
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 2. Set up paths and parameters from the config file
    structures_dir = Path(config['paths']['structures_dir'])
    output_dir = Path(config['paths']['output_dir'])
    mlip_model_path = config['paths']['mlip_model_path']
    device = config['calculators']['mlip']['device']
    supercell_matrix = config['phonopy_settings']['supercell_matrix']

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize the MACE calculator (or any other MLIP)
    # This assumes MACE is installed in the environment
    from mace.calculators import MACECalculator
    mlip_calculator = MACECalculator(
        model_path=mlip_model_path, 
        device=device,
        default_dtype=config['calculators']['mlip']['dtype']
    )

    # 3. Loop Through Structures and Run Calculations
    # Example: find all POSCAR files in the structures directory
    for structure_file in structures_dir.glob("POSCAR-*"):
        print(f"--- Processing {structure_file.name} ---")

        # Load the crystal structure using a helper function
        structure = load_structure(structure_file)

        # Initialize your calculator class from the library
        phonon_calc = MlipPhononCalculator(structure, supercell_matrix)

        # Run the core calculation
        phonon_results_obj = phonon_calc.calculate_force_constants(mlip_calculator)
        
        # Save all results using another helper function
        output_prefix = output_dir / f"{structure_file.stem}-mlip"
        save_phonopy_results(
            phonon_results_obj, 
            output_prefix,
            mesh=config['phonopy_settings']['mesh_density']
        )
        print(f"Results saved with prefix: {output_prefix}")


if __name__ == "__main__":
    run_workflow()
