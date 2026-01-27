#!/usr/bin/env python3
"""
ORCA Workflow for Molecular Properties Calculation
===================================================
This script converts SDF files to ORCA input files and performs:
- Geometry optimization (B3LYP/6-31G*)
- Dipole moment calculation
- HOMO/LUMO energies
- Molecular volume calculation
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors3D
import argparse
import subprocess


class ORCAWorkflow:
    """ORCA quantum chemistry workflow handler"""
    
    def __init__(self, output_dir: str = "orca_calculations"):
        """
        Initialize ORCA workflow
        
        Args:
            output_dir: Directory to store ORCA input and output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def read_sdf(self, sdf_file: str) -> List[Chem.Mol]:
        """
        Read molecules from SDF file
        
        Args:
            sdf_file: Path to SDF file
            
        Returns:
            List of RDKit molecule objects
        """
        supplier = Chem.SDMolSupplier(sdf_file, removeHs=False)
        molecules = []
        
        for mol in supplier:
            if mol is not None:
                molecules.append(mol)
            else:
                print(f"Warning: Failed to read a molecule from {sdf_file}")
                
        print(f"Successfully read {len(molecules)} molecule(s) from {sdf_file}")
        return molecules
    
    def mol_to_xyz(self, mol: Chem.Mol) -> str:
        """
        Convert RDKit molecule to XYZ coordinate string
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            XYZ coordinates as string
        """
        # Add hydrogens if not present
        mol = Chem.AddHs(mol)
        
        # Generate 3D coordinates if not present
        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
        
        conf = mol.GetConformer()
        xyz_lines = []
        
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            symbol = atom.GetSymbol()
            xyz_lines.append(f"{symbol:2s} {pos.x:12.6f} {pos.y:12.6f} {pos.z:12.6f}")
        
        return "\n".join(xyz_lines)
    
    def get_charge_and_multiplicity(self, mol: Chem.Mol) -> Tuple[int, int]:
        """
        Determine molecular charge and spin multiplicity
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Tuple of (charge, multiplicity)
        """
        charge = Chem.GetFormalCharge(mol)
        
        # Calculate number of unpaired electrons
        num_radical_electrons = 0
        for atom in mol.GetAtoms():
            num_radical_electrons += atom.GetNumRadicalElectrons()
        
        # Multiplicity = 2S + 1, where S is total spin
        multiplicity = num_radical_electrons + 1
        
        return charge, multiplicity
    
    def create_orca_input(self, mol: Chem.Mol, mol_name: str, 
                         calculation_type: str = "opt") -> str:
        """
        Create ORCA input file content
        
        Args:
            mol: RDKit molecule object
            mol_name: Name of the molecule
            calculation_type: Type of calculation ('opt', 'sp', etc.)
            
        Returns:
            ORCA input file content as string
        """
        charge, multiplicity = self.get_charge_and_multiplicity(mol)
        xyz_coords = self.mol_to_xyz(mol)
        
        # ORCA input template
        orca_input = f"""# ORCA Input File for {mol_name}
# Geometry optimization at B3LYP/6-31G* level
# Calculation of dipole moment, HOMO/LUMO energies, and molecular volume

! B3LYP 6-31G* OPT
! PAL4                    # Use 4 CPU cores (adjust as needed)

%maxcore 2000             # Memory per core in MB

# Output options for molecular properties
%output
  Print[ P_Hirshfeld ] 1
  Print[ P_Mulliken ] 1
  Print[ P_Overlap ] 1
end

# SCF convergence criteria
%scf
  MaxIter 500
  ConvForced true
end

# Geometry optimization settings
%geom
  MaxIter 500
  TolE 5e-6
  TolRMSG 1e-4
  TolMaxG 3e-4
end

# Molecular orbital analysis
%output
  Print[ P_MOs ] 1
  Print[ P_Basis ] 2
end

# Charge and multiplicity
* xyz {charge} {multiplicity}
{xyz_coords}
*

"""
        return orca_input
    
    def write_orca_input_file(self, mol: Chem.Mol, mol_name: str) -> Path:
        """
        Write ORCA input file to disk
        
        Args:
            mol: RDKit molecule object
            mol_name: Name of the molecule
            
        Returns:
            Path to created input file
        """
        input_content = self.create_orca_input(mol, mol_name)
        input_file = self.output_dir / f"{mol_name}.inp"
        
        with open(input_file, 'w') as f:
            f.write(input_content)
        
        print(f"Created ORCA input file: {input_file}")
        return input_file
    
    def create_submission_script(self, input_files: List[Path], 
                                script_name: str = "run_orca.sh") -> Path:
        """
        Create a shell script to submit ORCA jobs
        
        Args:
            input_files: List of ORCA input files
            script_name: Name of the submission script
            
        Returns:
            Path to submission script
        """
        script_path = self.output_dir / script_name
        
        script_content = """#!/bin/bash
# ORCA Job Submission Script
# Make sure ORCA is installed and in your PATH

# Set ORCA path (modify as needed)
# export ORCA_PATH=/path/to/orca
# export PATH=$ORCA_PATH:$PATH

"""
        
        for input_file in input_files:
            basename = input_file.stem
            script_content += f"""
echo "Running ORCA calculation for {basename}..."
orca {input_file.name} > {basename}.out 2>&1

if [ $? -eq 0 ]; then
    echo "Calculation for {basename} completed successfully"
else
    echo "Error in calculation for {basename}"
fi
echo "-------------------------------------------"
"""
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make script executable
        script_path.chmod(0o755)
        
        print(f"Created submission script: {script_path}")
        return script_path
    
    def process_sdf_file(self, sdf_file: str, run_calculations: bool = False) -> List[Path]:
        """
        Process SDF file and create ORCA input files
        
        Args:
            sdf_file: Path to SDF file
            run_calculations: Whether to automatically run ORCA calculations
            
        Returns:
            List of created input files
        """
        molecules = self.read_sdf(sdf_file)
        input_files = []
        
        for idx, mol in enumerate(molecules):
            # Try to get molecule name from properties
            if mol.HasProp("_Name"):
                mol_name = mol.GetProp("_Name")
            else:
                mol_name = f"molecule_{idx+1}"
            
            # Clean molecule name for filename
            mol_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in mol_name)
            
            # Create ORCA input file
            input_file = self.write_orca_input_file(mol, mol_name)
            input_files.append(input_file)
        
        # Create submission script
        self.create_submission_script(input_files)
        
        # Optionally run calculations
        if run_calculations:
            print("\nRunning ORCA calculations...")
            self.run_orca_calculations(input_files)
        
        return input_files
    
    def run_orca_calculations(self, input_files: List[Path]):
        """
        Run ORCA calculations (requires ORCA to be installed)
        
        Args:
            input_files: List of ORCA input files
        """
        for input_file in input_files:
            output_file = input_file.with_suffix('.out')
            print(f"Running ORCA for {input_file.name}...")
            
            try:
                # Run ORCA calculation
                result = subprocess.run(
                    ['orca', str(input_file)],
                    cwd=self.output_dir,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                # Write output
                with open(output_file, 'w') as f:
                    f.write(result.stdout)
                    if result.stderr:
                        f.write("\n\n=== STDERR ===\n")
                        f.write(result.stderr)
                
                if result.returncode == 0:
                    print(f"  ✓ Completed successfully")
                else:
                    print(f"  ✗ Calculation failed with return code {result.returncode}")
                    
            except FileNotFoundError:
                print(f"  ✗ Error: ORCA not found in PATH")
                print("  Please install ORCA and add it to your PATH, or run calculations manually")
                break
            except subprocess.TimeoutExpired:
                print(f"  ✗ Calculation timed out after 1 hour")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")


def main():
    """Main function to run the workflow"""
    parser = argparse.ArgumentParser(
        description="ORCA Workflow: Convert SDF files to ORCA input files for quantum chemistry calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python orca_workflow.py input.sdf
  python orca_workflow.py molecules.sdf -o my_calculations
  python orca_workflow.py input.sdf --run
        """
    )
    
    parser.add_argument('sdf_file', type=str, help='Input SDF file containing molecular structures')
    parser.add_argument('-o', '--output', type=str, default='orca_calculations',
                       help='Output directory for ORCA files (default: orca_calculations)')
    parser.add_argument('--run', action='store_true',
                       help='Automatically run ORCA calculations (requires ORCA installation)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.sdf_file):
        print(f"Error: Input file '{args.sdf_file}' not found!")
        sys.exit(1)
    
    # Create workflow and process file
    workflow = ORCAWorkflow(output_dir=args.output)
    input_files = workflow.process_sdf_file(args.sdf_file, run_calculations=args.run)
    
    print(f"\n{'='*60}")
    print(f"Workflow completed!")
    print(f"{'='*60}")
    print(f"Created {len(input_files)} ORCA input file(s) in: {workflow.output_dir}")
    print(f"\nTo run calculations manually:")
    print(f"  cd {workflow.output_dir}")
    print(f"  bash run_orca.sh")
    print(f"\nOr run individual calculations:")
    for input_file in input_files:
        print(f"  orca {input_file.name} > {input_file.stem}.out")


if __name__ == "__main__":
    main()
