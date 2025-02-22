import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem

# Configuration parameters
INPUT_DIR = os.getcwd()  # Default input directory
OUTPUT_CSV = "molecular_descriptors.csv"
DOCUMENTATION_FILE = "descriptor_documentation.txt"
MAX_EIGENVALUES = 15  # Number of largest eigenvalues to keep

def generate_descriptor_documentation():
    """Generate documentation file for all RDKit molecular descriptors."""
    descriptor_list = Descriptors._descList
    
    with open(DOCUMENTATION_FILE, "w", encoding="utf-8") as f:
        f.write("RDKit Molecular Descriptor Reference\n")
        f.write("="*40 + "\n\n")
        
        for name, _ in descriptor_list:
            try:
                descriptor_func = getattr(Descriptors, name)
                doc = descriptor_func.__doc__ or "No documentation available"
                cleaned_doc = " ".join(line.strip() for line in doc.split("\n") if line.strip())
                f.write(f"## {name} ##\n{cleaned_doc}\n\n")
            except AttributeError:
                f.write(f"## {name} ##\nDescriptor documentation not found\n\n")

def calculate_coulomb_matrix(mol):
    """
    Compute the Coulomb matrix representation for a molecule.
    
    Args:
        mol (rdkit.Chem.Mol): Input molecule
        
    Returns:
        np.ndarray: Coulomb matrix
    """
    # Ensure 3D coordinates exist
    if not mol.GetConformer().Is3D():
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol)
    
    # Get atomic properties
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    conf = mol.GetConformer()
    coords = np.array([conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())])
    
    # Compute Coulomb matrix
    n_atoms = len(atomic_numbers)
    cm = np.zeros((n_atoms, n_atoms))
    
    for i in range(n_atoms):
        for j in range(n_atoms):
            if i == j:
                cm[i, j] = 0.5 * (atomic_numbers[i] ** 2.4)
            else:
                dist = np.linalg.norm(coords[i] - coords[j])
                cm[i, j] = (atomic_numbers[i] * atomic_numbers[j]) / dist
                
    return cm

def process_molecule(file_path):
    """
    Process a single molecule from SDF file
    
    Args:
        file_path (str): Path to SDF file
        
    Returns:
        dict: Molecular features and metadata
    """
    try:
        # Read molecule
        mol = Chem.MolFromMolFile(file_path)
        if mol is None:
            print(f"Failed to load molecule: {file_path}")
            return None
        
        # Calculate descriptors
        descriptor_results = {}
        for name, func in Descriptors.descList:
            try:
                descriptor_results[name] = func(mol)
            except Exception as e:
                descriptor_results[name] = np.nan
                print(f"Error calculating {name} for {file_path}: {str(e)}")
        
        # Calculate Coulomb matrix features
        try:
            cm = calculate_coulomb_matrix(mol)
            eigvals = np.linalg.eigvalsh(cm)  # Use symmetric eigenvalue decomposition
            eigvals.sort()
            top_eigvals = eigvals[-MAX_EIGENVALUES:]  # Get largest eigenvalues
        except Exception as e:
            print(f"Error calculating Coulomb matrix for {file_path}: {str(e)}")
            top_eigvals = [np.nan] * MAX_EIGENVALUES
        
        # Package results
        result = {
            "file_name": os.path.basename(file_path),
            **descriptor_results,
            **{f"eigen_{i}": val for i, val in enumerate(top_eigvals)}
        }
        
        return result
    
    except Exception as e:
        print(f"Critical error processing {file_path}: {str(e)}")
        return None

def main():
    """Main processing pipeline"""
    # Generate documentation
    generate_descriptor_documentation()
    print(f"Generated descriptor documentation at {DOCUMENTATION_FILE}")
    
    # Get input files
    sdf_files = [
        os.path.join(INPUT_DIR, f) 
        for f in os.listdir(INPUT_DIR) 
        if f.lower().endswith(".sdf")
    ]
    
    if not sdf_files:
        print(f"No SDF files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(sdf_files)} SDF files for processing")
    
    # Parallel processing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(process_molecule, sdf_files)
    
    # Filter failed results
    successful_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(successful_results)}/{len(sdf_files)} molecules")
    
    # Save results
    if successful_results:
        df = pd.DataFrame(successful_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"Results saved to {OUTPUT_CSV}")
    else:
        print("No valid results to save")

if __name__ == "__main__":
    main()
