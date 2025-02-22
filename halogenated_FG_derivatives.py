from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from itertools import combinations, product
import os

def generate_functionalized_benzene(smiles, max_substituents=3, groups=['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']):
    """
    Generate benzene derivatives with specified functional groups.
    :param smiles: SMILES string of the parent molecule
    :param max_substituents: Maximum number of substituents (0 returns the original molecule)
    :param groups: List of functional groups, supports ['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']
    :return: List of RDKit molecules
    """
    # Initialize the molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Handle zero substituents case
    if max_substituents == 0:
        return [Chem.RemoveHs(Chem.AddHs(mol))]

    mol = Chem.AddHs(mol)

    # Validate functional groups
    valid_groups = ['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']
    for g in groups:
        if g not in valid_groups:
            raise ValueError(f"Invalid functional group. Choose from: {valid_groups}")

    # Define templates for functional groups with explicit bonds
    group_templates = {
        '-CN': {
            'atoms': [Chem.Atom('C'), Chem.Atom('N')],
            'bonds': [(0, 1, Chem.BondType.TRIPLE)]
        },
        '-CF3': {
            'atoms': [Chem.Atom('C'), Chem.Atom('F'), Chem.Atom('F'), Chem.Atom('F')],
            'bonds': [(0, 1, Chem.BondType.SINGLE),
                      (0, 2, Chem.BondType.SINGLE),
                      (0, 3, Chem.BondType.SINGLE)]
        },
        '-CH3': {
            'atoms': [Chem.Atom('C'), Chem.Atom('H'), Chem.Atom('H'), Chem.Atom('H')],
            'bonds': [(0, 1, Chem.BondType.SINGLE),
                      (0, 2, Chem.BondType.SINGLE),
                      (0, 3, Chem.BondType.SINGLE)]
        },
        '-OCH3': {
            'atoms': [Chem.Atom('O'), Chem.Atom('C'), Chem.Atom('H'), Chem.Atom('H'), Chem.Atom('H')],
            'bonds': [(0, 1, Chem.BondType.SINGLE),
                      (1, 2, Chem.BondType.SINGLE),
                      (1, 3, Chem.BondType.SINGLE),
                      (1, 4, Chem.BondType.SINGLE)]
        },
        '-COOH': {
            'atoms': [Chem.Atom('C'), Chem.Atom('O'), Chem.Atom('O'), Chem.Atom('H')],
            'bonds': [(0, 1, Chem.BondType.DOUBLE),
                      (0, 2, Chem.BondType.SINGLE),
                      (2, 3, Chem.BondType.SINGLE)]
        },
        '-OH': {
            'atoms': [Chem.Atom('O'), Chem.Atom('H')],
            'bonds': [(0, 1, Chem.BondType.SINGLE)]
        }
    }

    # Identify the benzene ring atoms
    benzene_atoms = []
    for ring in Chem.GetSymmSSSR(mol):
        if len(ring) == 6:  # Benzene ring
            is_aromatic = all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)
            if is_aromatic:
                benzene_atoms = list(ring)
                break

    # Find replaceable hydrogens on benzene carbons
    replaceable_hs = {}
    for atom_idx in benzene_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
            if hs:
                replaceable_hs[atom_idx] = hs[0]  # Take the first replaceable hydrogen

    # Generate all substitution combinations
    generated = []
    for k in range(1, min(max_substituents, len(replaceable_hs)) + 1):
        # All position combinations
        for positions in combinations(replaceable_hs.keys(), k):
            # All functional group combinations (with replacement)
            for group_comb in product(groups, repeat=k):
                emol = Chem.EditableMol(mol)
                # Process in reverse order to avoid index issues
                sorted_pos = sorted([(pos, replaceable_hs[pos]) for pos in positions],
                                   key=lambda x: x[1], reverse=True)

                # Track added groups to handle meta/para configurations
                added_groups = []
                for (carbon_idx, h_idx), group in zip(sorted_pos, group_comb):
                    # Remove the hydrogen atom
                    emol.RemoveAtom(h_idx)

                    # Add the functional group
                    template = group_templates[group]
                    parent_idx = carbon_idx

                    # Add atoms and record their indices
                    new_indices = []
                    for atom in template['atoms']:
                        new_idx = emol.AddAtom(atom)
                        new_indices.append(new_idx)

                    # Connect to parent carbon
                    emol.AddBond(parent_idx, new_indices[0], Chem.BondType.SINGLE)

                    # Create internal bonds for the substituent
                    for bond in template['bonds']:
                        src, dst, bond_type = bond
                        emol.AddBond(new_indices[src], new_indices[dst], bond_type)

                    # Special cases
                    if group == '-COOH':
                        # Set formal charges for carboxyl group
                        carboxy_oxygen = new_indices[2]  # The -OH oxygen
                        emol.GetMol().GetAtomWithIdx(carboxy_oxygen).SetFormalCharge(0)
                        emol.GetMol().GetAtomWithIdx(new_indices[0]).SetFormalCharge(0)

                    added_groups.append((carbon_idx, new_indices))

                # Attempt to sanitize the new molecule
                new_mol = emol.GetMol()
                try:
                    Chem.SanitizeMol(new_mol)
                    # Clean up the molecule
                    new_mol = Chem.RemoveHs(new_mol)
                    # Regularize the SMILES for deduplication
                    canonical_smi = Chem.MolToSmiles(new_mol, kekuleSmiles=True)
                    generated.append((canonical_smi, new_mol))
                except Exception as e:
                    print(f"Error sanitizing molecule: {str(e)}")
                    continue

    # Deduplicate using canonical SMILES
    unique_mols = {}
    for smi, mol in generated:
        if smi not in unique_mols:
            unique_mols[smi] = mol

    return list(unique_mols.values())

def generate_poly_halogen_benzene(smiles, max_halogens=5, halogens=['F', 'Cl', 'Br']):
    """
    Generate halogenated benzene derivatives or return original molecule if max_halogens=0.
    :param smiles: SMILES string of the parent molecule
    :param max_halogens: Maximum number of halogens (0 returns the original molecule)
    :param halogens: List of halogens, supports ['F', 'Cl', 'Br']
    :return: List of RDKit molecules
    """
    # Initialize molecule
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Handle zero halogens case
    if max_halogens == 0:
        return [Chem.RemoveHs(Chem.AddHs(mol))]

    mol = Chem.AddHs(mol)

    # Validate halogens
    valid_halogens = ['F', 'Cl', 'Br']
    for halogen in halogens:
        if halogen not in valid_halogens:
            raise ValueError(f"Invalid halogen. Choose from {valid_halogens}.")

    # Identify benzene ring atoms
    benzene_atoms = []
    for ring in Chem.GetSymmSSSR(mol):
        if len(ring) == 6:
            is_aromatic = True
            for idx in ring:
                if not mol.GetAtomWithIdx(idx).GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic:
                benzene_atoms = list(ring)
                break

    # Find replaceable hydrogens on benzene carbons
    replaceable_hs = {}
    for atom_idx in benzene_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == 'C':
            hs = [n.GetIdx() for n in atom.GetNeighbors() if n.GetSymbol() == 'H']
            if hs:
                replaceable_hs[atom_idx] = hs

    generated = []
    for k in range(1, max_halogens + 1):
        # Generate all combinations of substitution positions
        for carbons in combinations(replaceable_hs.keys(), k):
            # Check all selected carbons have hydrogens
            if any(c not in replaceable_hs for c in carbons):
                continue
            h_indices = [replaceable_hs[c][0] for c in carbons]

            # Generate all possible halogen combinations for these positions
            for halogens_tuple in product(halogens, repeat=k):
                emol = Chem.EditableMol(mol)
                h_sorted = sorted(h_indices, reverse=True)
                for h_idx, halogen in zip(h_sorted, halogens_tuple):
                    parent = mol.GetAtomWithIdx(h_idx).GetNeighbors()[0].GetIdx()
                    emol.RemoveAtom(h_idx)
                    new_idx = emol.AddAtom(Chem.Atom(halogen))
                    emol.AddBond(parent, new_idx, Chem.BondType.SINGLE)

                new_mol = emol.GetMol()
                try:
                    Chem.SanitizeMol(new_mol)
                    new_mol = Chem.RemoveHs(new_mol)
                    generated.append(new_mol)
                except:
                    continue

    # Deduplicate
    unique_smiles = {Chem.MolToSmiles(mol, isomericSmiles=True) for mol in generated}
    return [Chem.MolFromSmiles(smi) for smi in unique_smiles]

def process_base_smiles(base_smiles_dict, output_dir="output", 
                       max_substituents=1, max_halogens=2):
    """Process all base SMILES with configurable substitution levels"""
    os.makedirs(output_dir, exist_ok=True)
    
    for base_name, smiles in base_smiles_dict.items():
        try:
            print(f"\nProcessing {base_name} ({smiles})")
            final_derivatives = []

            # Stage 1: Functionalization
            if max_substituents >= 0:
                derivatives = generate_functionalized_benzene(
                    smiles,
                    max_substituents=max_substituents,
                    groups=['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']
                )
                print(f"Generated {len(derivatives)} primary derivatives")
            else:
                raise ValueError("max_substituents cannot be negative")

            # Stage 2: Halogenation
            if max_halogens >= 0:
                for mol in derivatives:
                    try:
                        halogenated = generate_poly_halogen_benzene(
                            Chem.MolToSmiles(mol), 
                            max_halogens=max_halogens, 
                            halogens=['F', 'Cl', 'Br']
                        )
                        final_derivatives.extend(halogenated)
                    except Exception as e:
                        print(f"Error in halogenation: {str(e)}")
                        continue
                print(f"Total derivatives after halogenation: {len(final_derivatives)}")
            else:
                raise ValueError("max_halogens cannot be negative")

            # Save results
            save_results(base_name, final_derivatives, output_dir)

        except Exception as e:
            print(f"Failed to process {base_name}: {str(e)}")
            continue

    print("\nBatch processing completed. Check output directory for results.")

def save_results(base_name, molecules, output_dir):
    """Save SDF files and images"""
    # Save SDF
    sdf_dir = os.path.join(output_dir, "sdf")
    os.makedirs(sdf_dir, exist_ok=True)
    for idx, mol in enumerate(molecules):
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)
            writer = Chem.SDWriter(os.path.join(sdf_dir, f"{base_name}_{idx+1}.sdf"))
            writer.write(mol)
            writer.close()
        except Exception as e:
            print(f"Error saving SDF {idx+1}: {str(e)}")
    
    # Save image
    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    try:
        img = Draw.MolsToGridImage(
            molecules[:16],
            molsPerRow=4,
            subImgSize=(300, 300),
            legends=[f"{base_name} derivative {i+1}" for i in range(16)],
            returnPNG=False
        )
        img.save(os.path.join(img_dir, f"{base_name}_derivatives.png"))
    except Exception as e:
        print(f"Visualization failed: {str(e)}")

if __name__ == "__main__":
    # Define all base molecules with their SMILES
    base_smiles_dict = {
        "PBA": "C1=CC=C(C=C1)CCCCN",
        "PPA": "C1=CC=C(C=C1)CCCN",
        "PEA": "C1=CC=C(C=C1)CCN",
        "PMA": "C1=CC=C(C=C1)CN",
        "PEA-4-N": "C1=CN=CC=C1CCN",
        "PEA-3-N": "C1=CC(=CN=C1)CCN",
        "PEA-2-N": "C1=CC=NC(=C1)CCN",
        "PPA-4-N": "C1=CN=CC=C1CCCN",
        "PPA-3-N": "C1=CC(=CN=C1)CCCN",
        "PPA-2-N": "C1=CC=NC(=C1)CCCN",
        "PMA-4-N": "C1=CN=CC=C1CN",
        "PMA-3-N": "C1=CC(=CN=C1)CN",
        "PMA-2-N": "C1=CC=NC(=C1)CN",
        "PBA-2-N": "C1=CC=NC(=C1)CCCCN",
        "PBA-3-N": "C1=CC(=CN=C1)CCCCN",
        "PBA-4-N": "C1=CN=CC=C1CCCCN"
    }

    # Process all base molecules
    process_base_smiles(
        base_smiles_dict,
        output_dir="functionalization_halogenation_derivatives_output",
        max_substituents=1,  # Set to 0 to skip functionalization
        max_halogens=2       # Set to 0 to skip halogenation
    )
