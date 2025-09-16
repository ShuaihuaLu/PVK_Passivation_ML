#!/usr/bin/env python3
"""
Molecular generation.py

Generate substituted benzene derivatives by:
  1) Attaching functional groups to benzene carbons (CN, CF3, CH3, OCH3, COOH, OH)
  2) Optionally further halogenating selected positions (F, Cl, Br)

Outputs:
  - SDF files for each derivative (in output/sdf/)
  - Grid image (output/images/<base>_derivatives.png) for up to the first 16 derivatives

Notes:
  - This script uses RDKit. Make sure RDKit is installed in your Python environment.
  - The code works by finding aromatic 6-membered rings (benzene), identifying replaceable
    hydrogens on ring carbons, and performing in-place edits with EditableMol.
  - Some sanitization/embedding steps may fail for certain generated topologies; those
    cases are skipped with an error message.
"""

from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from itertools import combinations, product
import os

def generate_functionalized_benzene(smiles,
                                    max_substituents=3,
                                    groups=['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']):
    """
    Generate benzene derivatives by replacing ring hydrogens with functional groups.

    Returns a list of tuples: (RDKit Mol object, annotation_string)

    :param smiles: SMILES string of the parent molecule
    :param max_substituents: maximum number of substituents to add (0 returns the original molecule)
    :param groups: list of functional groups to use (supported: -CN, -CF3, -CH3, -OCH3, -COOH, -OH)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    # Zero-substituent case: return the original molecule (sanitized, H-stripped)
    if max_substituents == 0:
        base = Chem.RemoveHs(Chem.AddHs(mol))
        base.SetProp("annotation", "none")
        return [(base, "none")]

    mol = Chem.AddHs(mol)  # work with explicit hydrogens for safe atom removals

    supported = ['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']
    for g in groups:
        if g not in supported:
            raise ValueError(f"Unsupported functional group '{g}'. Supported groups: {supported}")

    # Templates for each functional group (atoms + connected bonds between template atoms)
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

    # Find aromatic six-membered rings (benzene-like)
    benzene_rings = []
    for ring in Chem.GetSymmSSSR(mol):
        if len(ring) == 6:
            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
                benzene_rings.append(list(ring))

    # Identify replaceable hydrogens attached to ring carbons
    replaceable_hs = {}
    for ring in benzene_rings:
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == 'C':
                hs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
                if hs:
                    replaceable_hs.setdefault(atom_idx, []).extend(hs)

    # For simplicity, pick the first hydrogen for each carbon as the replacement target
    replaceable_hs = {k: v[0] for k, v in replaceable_hs.items()}

    generated = []
    max_k = min(max_substituents, len(replaceable_hs))
    for k in range(1, max_k + 1):
        # combinations of ring positions
        for positions in combinations(replaceable_hs.keys(), k):
            # all combinations of functional groups (with repetition allowed)
            for group_comb in product(groups, repeat=k):
                emol = Chem.EditableMol(mol)
                # Sorted by hydrogen index descending to avoid index shifts when removing atoms
                sorted_pos = sorted([(pos, replaceable_hs[pos]) for pos in positions],
                                    key=lambda x: x[1], reverse=True)
                annotation_parts = []
                try:
                    for (carbon_idx, h_idx), group in zip(sorted_pos, group_comb):
                        # Remove the hydrogen atom to make space for the substituent
                        emol.RemoveAtom(h_idx)
                        template = group_templates[group]
                        new_indices = []
                        # Add template atoms
                        for atom in template['atoms']:
                            new_idx = emol.AddAtom(atom)
                            new_indices.append(new_idx)
                        # Attach template's first atom to the carbon (single bond by default)
                        emol.AddBond(carbon_idx, new_indices[0], Chem.BondType.SINGLE)
                        # Add internal bonds inside the template
                        for src, dst, btype in template['bonds']:
                            emol.AddBond(new_indices[src], new_indices[dst], btype)
                        # Special handling for some templates if needed (no charges set here)
                        annotation_parts.append(group.replace('-', ''))  # remove '-' for nicer annotation
                except Exception as exc:
                    # If any edit fails, skip this combination
                    # Print debug information and continue
                    print(f"Editing failed for positions {positions} with groups {group_comb}: {exc}")
                    continue

                new_mol = emol.GetMol()
                try:
                    Chem.SanitizeMol(new_mol)
                    new_mol = Chem.RemoveHs(new_mol)
                    canonical = Chem.MolToSmiles(new_mol, isomericSmiles=True)
                    annot = "FG_" + "_".join(annotation_parts) if annotation_parts else "none"
                    new_mol.SetProp("annotation", annot)
                    generated.append((new_mol, annot))
                except Exception as exc:
                    print(f"Sanitization failed for generated molecule: {exc}")
                    continue

    # Deduplicate by canonical SMILES
    unique = {}
    for mol_obj, annot in generated:
        try:
            smi = Chem.MolToSmiles(mol_obj, isomericSmiles=True)
            if smi not in unique:
                unique[smi] = (mol_obj, annot)
        except Exception:
            continue

    return list(unique.values())  # [(mol, annot), ...]

def generate_poly_halogen_benzene(smiles,
                                  max_halogens=5,
                                  halogens=['F', 'Cl', 'Br'],
                                  parent_annotation=""):
    """
    Add halogens to benzene ring hydrogens.

    Returns a list of tuples: (RDKit Mol object, annotation_string)

    :param smiles: SMILES string of the parent molecule (can be an already functionalized mol SMILES)
    :param max_halogens: maximum number of halogen substituents (0 returns the input molecule)
    :param halogens: list of halogen elements to use (supported: 'F', 'Cl', 'Br')
    :param parent_annotation: annotation carried from previous functionalization (used in result annotation)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string provided.")

    if max_halogens == 0:
        base = Chem.RemoveHs(Chem.AddHs(mol))
        annot = parent_annotation if parent_annotation else "none"
        base.SetProp("annotation", annot)
        return [(base, annot)]

    mol = Chem.AddHs(mol)

    supported = ['F', 'Cl', 'Br']
    for h in halogens:
        if h not in supported:
            raise ValueError(f"Unsupported halogen '{h}'. Supported: {supported}")

    # Find aromatic 6-member rings
    benzene_rings = []
    for ring in Chem.GetSymmSSSR(mol):
        if len(ring) == 6 and all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring):
            benzene_rings.append(list(ring))

    replaceable_hs = {}
    for ring in benzene_rings:
        for atom_idx in ring:
            atom = mol.GetAtomWithIdx(atom_idx)
            if atom.GetSymbol() == 'C':
                hs = [nbr.GetIdx() for nbr in atom.GetNeighbors() if nbr.GetSymbol() == 'H']
                if hs:
                    replaceable_hs.setdefault(atom_idx, []).extend(hs)

    generated = []
    max_k = min(max_halogens, len(replaceable_hs))
    for k in range(1, max_k + 1):
        for carbons in combinations(replaceable_hs.keys(), k):
            # ensure all carbons have at least one replaceable H
            if any(c not in replaceable_hs for c in carbons):
                continue
            # pick the first H for each carbon (conservative)
            h_indices = [replaceable_hs[c][0] for c in carbons]
            for halogen_tuple in product(halogens, repeat=k):
                emol = Chem.EditableMol(mol)
                # remove hydrogens in reverse-sorted order to avoid reindexing issues
                for h_idx, hal in sorted(zip(h_indices, halogen_tuple), key=lambda x: x[0], reverse=True):
                    # find parent carbon for this hydrogen in the original mol (before edits)
                    parent_atom = mol.GetAtomWithIdx(h_idx).GetNeighbors()[0].GetIdx()
                    emol.RemoveAtom(h_idx)
                    new_idx = emol.AddAtom(Chem.Atom(hal))
                    emol.AddBond(parent_atom, new_idx, Chem.BondType.SINGLE)
                new_mol = emol.GetMol()
                try:
                    Chem.SanitizeMol(new_mol)
                    new_mol = Chem.RemoveHs(new_mol)
                    hal_annot = "Hal_" + "_".join(halogen_tuple)
                    combined_annot = (parent_annotation + "_" + hal_annot) if parent_annotation and parent_annotation != "none" else hal_annot
                    new_mol.SetProp("annotation", combined_annot)
                    generated.append((new_mol, combined_annot))
                except Exception as exc:
                    print(f"Halogenation sanitization failed: {exc}")
                    continue

    # Deduplicate by canonical SMILES
    unique = {}
    for mol_obj, annot in generated:
        try:
            smi = Chem.MolToSmiles(mol_obj, isomericSmiles=True)
            if smi not in unique:
                unique[smi] = (mol_obj, annot)
        except Exception:
            continue

    return list(unique.values())

def save_results(base_name, molecules_with_annotation, output_dir):
    """
    Save generated molecules:
      - SDF files (output/sdf/)
      - A grid image (output/images/) with up to 16 molecules and annotated legends

    :param base_name: base name for files
    :param molecules_with_annotation: list of (mol, annotation) tuples
    :param output_dir: top-level output directory
    """
    sdf_dir = os.path.join(output_dir, "sdf")
    os.makedirs(sdf_dir, exist_ok=True)

    for idx, (mol, annot) in enumerate(molecules_with_annotation):
        try:
            mol_with_H = Chem.AddHs(mol)
            # Generate 3D coordinates and do a quick MMFF optimization (best-effort)
            AllChem.EmbedMolecule(mol_with_H, randomSeed=42)
            try:
                AllChem.MMFFOptimizeMolecule(mol_with_H)
            except Exception:
                # If MMFF fails, continue with embedded geometry
                pass
            fname = f"{base_name}_{idx+1}_{annot}.sdf"
            writer = Chem.SDWriter(os.path.join(sdf_dir, fname))
            writer.write(mol_with_H)
            writer.close()
        except Exception as exc:
            print(f"Error writing SDF for {base_name} index {idx+1}: {exc}")

    img_dir = os.path.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    try:
        # Create legends for up to 16 molecules
        mols_for_grid = [mol for mol, _ in molecules_with_annotation[:16]]
        legends = [f"{base_name} d.{i+1} [{annot}]" for i, (_, annot) in enumerate(molecules_with_annotation[:16])]
        if mols_for_grid:
            grid = Draw.MolsToGridImage(
                mols_for_grid,
                molsPerRow=4,
                subImgSize=(300, 300),
                legends=legends,
                returnPNG=False
            )
            grid.save(os.path.join(img_dir, f"{base_name}_derivatives.png"))
    except Exception as exc:
        print(f"Visualization failed for {base_name}: {exc}")

def process_base_smiles(base_smiles_dict,
                        output_dir="output",
                        max_substituents=1,
                        max_halogens=2):
    """
    Process a dictionary of base molecules (name -> SMILES), perform functionalization and halogenation,
    and save results.

    :param base_smiles_dict: dict of {name: smiles}
    :param output_dir: top-level directory to save results
    :param max_substituents: max number of functional groups to add in stage 1
    :param max_halogens: max number of halogens to add in stage 2
    """
    os.makedirs(output_dir, exist_ok=True)

    for base_name, smiles in base_smiles_dict.items():
        print(f"\nProcessing {base_name}  SMILES: {smiles}")
        try:
            # Stage 1: functionalization
            func_derivatives = generate_functionalized_benzene(
                smiles,
                max_substituents=max_substituents,
                groups=['-CN', '-CF3', '-CH3', '-OCH3', '-COOH', '-OH']
            )
            print(f"  Functionalization generated: {len(func_derivatives)} derivatives")

            # Stage 2: halogenation (applied to each functionalized derivative)
            final_derivatives = []
            for mol_obj, annot in func_derivatives:
                try:
                    halogenated = generate_poly_halogen_benzene(
                        Chem.MolToSmiles(mol_obj),
                        max_halogens=max_halogens,
                        halogens=['F', 'Cl', 'Br'],
                        parent_annotation=annot
                    )
                    final_derivatives.extend(halogenated)
                except Exception as exc:
                    print(f"  Halogenation error for {base_name} derivative {annot}: {exc}")
                    continue

            print(f"  Total derivatives after halogenation: {len(final_derivatives)}")

            # Save SDFs and images for this base molecule
            save_results(base_name, final_derivatives, output_dir)
            print(f"  Saved outputs for {base_name} to {output_dir}")

        except Exception as exc:
            print(f"Failed to process {base_name}: {exc}")
            continue

    print("\nBatch processing completed. Check the output directory for results.")

if __name__ == "__main__":
    # Define base molecules with SMILES strings
    base_smiles_dict = {
        "PEA-C6-2": "C1=CC=C2C=C(C=CC2=C1)CCN",
        "PPA-C6-2": "C1=CC=C2C=C(C=CC2=C1)CCCN",
        "PMA-C6-2": "C1=CC=C2C=C(C=CC2=C1)CN",
        "PBA-C6-2": "C1=CC=C2C=C(C=CC2=C1)CCCCN",
        "PEA-C6-1": "C1=CC=C2C(=C1)C=CC=C2CCN",
        "PPA-C6-1": "C1=CC=C2C(=C1)C=CC=C2CCCN",
        "PMA-C6-1": "C1=CC=C2C(=C1)C=CC=C2CN",
        "PBA-C6-1": "C1=CC=C2C(=C1)C=CC=C2CCCCN",
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
        "PBA-4-N": "C1=CN=CC=C1CCCCN",
    }

    # Run processing: change max_substituents and max_halogens as needed
    process_base_smiles(
        base_smiles_dict,
        output_dir="output",
        max_substituents=1,  # set to 0 to skip functionalization
        max_halogens=1       # set to 0 to skip halogenation
    )
