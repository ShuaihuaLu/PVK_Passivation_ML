import re
import math
import csv
import argparse
import os

def parse_fchk(fchk_file):
    """
    Extract key electronic structure information from a Gaussian fchk file.

    Args:
        fchk_file: Path to the Gaussian formatted checkpoint file.

    Returns:
        dict: A dictionary containing HOMO, LUMO, Gap, and Dipole Moment.
    """
    results = {}

    # Conversion factor from atomic units to Debye
    AU_TO_DEBYE = 2.541746

    with open(fchk_file, 'r') as f:
        lines = f.readlines()

    # Parse the total number of electrons
    n_electrons = None
    for line in lines:
        if "Number of electrons" in line:
            n_electrons = int(line.split()[-1])
            break

    if n_electrons is None:
        raise ValueError("Total number of electrons not found")

    # Parse Alpha orbital energies
    alpha_energies = []
    in_alpha = False
    for line in lines:
        if "Alpha Orbital Energies" in line:
            in_alpha = True
            n_orbitals = int(re.search(r'N=\s*(\d+)', line).group(1))
            continue
        if in_alpha:
            alpha_energies.extend(map(float, line.split()))
            if len(alpha_energies) >= n_orbitals:
                break

    if not alpha_energies:
        raise ValueError("Alpha orbital energies not found")

    # Calculate HOMO and LUMO
    homo_index = (n_electrons // 2) - 1  # Assumes closed-shell system
    try:
        homo = alpha_energies[homo_index]
        lumo = alpha_energies[homo_index + 1]
    except IndexError:
        raise ValueError("Incomplete orbital energy data")

    results['HOMO (eV)'] = homo
    results['LUMO (eV)'] = lumo
    results['HOMO-LUMO Gap (eV)'] = lumo - homo

    # Parse Dipole Moment
    dipole = []
    in_dipole = False
    for line in lines:
        if "Dipole Moment" in line and not line.strip().endswith("[Input]"):
            in_dipole = True
            continue
        if in_dipole:
            dipole = list(map(float, line.split()))
            break

    if len(dipole) != 3:
        raise ValueError("Incomplete dipole moment data")

    # Calculate total dipole moment (convert to Debye)
    dipole_moment = math.sqrt(sum(x**2 for x in dipole))
    results['Dipole Moment (Debye)'] = dipole_moment * AU_TO_DEBYE

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Gaussian fchk files and save results to CSV.")
    parser.add_argument("files", nargs="+", help="Input .fchk files")
    parser.add_argument("-o", "--output", default="results.csv", help="Output CSV filename")
    args = parser.parse_args()

    # Prepare CSV data
    csv_data = []
    fieldnames = ["Filename", "HOMO (eV)", "LUMO (eV)", "HOMO-LUMO Gap (eV)", "Dipole Moment (Debye)"]

    for fchk_file in args.files:
        try:
            data = parse_fchk(fchk_file)
            # 提取不带路径和后缀的文件名
            filename = os.path.basename(fchk_file)          # 从路径中提取文件名（如 "dir/file.fchk" → "file.fchk"）
            basename = os.path.splitext(filename)[0]        # 去除.fchk后缀（"file.fchk" → "file"）
            
            csv_data.append({
                "Filename": basename,
                "HOMO (eV)": data["HOMO (eV)"],
                "LUMO (eV)": data["LUMO (eV)"],
                "HOMO-LUMO Gap (eV)": data["HOMO-LUMO Gap (eV)"],
                "Dipole Moment (Debye)": data["Dipole Moment (Debye)"]
            })
        except Exception as e:
            print(f"Error processing {fchk_file}: {str(e)}")

    # Write to CSV
    with open(args.output, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)

    print(f"Processed {len(csv_data)} files. Results saved to {args.output}")
