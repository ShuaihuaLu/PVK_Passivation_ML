# Computational Chemistry Workflows

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![ORCA](https://img.shields.io/badge/ORCA-compatible-orange.svg)](https://orcaforum.kofo.mpg.de/)
[![VASP](https://img.shields.io/badge/VASP-5.4%2B-red.svg)](https://www.vasp.at/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Automated Python workflows for quantum chemistry and materials calculations. This repository provides complete automation for **ORCA** (molecular calculations) and **VASP** (materials/solid-state calculations), from input file generation to results analysis.

## 🎯 Overview

This package contains two independent but complementary workflows:

### 🧪 ORCA Workflow - Molecular Quantum Chemistry
Automated workflow for small molecule calculations:
- Convert SDF files to ORCA input files
- Geometry optimization at B3LYP/6-31G* level
- Calculate molecular properties (dipole moment, HOMO/LUMO, energy gap)
- Extract and export results to CSV/JSON

### 🔬 VASP Workflow - Materials Calculations
Automated workflow for solid-state materials:
- Read POSCAR/CONTCAR structure files
- Structure relaxation with DFT-D3 vdW correction
- Electronic structure calculations (DOS and band structure)
- HPC job submission (PBS/SLURM)
- Comprehensive results parsing

---

## 📋 Table of Contents

- [Features Comparison](#-features-comparison)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
  - [ORCA Workflow](#orca-workflow)
  - [VASP Workflow](#vasp-workflow)
- [ORCA Documentation](#-orca-molecular-calculations)
- [VASP Documentation](#-vasp-materials-calculations)
- [Output Files](#-output-files)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## ✨ Features Comparison

| Feature | ORCA Workflow | VASP Workflow |
|---------|---------------|---------------|
| **Input Format** | SDF files | POSCAR/CONTCAR |
| **System Type** | Molecules | Crystals/Materials |
| **Theory Level** | B3LYP/6-31G* | PBE + DFT-D3 |
| **Calculations** | Opt + Properties | Relax + DOS + Bands |
| **Output** | HOMO/LUMO, Dipole | Band gap, DOS, Forces |
| **HPC Integration** | Optional | Built-in (PBS/SLURM) |
| **Batch Processing** | ✅ Multiple molecules | ✅ Multiple materials |
| **Result Export** | CSV, JSON | CSV, JSON |

---

## 🚀 Installation

### Prerequisites

**For ORCA Workflow:**
- Python 3.7+
- [RDKit](https://www.rdkit.org/) - Chemical informatics
- NumPy
- ORCA (optional, for running calculations)

**For VASP Workflow:**
- Python 3.7+
- NumPy
- VASP 5.4+ (with valid license)
- VASP pseudopotentials (POTCAR files)

### Install Dependencies

```bash
# Clone repository
git clone https://github.com/yourusername/computational-chemistry-workflows.git
cd computational-chemistry-workflows

# For ORCA workflow
conda install -c conda-forge rdkit numpy

# For VASP workflow
pip install numpy

# Or install all dependencies
pip install -r requirements.txt  # ORCA
pip install -r requirements_vasp.txt  # VASP
```

### Setup Software Environments

**ORCA:**
```bash
# Add ORCA to PATH
export PATH=/path/to/orca:$PATH
```

**VASP:**
```bash
# Set VASP executable
export VASP_BIN=/path/to/vasp/bin/vasp_std

# Set pseudopotential path
export VASP_PP_PATH=/path/to/vasp/potentials/potpaw_PBE
```

---

## 🎬 Quick Start

### ORCA Workflow

**Generate ORCA Input Files:**
```bash
# Basic usage
python orca_workflow.py molecules.sdf -n my_molecules

# With custom output directory
python orca_workflow.py molecules.sdf -o orca_calc
```

**Run Calculations:**
```bash
cd orca_calculations
bash run_orca.sh
```

**Parse Results:**
```bash
python parse_orca_output.py -d orca_calculations --csv --json
```

**Complete Example:**
```bash
# 1. Generate inputs
python orca_workflow.py example_molecules.sdf

# 2. Run ORCA (if installed)
cd orca_calculations
orca molecule_1.inp > molecule_1.out

# 3. Parse results
cd ..
python parse_orca_output.py -d orca_calculations --csv
```

### VASP Workflow

**Generate VASP Input Files:**
```bash
# Basic usage
python vasp_workflow.py structure.vasp -n material_name

# With custom k-points
python vasp_workflow.py POSCAR -n graphene \
  --relax-kgrid 12 12 1 \
  --dos-kgrid 24 24 1 \
  --system hexagonal
```

**Submit Jobs:**
```bash
# Add POTCAR files first
cd material_name_relax
cat $VASP_PP_PATH/C/POTCAR > POTCAR

# Submit to cluster
qsub submit.pbs  # PBS
sbatch submit.slurm  # SLURM
```

**Parse Results:**
```bash
python parse_vasp_output.py -d vasp_calculations --csv --json
```

**Complete Example:**
```bash
# 1. Setup workflow
python vasp_workflow.py example_graphene.vasp -n graphene

# 2. Add POTCAR to each directory
cd graphene_relax && cat $VASP_PP_PATH/C/POTCAR > POTCAR
cd ../graphene_dos && cat $VASP_PP_PATH/C/POTCAR > POTCAR
cd ../graphene_band && cat $VASP_PP_PATH/C/POTCAR > POTCAR

# 3. Submit jobs sequentially
cd graphene_relax && qsub submit.pbs
# Wait for completion...
cd ../graphene_dos && qsub submit.pbs
# Wait for completion...
cd ../graphene_band && qsub submit.pbs

# 4. Analyze results
cd ../..
python parse_vasp_output.py -d vasp_calculations --csv --json
```

---

## 🧪 ORCA Molecular Calculations

### Calculation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Functional** | B3LYP | Hybrid DFT functional |
| **Basis Set** | 6-31G* | Polarization on heavy atoms |
| **Optimization** | Quasi-Newton | Geometry optimization |
| **Convergence** | EDIFF=5e-6 | Energy convergence |

### Calculated Properties

- ✅ **Optimized Geometry** - Equilibrium structure
- ✅ **Dipole Moment** - Molecular polarity (Debye)
- ✅ **HOMO Energy** - Highest occupied orbital (eV)
- ✅ **LUMO Energy** - Lowest unoccupied orbital (eV)
- ✅ **HOMO-LUMO Gap** - Electronic energy gap (eV)
- ✅ **Total Energy** - Final SCF energy (Hartree)

### Usage

```bash
# Generate input files
python orca_workflow.py input.sdf -n job_name

# Specify output directory
python orca_workflow.py molecules.sdf -o my_calculations

# Auto-run calculations (requires ORCA)
python orca_workflow.py input.sdf --run

# Parse single output
python parse_orca_output.py molecule.out

# Parse all outputs
python parse_orca_output.py -d orca_calculations --csv --json
```

### ORCA Input File Template

```bash
! B3LYP 6-31G* OPT
! PAL4                    # Use 4 CPU cores

%maxcore 2000             # Memory per core (MB)

%scf
  MaxIter 500
  ConvForced true
end

%geom
  MaxIter 500
  TolE 5e-6
  TolRMSG 1e-4
  TolMaxG 3e-4
end

* xyz charge multiplicity
...coordinates...
*
```

### Customization

**Change theory level:**
```bash
! B3LYP 6-311++G**         # Larger basis set
```

**Add dispersion correction:**
```bash
! D3BJ                     # DFT-D3 with BJ damping
```

**Add solvent effects:**
```bash
! CPCM(Water)             # Water solvation
```

### ORCA Output Files

```
orca_calculations/
├── molecule_1.inp         # Input file
├── molecule_1.out         # Output file
├── molecule_1.xyz         # Optimized structure
├── molecule_1.gbw         # Wavefunction
├── run_orca.sh           # Batch script
├── orca_results.csv      # Summary table
└── orca_results.json     # Detailed results
```

---

## 🔬 VASP Materials Calculations

### Calculation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Functional** | PBE | Perdew-Burke-Ernzerhof |
| **Cutoff** | 520 eV | Plane-wave cutoff |
| **vdW Correction** | IVDW=11 | DFT-D3 method |
| **K-points** | Gamma-centered | Automatic mesh |
| **EDIFF** | 1E-05 eV | SCF convergence |
| **EDIFFG** | -0.01 eV/Å | Force convergence |
| **ISMEAR** | 0 | Gaussian smearing |
| **SIGMA** | 0.05 eV | Smearing width |
| **POTIM** | 0.2 fs | Ionic time step |
| **ISIF** | 2 | Relax ions only |

### Workflow Steps

**Step 1: Structure Relaxation**
- Optimize atomic positions (ISIF=2)
- Conjugate gradient algorithm
- Converge forces < 0.01 eV/Å

**Step 2: Density of States**
- Static calculation on relaxed structure
- Tetrahedron method (ISMEAR=-5)
- Dense k-point mesh for accurate DOS

**Step 3: Band Structure**
- Non-SCF calculation
- High-symmetry k-path
- Extract band dispersion

### Usage

```bash
# Basic setup
python vasp_workflow.py structure.vasp -n material

# Custom k-points
python vasp_workflow.py POSCAR -n mat \
  --relax-kgrid 6 6 6 \
  --dos-kgrid 12 12 12

# For 2D materials
python vasp_workflow.py graphene.vasp -n graphene \
  --relax-kgrid 12 12 1 \
  --dos-kgrid 24 24 1 \
  --system hexagonal

# SLURM cluster
python vasp_workflow.py POSCAR -n material --scheduler slurm

# Parse results
python parse_vasp_output.py -c material_relax
python parse_vasp_output.py -d vasp_calculations --csv --json
```

### VASP INCAR Template

```bash
# Structure Relaxation
PREC     = Accurate
ENCUT    = 520
EDIFF    = 1E-05
EDIFFG   = -0.01
IBRION   = 2              # CG algorithm
ISIF     = 2              # Relax ions only
NSW      = 200
ISMEAR   = 0
SIGMA    = 0.05
POTIM    = 0.2

# DFT functional
GGA      = PE             # PBE

# vdW correction
IVDW     = 11             # DFT-D3
LVDW     = .TRUE.

# Performance
LREAL    = Auto
NCORE    = 4
```

### K-point Selection Guide

| Material Type | Relaxation | DOS | Example |
|---------------|------------|-----|---------|
| **3D Bulk** | 4×4×4 | 8×8×8 | Diamond, NaCl |
| **2D Monolayer** | 12×12×1 | 24×24×1 | Graphene, MoS₂ |
| **1D Nanowire** | 1×1×12 | 1×1×24 | Carbon nanotube |
| **Surface Slab** | 8×8×1 | 16×16×1 | Metal surfaces |

### VASP Output Files

```
vasp_calculations/
├── material_relax/
│   ├── INCAR              # Parameters
│   ├── KPOINTS            # K-mesh
│   ├── POSCAR             # Input structure
│   ├── POTCAR             # Pseudopotentials
│   ├── submit.pbs         # Job script
│   ├── OUTCAR             # Main output
│   ├── CONTCAR            # Relaxed structure
│   └── OSZICAR            # Energy convergence
├── material_dos/
│   └── DOSCAR             # DOS data
├── material_band/
│   └── EIGENVAL           # Band energies
├── vasp_summary.csv       # Summary table
└── vasp_results.json      # Detailed results
```

---

## 📊 Output Files

### CSV Output Format

**ORCA (orca_results.csv):**
| Column | Description | Unit |
|--------|-------------|------|
| molecule_name | Identifier | - |
| final_energy | Total energy | Hartree |
| dipole_magnitude_debye | Dipole moment | Debye |
| HOMO_eV | HOMO energy | eV |
| LUMO_eV | LUMO energy | eV |
| gap_eV | HOMO-LUMO gap | eV |

**VASP (vasp_summary.csv):**
| Column | Description | Unit |
|--------|-------------|------|
| calculation_name | Identifier | - |
| converged | Status | boolean |
| final_energy | Total energy | eV |
| energy_per_atom | Energy/atom | eV |
| max_force | Max force | eV/Å |
| fermi_energy | Fermi level | eV |
| band_gap | Band gap | eV |
| volume | Cell volume | ų |

### JSON Output

Both workflows export detailed JSON files containing:
- Complete calculation parameters
- Convergence history
- All molecular/structural properties
- Orbital/band information
- Force and stress tensors (VASP)

---

## 📁 Project Structure

```
computational-chemistry-workflows/
├── orca_workflow.py           # ORCA main script
├── parse_orca_output.py       # ORCA parser
├── vasp_workflow.py           # VASP main script
├── parse_vasp_output.py       # VASP parser
├── test_workflow.py           # ORCA test script
├── requirements.txt           # ORCA dependencies
├── requirements_vasp.txt      # VASP dependencies
├── README.md                  # This file
├── QUICKSTART.md             # ORCA quick start
├── QUICKSTART_VASP.md        # VASP quick start
├── PROJECT_OVERVIEW.md       # ORCA overview
├── PROJECT_OVERVIEW_VASP.md  # VASP overview
├── example_molecules.sdf     # ORCA example
├── example_benzene.inp       # ORCA input example
├── example_benzene.out       # ORCA output example
└── example_graphene.vasp     # VASP example
```

---

## 🎨 Customization

### ORCA Customization

**Modify calculation parameters:**
```python
# In orca_workflow.py
def create_orca_input(...):
    orca_input = f"""
! B3LYP 6-311G** OPT        # Change basis set
! D3BJ                      # Add dispersion
! CPCM(Water)              # Add solvent
...
```

**Change CPU cores:**
```bash
! PAL8  # Use 8 cores instead of 4
```

### VASP Customization

**Modify parameters:**
```python
# In vasp_workflow.py
self.calc_params = {
    'ENCUT': 600,           # Higher cutoff
    'EDIFF': 1E-06,         # Tighter convergence
    'ISMEAR': -5,           # Tetrahedron method
}
```

**Add magnetic calculations:**
```bash
# Edit generated INCAR
ISPIN    = 2              # Spin-polarized
MAGMOM   = 2*5 4*0        # Initial moments
```

---

## ⚡ Performance Tips

### ORCA

1. **Memory allocation:**
   ```
   %maxcore 4000  # 4GB per core
   ```

2. **Parallel efficiency:**
   ```
   ! PAL8  # Use 8 cores for medium molecules
   ```

3. **SCF convergence:**
   ```
   %scf
     MaxIter 1000
     DIISMaxEq 10
   end
   ```

### VASP

1. **K-point convergence:**
   - Start with coarse grid (2×2×2)
   - Gradually increase to convergence
   - Use denser grid for DOS/bands

2. **Memory optimization:**
   ```
   NCORE = 8      # More cores per band
   LREAL = Auto   # Real space projection
   ```

3. **Parallel efficiency:**
   ```
   NCORE = sqrt(total_cores)  # Optimal setting
   Example: 24 cores → NCORE = 4-6
   ```

---

## 🐛 Troubleshooting

### Common ORCA Issues

**Problem: RDKit not found**
```bash
conda install -c conda-forge rdkit
```

**Problem: ORCA not found**
```bash
export PATH=/path/to/orca:$PATH
```

**Problem: SCF not converging**
```bash
# Increase iterations in INCAR
%scf MaxIter 1000 end
```

### Common VASP Issues

**Problem: POTCAR missing**
```bash
# Follow POTCAR_INFO.txt instructions
cat $VASP_PP_PATH/Element/POTCAR > POTCAR
```

**Problem: SCF not converging**
```bash
# Edit INCAR
ALGO = All
NELM = 200
```

**Problem: Out of memory**
```bash
# Edit INCAR
NCORE = 8      # Increase NCORE
LREAL = Auto   # Use real space
```

**Problem: Wrong k-points for 2D**
```bash
# Use: 12×12×1 instead of 4×4×4
python vasp_workflow.py POSCAR -n mat --relax-kgrid 12 12 1
```

---

## 📈 Computational Cost Estimates

### ORCA

| Molecule Size | Atoms | Time (4 cores) |
|---------------|-------|----------------|
| Small | < 30 | Minutes |
| Medium | 30-80 | Hours |
| Large | > 80 | Days |

### VASP

| System | Atoms | K-points | Relax | DOS | Band |
|--------|-------|----------|-------|-----|------|
| Small | < 10 | 4³ | 0.5-2h | 0.5-1h | 0.2-0.5h |
| Medium | 10-50 | 4³ | 2-12h | 1-4h | 0.5-2h |
| Large | 50-100 | 2³ | 12-48h | 4-12h | 1-4h |
| Very Large | > 100 | 2³ | 2-7d | 12-24h | 2-8h |

*Times vary with hardware and settings*

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

**Note:** 
- ORCA requires a separate license from the ORCA developers
- VASP requires a valid license from the VASP developers

---

## 📚 References

### ORCA
1. Neese, F. *"The ORCA program system."* WIREs Comput. Mol. Sci. **2012**, *2*, 73-78.
2. Becke, A. D. *"Density-functional thermochemistry. III."* J. Chem. Phys. **1993**, *98*, 5648.

### VASP
1. Kresse, G.; Furthmüller, J. *"Efficient iterative schemes for ab initio total-energy calculations."* Phys. Rev. B **1996**, *54*, 11169.
2. Perdew, J. P.; Burke, K.; Ernzerhof, M. *"Generalized gradient approximation made simple."* Phys. Rev. Lett. **1996**, *77*, 3865.
3. Grimme, S.; et al. *"A consistent and accurate ab initio parametrization of DFT-D."* J. Chem. Phys. **2010**, *132*, 154104.

---

## 📊 Citation

If you use these workflows in your research, please cite:

```bibtex
@software{computational_chemistry_workflows,
  author = {Your Name},
  title = {Computational Chemistry Workflows: ORCA and VASP Automation},
  year = {2026},
  url = {https://github.com/yourusername/computational-chemistry-workflows}
}
```

Also cite the respective software packages (ORCA and/or VASP) as appropriate.

---

## 📞 Support

- **Documentation**: See individual QUICKSTART guides
- **Issues**: Open an issue on GitHub
- **ORCA Manual**: https://orcaforum.kofo.mpg.de/
- **VASP Wiki**: https://www.vasp.at/wiki/

---

## 🙏 Acknowledgments

- ORCA developers at Max Planck Institute
- VASP development team at University of Vienna
- RDKit community
- Computational chemistry and materials science communities
- All contributors to this project

---

<div align="center">

**[ORCA Quick Start](QUICKSTART.md)** • **[VASP Quick Start](QUICKSTART_VASP.md)** • **[Examples](examples/)** • **[Issues](https://github.com/yourusername/workflows/issues)**

Made with ❤️ for the computational science community

**Version 1.0.0** | **Released 2026-01-27**

</div>
