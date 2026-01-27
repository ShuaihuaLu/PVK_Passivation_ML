#!/usr/bin/env python3
"""
VASP Materials Calculation Workflow
====================================
Automated workflow for VASP calculations including:
1. Structure relaxation (ISIF=2, ion positions only)
2. Electronic structure calculation (DOS and band structure)
3. Job submission to CPU compute nodes

Calculation Parameters:
- Functional: PBE
- Cutoff: 520 eV
- vdW correction: IVDW=11 (DFT-D3 method)
- K-points: Gamma-centered
- EDIFF: 1E-05 eV
- EDIFFG: -0.01 eV/Å
- ISMEAR: 0 (Gaussian smearing)
- SIGMA: 0.05 eV
- POTIM: 0.2 fs
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import json
import numpy as np
from datetime import datetime


class VASPWorkflow:
    """VASP workflow manager for materials calculations"""
    
    def __init__(self, working_dir: str = "vasp_calculations"):
        """
        Initialize VASP workflow
        
        Args:
            working_dir: Root directory for all VASP calculations
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(exist_ok=True)
        
        # Calculation-specific parameters
        self.calc_params = {
            'PREC': 'Accurate',
            'ENCUT': 520,           # Cutoff energy (eV)
            'EDIFF': 1E-05,         # Electronic convergence (eV)
            'EDIFFG': -0.01,        # Ionic convergence (eV/Å)
            'ISMEAR': 0,            # Gaussian smearing
            'SIGMA': 0.05,          # Smearing width (eV)
            'POTIM': 0.2,           # Time step for ionic motion (fs)
            'IVDW': 11,             # DFT-D3 vdW correction
            'LREAL': 'Auto',        # Real space projection
            'ALGO': 'Normal',       # Electronic minimization algorithm
            'NELM': 100,            # Max electronic steps
        }
        
    def read_poscar(self, poscar_file: str) -> Dict:
        """
        Read VASP POSCAR/CONTCAR file
        
        Args:
            poscar_file: Path to POSCAR or CONTCAR file
            
        Returns:
            Dictionary containing structure information
        """
        print(f"Reading structure from: {poscar_file}")
        
        with open(poscar_file, 'r') as f:
            lines = f.readlines()
        
        structure = {
            'comment': lines[0].strip(),
            'scale': float(lines[1].strip()),
            'lattice': [],
            'elements': [],
            'element_counts': [],
            'coordinate_type': '',
            'positions': [],
            'selective_dynamics': False,
            'velocities': []
        }
        
        # Read lattice vectors
        for i in range(2, 5):
            structure['lattice'].append([float(x) for x in lines[i].split()])
        
        # Read element names and counts
        element_line = lines[5].strip().split()
        
        # Check if line 5 contains element symbols
        if element_line[0].isalpha():
            structure['elements'] = element_line
            structure['element_counts'] = [int(x) for x in lines[6].split()]
            coord_line = 7
        else:
            # Old VASP format without element names
            structure['element_counts'] = [int(x) for x in element_line]
            coord_line = 6
            # Try to guess elements from comment line
            print("Warning: POSCAR doesn't contain element symbols in standard position")
        
        # Check for selective dynamics
        if lines[coord_line].strip()[0] in ['S', 's']:
            structure['selective_dynamics'] = True
            coord_line += 1
        
        # Read coordinate type
        coord_type = lines[coord_line].strip()[0].upper()
        structure['coordinate_type'] = 'Direct' if coord_type == 'D' else 'Cartesian'
        
        # Read atomic positions
        total_atoms = sum(structure['element_counts'])
        for i in range(coord_line + 1, coord_line + 1 + total_atoms):
            if i < len(lines):
                pos_data = lines[i].split()
                position = [float(pos_data[0]), float(pos_data[1]), float(pos_data[2])]
                structure['positions'].append(position)
        
        print(f"  System: {structure['comment']}")
        print(f"  Elements: {' '.join(structure['elements'])}")
        print(f"  Atoms: {structure['element_counts']} (Total: {total_atoms})")
        
        return structure
    
    def write_poscar(self, structure: Dict, output_file: str):
        """
        Write VASP POSCAR file
        
        Args:
            structure: Structure dictionary
            output_file: Output file path
        """
        with open(output_file, 'w') as f:
            f.write(f"{structure['comment']}\n")
            f.write(f"  {structure['scale']:.10f}\n")
            
            # Write lattice vectors
            for vec in structure['lattice']:
                f.write(f"  {vec[0]:20.16f}  {vec[1]:20.16f}  {vec[2]:20.16f}\n")
            
            # Write element names and counts
            f.write("  " + "  ".join(structure['elements']) + "\n")
            f.write("  " + "  ".join(map(str, structure['element_counts'])) + "\n")
            
            # Write selective dynamics if present
            if structure['selective_dynamics']:
                f.write("Selective dynamics\n")
            
            # Write coordinate type
            f.write(f"{structure['coordinate_type']}\n")
            
            # Write positions
            for pos in structure['positions']:
                f.write(f"  {pos[0]:20.16f}  {pos[1]:20.16f}  {pos[2]:20.16f}\n")
        
        print(f"  Written POSCAR to: {output_file}")
    
    def create_incar_relax(self) -> str:
        """
        Create INCAR file for structure relaxation (ISIF=2)
        
        Returns:
            INCAR content as string
        """
        incar = """# ========================================
# VASP INCAR: Structure Relaxation (ISIF=2)
# Generated: {timestamp}
# ========================================

# Electronic minimization
PREC     = {PREC}          # Precision mode
ENCUT    = {ENCUT}         # Cutoff energy (eV)
EDIFF    = {EDIFF}         # Electronic SCF convergence (eV)
NELM     = {NELM}          # Maximum electronic steps
ALGO     = {ALGO}          # SCF algorithm

# Ionic relaxation
IBRION   = 2               # Conjugate gradient algorithm
NSW      = 200             # Maximum ionic steps
ISIF     = 2               # Relax ions only (no cell shape/volume)
EDIFFG   = {EDIFFG}        # Ionic convergence criterion (eV/Å)
POTIM    = {POTIM}         # Time step for ionic motion (fs)

# Electronic structure
ISMEAR   = {ISMEAR}        # Smearing method (0=Gaussian)
SIGMA    = {SIGMA}         # Smearing width (eV)

# DFT functional (PBE is default)
GGA      = PE              # PBE functional

# vdW correction
IVDW     = {IVDW}          # DFT-D3 method of Grimme
LVDW     = .TRUE.          # Enable vdW correction

# Performance optimization
LREAL    = {LREAL}         # Real space projection (Auto for large systems)
NCORE    = 4               # Number of cores per orbital

# Output control
LWAVE    = .TRUE.          # Write WAVECAR (for electronic structure calculation)
LCHARG   = .TRUE.          # Write CHGCAR (for charge density)
LVTOT    = .FALSE.         # Don't write total potential
LORBIT   = 11              # Write DOSCAR and lm-decomposed PROCAR

# Accuracy settings
ADDGRID  = .TRUE.          # Additional support grid for augmentation charges
LASPH    = .TRUE.          # Include non-spherical contributions
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **self.calc_params
        )
        
        return incar
    
    def create_incar_dos(self) -> str:
        """
        Create INCAR file for DOS calculation
        
        Returns:
            INCAR content as string
        """
        incar = """# ========================================
# VASP INCAR: Electronic Structure (DOS)
# Generated: {timestamp}
# ========================================

# Electronic minimization
PREC     = {PREC}
ENCUT    = {ENCUT}
EDIFF    = {EDIFF}
NELM     = {NELM}
ALGO     = {ALGO}

# Static calculation (no ionic relaxation)
IBRION   = -1              # No ionic update
NSW      = 0               # No ionic steps

# Electronic structure
ISMEAR   = -5              # Tetrahedron method with Blöchl corrections
SIGMA    = 0.05            # Not used with ISMEAR=-5, but kept for consistency

# DFT functional
GGA      = PE              # PBE functional

# vdW correction (should match relaxation)
IVDW     = {IVDW}
LVDW     = .TRUE.

# DOS-specific settings
LORBIT   = 11              # lm-decomposed DOS
NEDOS    = 3000            # Number of gridpoints for DOS
EMIN     = -20.0           # Minimum energy for DOS (eV)
EMAX     = 20.0            # Maximum energy for DOS (eV)

# Performance
LREAL    = {LREAL}
NCORE    = 4

# Output control
LWAVE    = .FALSE.         # Don't write WAVECAR
LCHARG   = .TRUE.          # Write CHGCAR for band structure
LVTOT    = .FALSE.

# Accuracy
ADDGRID  = .TRUE.
LASPH    = .TRUE.
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **self.calc_params
        )
        
        return incar
    
    def create_incar_band(self) -> str:
        """
        Create INCAR file for band structure calculation
        
        Returns:
            INCAR content as string
        """
        incar = """# ========================================
# VASP INCAR: Band Structure Calculation
# Generated: {timestamp}
# ========================================

# Electronic minimization
PREC     = {PREC}
ENCUT    = {ENCUT}
EDIFF    = {EDIFF}
NELM     = {NELM}
ALGO     = {ALGO}

# Static calculation
IBRION   = -1
NSW      = 0

# Electronic structure
ISMEAR   = 0               # Gaussian smearing for band structure
SIGMA    = 0.05
ICHARG   = 11              # Read charge density from CHGCAR (non-SCF)

# DFT functional
GGA      = PE

# vdW correction
IVDW     = {IVDW}
LVDW     = .TRUE.

# Band structure specific
LORBIT   = 11              # Project bands onto atoms
NEDOS    = 3000            # Fine energy grid

# Performance
LREAL    = .FALSE.         # Reciprocal space (more accurate for bands)
NCORE    = 4

# Output control
LWAVE    = .FALSE.
LCHARG   = .FALSE.
LVTOT    = .FALSE.

# Accuracy
ADDGRID  = .TRUE.
LASPH    = .TRUE.
""".format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **self.calc_params
        )
        
        return incar
    
    def create_kpoints_gamma(self, grid: Tuple[int, int, int] = (1, 1, 1)) -> str:
        """
        Create KPOINTS file with Gamma-centered mesh
        
        Args:
            grid: K-point mesh (kx, ky, kz)
            
        Returns:
            KPOINTS content as string
        """
        kpoints = """Automatic mesh (Gamma-centered)
0
Gamma
{0} {1} {2}
0 0 0
""".format(*grid)
        
        return kpoints
    
    def create_kpoints_band(self, system_type: str = 'cubic') -> str:
        """
        Create KPOINTS file for band structure calculation
        
        Args:
            system_type: Crystal system type ('cubic', 'hexagonal', etc.)
            
        Returns:
            KPOINTS content as string
        """
        if system_type == 'cubic':
            # Standard cubic path: Γ-X-M-Γ-R-X
            kpoints = """K-path for band structure (Cubic)
40                 # Number of points between high-symmetry points
Line-mode
Reciprocal
0.0 0.0 0.0    ! Gamma
0.5 0.0 0.5    ! X

0.5 0.0 0.5    ! X  
0.5 0.5 0.5    ! M

0.5 0.5 0.5    ! M
0.0 0.0 0.0    ! Gamma

0.0 0.0 0.0    ! Gamma
0.5 0.5 0.0    ! R

0.5 0.5 0.0    ! R
0.5 0.0 0.5    ! X
"""
        elif system_type == 'hexagonal':
            # Hexagonal path: Γ-M-K-Γ-A
            kpoints = """K-path for band structure (Hexagonal)
40
Line-mode
Reciprocal
0.0000 0.0000 0.0000    ! Gamma
0.5000 0.0000 0.0000    ! M

0.5000 0.0000 0.0000    ! M
0.3333 0.3333 0.0000    ! K

0.3333 0.3333 0.0000    ! K
0.0000 0.0000 0.0000    ! Gamma

0.0000 0.0000 0.0000    ! Gamma
0.0000 0.0000 0.5000    ! A
"""
        else:
            # Generic path
            kpoints = """K-path for band structure (Generic)
40
Line-mode
Reciprocal
0.0 0.0 0.0    ! Gamma
0.5 0.0 0.0    ! X

0.5 0.0 0.0    ! X
0.5 0.5 0.0    ! M

0.5 0.5 0.0    ! M
0.0 0.0 0.0    ! Gamma

0.0 0.0 0.0    ! Gamma
0.0 0.0 0.5    ! Z
"""
        
        return kpoints
    
    def create_potcar_info(self, elements: List[str]) -> str:
        """
        Create instructions for POTCAR file
        
        Args:
            elements: List of element symbols
            
        Returns:
            Information string about POTCAR setup
        """
        info = """# ========================================
# POTCAR Setup Instructions
# ========================================

You need to concatenate POTCAR files for the following elements in this order:
"""
        for i, elem in enumerate(elements, 1):
            info += f"{i}. {elem} (PBE functional)\n"
        
        info += """
Command to create POTCAR:
cat """
        
        for elem in elements:
            info += f"$VASP_PP_PATH/PBE/{elem}/POTCAR "
        
        info += "> POTCAR\n"
        info += """
Where $VASP_PP_PATH is your VASP pseudopotential directory.

Recommended POTCAR variants:
- Use PAW_PBE for standard calculations
- For 3d transition metals, consider using _sv or _pv variants
- For post-3d elements, consider using _d variants

Example VASP_PP_PATH:
export VASP_PP_PATH=/path/to/vasp/potentials/potpaw_PBE
"""
        
        return info
    
    def setup_relaxation(self, poscar_file: str, job_name: str, 
                        kpoint_grid: Tuple[int, int, int] = (4, 4, 4)) -> Path:
        """
        Setup structure relaxation calculation
        
        Args:
            poscar_file: Input POSCAR file
            job_name: Name for this calculation
            kpoint_grid: K-point mesh
            
        Returns:
            Path to relaxation directory
        """
        print("\n" + "="*60)
        print("Setting up STRUCTURE RELAXATION calculation")
        print("="*60)
        
        # Create relaxation directory
        relax_dir = self.working_dir / f"{job_name}_relax"
        relax_dir.mkdir(exist_ok=True)
        
        # Read and copy structure
        structure = self.read_poscar(poscar_file)
        self.write_poscar(structure, str(relax_dir / "POSCAR"))
        
        # Create INCAR
        incar_content = self.create_incar_relax()
        with open(relax_dir / "INCAR", 'w') as f:
            f.write(incar_content)
        print(f"  Created INCAR (ISIF=2, vdW=D3)")
        
        # Create KPOINTS
        kpoints_content = self.create_kpoints_gamma(kpoint_grid)
        with open(relax_dir / "KPOINTS", 'w') as f:
            f.write(kpoints_content)
        print(f"  Created KPOINTS (Gamma-centered {kpoint_grid[0]}x{kpoint_grid[1]}x{kpoint_grid[2]})")
        
        # Create POTCAR info
        potcar_info = self.create_potcar_info(structure['elements'])
        with open(relax_dir / "POTCAR_INFO.txt", 'w') as f:
            f.write(potcar_info)
        print(f"  Created POTCAR_INFO.txt")
        
        print(f"\nRelaxation setup complete: {relax_dir}")
        return relax_dir
    
    def setup_dos(self, relax_dir: Path, kpoint_grid: Tuple[int, int, int] = (8, 8, 8)) -> Path:
        """
        Setup DOS calculation using relaxed structure
        
        Args:
            relax_dir: Directory containing relaxation results
            kpoint_grid: Denser K-point mesh for DOS
            
        Returns:
            Path to DOS directory
        """
        print("\n" + "="*60)
        print("Setting up DOS calculation")
        print("="*60)
        
        # Create DOS directory
        dos_dir = relax_dir.parent / f"{relax_dir.stem.replace('_relax', '_dos')}"
        dos_dir.mkdir(exist_ok=True)
        
        # Copy relaxed structure (CONTCAR -> POSCAR)
        contcar = relax_dir / "CONTCAR"
        if not contcar.exists():
            print(f"Warning: CONTCAR not found in {relax_dir}")
            print("  Using original POSCAR instead")
            shutil.copy(relax_dir / "POSCAR", dos_dir / "POSCAR")
        else:
            shutil.copy(contcar, dos_dir / "POSCAR")
            print(f"  Copied relaxed structure from CONTCAR")
        
        # Copy CHGCAR and WAVECAR if they exist (for faster convergence)
        for file in ['CHGCAR', 'WAVECAR']:
            src = relax_dir / file
            if src.exists():
                shutil.copy(src, dos_dir / file)
                print(f"  Copied {file}")
        
        # Create INCAR for DOS
        incar_content = self.create_incar_dos()
        with open(dos_dir / "INCAR", 'w') as f:
            f.write(incar_content)
        print(f"  Created INCAR (ISMEAR=-5 for DOS)")
        
        # Create denser KPOINTS
        kpoints_content = self.create_kpoints_gamma(kpoint_grid)
        with open(dos_dir / "KPOINTS", 'w') as f:
            f.write(kpoints_content)
        print(f"  Created KPOINTS ({kpoint_grid[0]}x{kpoint_grid[1]}x{kpoint_grid[2]})")
        
        # Copy POTCAR info
        potcar_info_src = relax_dir / "POTCAR_INFO.txt"
        if potcar_info_src.exists():
            shutil.copy(potcar_info_src, dos_dir / "POTCAR_INFO.txt")
        
        print(f"\nDOS setup complete: {dos_dir}")
        return dos_dir
    
    def setup_band(self, dos_dir: Path, system_type: str = 'cubic') -> Path:
        """
        Setup band structure calculation
        
        Args:
            dos_dir: Directory containing DOS calculation results
            system_type: Crystal system type
            
        Returns:
            Path to band structure directory
        """
        print("\n" + "="*60)
        print("Setting up BAND STRUCTURE calculation")
        print("="*60)
        
        # Create band directory
        band_dir = dos_dir.parent / f"{dos_dir.stem.replace('_dos', '_band')}"
        band_dir.mkdir(exist_ok=True)
        
        # Copy structure and CHGCAR
        shutil.copy(dos_dir / "POSCAR", band_dir / "POSCAR")
        print(f"  Copied POSCAR")
        
        chgcar = dos_dir / "CHGCAR"
        if chgcar.exists():
            shutil.copy(chgcar, band_dir / "CHGCAR")
            print(f"  Copied CHGCAR (required for non-SCF calculation)")
        else:
            print(f"  Warning: CHGCAR not found in {dos_dir}")
        
        # Create INCAR for band structure
        incar_content = self.create_incar_band()
        with open(band_dir / "INCAR", 'w') as f:
            f.write(incar_content)
        print(f"  Created INCAR (non-SCF band calculation)")
        
        # Create KPOINTS for band structure
        kpoints_content = self.create_kpoints_band(system_type)
        with open(band_dir / "KPOINTS", 'w') as f:
            f.write(kpoints_content)
        print(f"  Created KPOINTS (high-symmetry path for {system_type})")
        
        # Copy POTCAR info
        potcar_info_src = dos_dir / "POTCAR_INFO.txt"
        if potcar_info_src.exists():
            shutil.copy(potcar_info_src, band_dir / "POTCAR_INFO.txt")
        
        print(f"\nBand structure setup complete: {band_dir}")
        return band_dir
    
    def create_submission_script(self, calc_dir: Path, calc_type: str,
                                scheduler: str = 'pbs', 
                                nodes: int = 1, ppn: int = 24,
                                walltime: str = '24:00:00',
                                queue: str = 'batch') -> Path:
        """
        Create job submission script for HPC cluster
        
        Args:
            calc_dir: Calculation directory
            calc_type: Type of calculation ('relax', 'dos', 'band')
            scheduler: Job scheduler ('pbs' or 'slurm')
            nodes: Number of compute nodes
            ppn: Processors per node
            walltime: Maximum walltime
            queue: Queue/partition name
            
        Returns:
            Path to submission script
        """
        job_name = f"{calc_dir.name}_{calc_type}"
        
        if scheduler.lower() == 'pbs':
            script = self._create_pbs_script(job_name, calc_dir, nodes, ppn, walltime, queue)
            script_file = calc_dir / "submit.pbs"
        elif scheduler.lower() == 'slurm':
            script = self._create_slurm_script(job_name, calc_dir, nodes, ppn, walltime, queue)
            script_file = calc_dir / "submit.slurm"
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        with open(script_file, 'w') as f:
            f.write(script)
        
        # Make script executable
        script_file.chmod(0o755)
        
        print(f"  Created job script: {script_file.name}")
        return script_file
    
    def _create_pbs_script(self, job_name: str, calc_dir: Path,
                          nodes: int, ppn: int, walltime: str, queue: str) -> str:
        """Create PBS job submission script"""
        
        script = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l nodes={nodes}:ppn={ppn}
#PBS -l walltime={walltime}
#PBS -q {queue}
#PBS -j oe
#PBS -o {job_name}.log

# ========================================
# VASP Job Submission Script (PBS)
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ========================================

# Change to working directory
cd $PBS_O_WORKDIR

# Load modules (modify according to your system)
module purge
module load intel/2020
module load vasp/5.4.4

# Set OpenMP threads
export OMP_NUM_THREADS=1

# Set VASP executable
VASP_EXE=vasp_std

# Number of MPI processes
NPROCS=$(cat $PBS_NODEFILE | wc -l)

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Job name: $PBS_JOBNAME"
echo "Working directory: $PWD"
echo "Number of nodes: {nodes}"
echo "Processes per node: {ppn}"
echo "Total processes: $NPROCS"
echo "=========================================="

# Check for required files
echo ""
echo "Checking input files..."
for file in POSCAR INCAR KPOINTS POTCAR; do
    if [ -f $file ]; then
        echo "  ✓ $file found"
    else
        echo "  ✗ $file NOT FOUND - Aborting!"
        exit 1
    fi
done

# Run VASP
echo ""
echo "Starting VASP calculation..."
echo "=========================================="
time mpirun -np $NPROCS $VASP_EXE

# Check if calculation completed successfully
if [ -f CONTCAR ] && [ -s CONTCAR ]; then
    echo ""
    echo "=========================================="
    echo "Calculation completed successfully"
    echo "Job finished at: $(date)"
    echo "=========================================="
    
    # Extract final energy
    if [ -f OSZICAR ]; then
        echo ""
        echo "Final energy:"
        tail -1 OSZICAR
    fi
else
    echo ""
    echo "=========================================="
    echo "WARNING: Calculation may have failed!"
    echo "Check output files for errors"
    echo "=========================================="
fi

# Save disk space by compressing large files
# echo ""
# echo "Compressing large files..."
# for file in WAVECAR CHG; do
#     if [ -f $file ]; then
#         gzip $file
#         echo "  Compressed $file"
#     fi
# done

exit 0
"""
        return script
    
    def _create_slurm_script(self, job_name: str, calc_dir: Path,
                            nodes: int, ppn: int, walltime: str, queue: str) -> str:
        """Create SLURM job submission script"""
        
        ntasks = nodes * ppn
        
        script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={ppn}
#SBATCH --time={walltime}
#SBATCH --partition={queue}
#SBATCH --output={job_name}_%j.log
#SBATCH --error={job_name}_%j.err

# ========================================
# VASP Job Submission Script (SLURM)
# Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ========================================

# Load modules (modify according to your system)
module purge
module load intel/2020
module load vasp/5.4.4

# Set OpenMP threads
export OMP_NUM_THREADS=1

# Set VASP executable
VASP_EXE=vasp_std

echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Total processes: $SLURM_NTASKS"
echo "=========================================="

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Check for required files
echo ""
echo "Checking input files..."
for file in POSCAR INCAR KPOINTS POTCAR; do
    if [ -f $file ]; then
        echo "  ✓ $file found"
    else
        echo "  ✗ $file NOT FOUND - Aborting!"
        exit 1
    fi
done

# Run VASP
echo ""
echo "Starting VASP calculation..."
echo "=========================================="
time srun $VASP_EXE

# Check if calculation completed successfully
if [ -f CONTCAR ] && [ -s CONTCAR ]; then
    echo ""
    echo "=========================================="
    echo "Calculation completed successfully"
    echo "Job finished at: $(date)"
    echo "=========================================="
    
    # Extract final energy
    if [ -f OSZICAR ]; then
        echo ""
        echo "Final energy:"
        tail -1 OSZICAR
    fi
else
    echo ""
    echo "=========================================="
    echo "WARNING: Calculation may have failed!"
    echo "Check output files for errors"
    echo "=========================================="
fi

exit 0
"""
        return script
    
    def run_complete_workflow(self, poscar_file: str, job_name: str,
                             relax_kgrid: Tuple[int, int, int] = (4, 4, 4),
                             dos_kgrid: Tuple[int, int, int] = (8, 8, 8),
                             system_type: str = 'cubic',
                             scheduler: str = 'pbs',
                             create_job_scripts: bool = True):
        """
        Run complete workflow: relaxation + DOS + band structure
        
        Args:
            poscar_file: Input POSCAR file
            job_name: Base name for calculations
            relax_kgrid: K-point grid for relaxation
            dos_kgrid: K-point grid for DOS
            system_type: Crystal system type
            scheduler: Job scheduler type
            create_job_scripts: Whether to create job submission scripts
        """
        print("\n" + "#"*60)
        print("# VASP COMPLETE WORKFLOW")
        print("#"*60)
        print(f"Input file: {poscar_file}")
        print(f"Job name: {job_name}")
        print(f"Scheduler: {scheduler.upper()}")
        print("#"*60)
        
        # Step 1: Setup relaxation
        relax_dir = self.setup_relaxation(poscar_file, job_name, relax_kgrid)
        
        if create_job_scripts:
            self.create_submission_script(relax_dir, 'relax', scheduler,
                                        nodes=1, ppn=24, walltime='24:00:00')
        
        # Step 2: Setup DOS
        dos_dir = self.setup_dos(relax_dir, dos_kgrid)
        
        if create_job_scripts:
            self.create_submission_script(dos_dir, 'dos', scheduler,
                                        nodes=1, ppn=24, walltime='12:00:00')
        
        # Step 3: Setup band structure
        band_dir = self.setup_band(dos_dir, system_type)
        
        if create_job_scripts:
            self.create_submission_script(band_dir, 'band', scheduler,
                                        nodes=1, ppn=24, walltime='6:00:00')
        
        # Print summary
        print("\n" + "="*60)
        print("WORKFLOW SETUP COMPLETE")
        print("="*60)
        print("\nDirectory structure:")
        print(f"  {relax_dir.name}/")
        print(f"  {dos_dir.name}/")
        print(f"  {band_dir.name}/")
        
        print("\nNext steps:")
        print("1. Copy POTCAR files to each directory:")
        print(f"   See {relax_dir}/POTCAR_INFO.txt for instructions")
        print("\n2. Submit jobs sequentially:")
        if scheduler == 'pbs':
            print(f"   cd {relax_dir} && qsub submit.pbs")
            print(f"   # Wait for relaxation to finish")
            print(f"   cd ../{dos_dir.name} && qsub submit.pbs")
            print(f"   # Wait for DOS to finish")
            print(f"   cd ../{band_dir.name} && qsub submit.pbs")
        elif scheduler == 'slurm':
            print(f"   cd {relax_dir} && sbatch submit.slurm")
            print(f"   # Wait for relaxation to finish")
            print(f"   cd ../{dos_dir.name} && sbatch submit.slurm")
            print(f"   # Wait for DOS to finish")
            print(f"   cd ../{band_dir.name} && sbatch submit.slurm")
        
        print("\n3. Analyze results:")
        print(f"   python parse_vasp_output.py -d {self.working_dir}")
        print("="*60)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="VASP Materials Calculation Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with single POSCAR file
  python vasp_workflow.py structure.vasp -n my_material
  
  # Specify K-point grids
  python vasp_workflow.py POSCAR -n graphene --relax-kgrid 6 6 2 --dos-kgrid 12 12 4
  
  # Use SLURM scheduler
  python vasp_workflow.py POSCAR -n material --scheduler slurm
  
  # Specify crystal system for band structure
  python vasp_workflow.py POSCAR -n hexagonal_mat --system hexagonal

Calculation Parameters:
  - Functional: PBE
  - Cutoff: 520 eV
  - vdW: DFT-D3 (IVDW=11)
  - EDIFF: 1E-05 eV
  - EDIFFG: -0.01 eV/Å
  - ISMEAR: 0 (Gaussian)
  - SIGMA: 0.05 eV
  - POTIM: 0.2 fs
  - ISIF: 2 (relax ions only)
        """
    )
    
    parser.add_argument('poscar', type=str,
                       help='Input POSCAR/CONTCAR file')
    parser.add_argument('-n', '--name', type=str, required=True,
                       help='Job name (used for directory names)')
    parser.add_argument('-o', '--output', type=str, default='vasp_calculations',
                       help='Output directory (default: vasp_calculations)')
    parser.add_argument('--relax-kgrid', type=int, nargs=3, default=[4, 4, 4],
                       metavar=('KX', 'KY', 'KZ'),
                       help='K-point grid for relaxation (default: 4 4 4)')
    parser.add_argument('--dos-kgrid', type=int, nargs=3, default=[8, 8, 8],
                       metavar=('KX', 'KY', 'KZ'),
                       help='K-point grid for DOS (default: 8 8 8)')
    parser.add_argument('--system', type=str, default='cubic',
                       choices=['cubic', 'hexagonal', 'generic'],
                       help='Crystal system type for band structure (default: cubic)')
    parser.add_argument('--scheduler', type=str, default='pbs',
                       choices=['pbs', 'slurm'],
                       help='Job scheduler type (default: pbs)')
    parser.add_argument('--no-job-scripts', action='store_true',
                       help='Do not create job submission scripts')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.poscar):
        print(f"Error: Input file '{args.poscar}' not found!")
        sys.exit(1)
    
    # Create workflow and run
    workflow = VASPWorkflow(working_dir=args.output)
    
    workflow.run_complete_workflow(
        poscar_file=args.poscar,
        job_name=args.name,
        relax_kgrid=tuple(args.relax_kgrid),
        dos_kgrid=tuple(args.dos_kgrid),
        system_type=args.system,
        scheduler=args.scheduler,
        create_job_scripts=not args.no_job_scripts
    )


if __name__ == "__main__":
    main()
