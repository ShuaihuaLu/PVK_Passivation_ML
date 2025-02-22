#!/bin/bash
#SBATCH -J shlu              # Job name (shlu)
#SBATCH -p C6240             # Partition/queue to use (C6240)
##SBATCH -n 8                # Number of tasks (commented out)
#SBATCH -o %j.log            # Standard output log file (named after job ID)
#SBATCH -e %j.log            # Standard error log file (named after job ID)
#SBATCH --ntasks=1           # Number of tasks (1 task for this job)
#SBATCH --cpus-per-task=8    # Number of CPU cores per task (8 cores)
#SBATCH --nodes=1            # Number of nodes to use (1 node)
#SBATCH --mem=10G            # Memory required per task (10 GB)
#SBATCH --nodelist=s03       # Specific node to run the job on (s03)

ulimit -s unlimited
echo $SLURM_JOB_ID > ./jobid

module load gaussian/g09


# Find all .inp files in the current directory and its subdirectories
input_files=($(find . -type f -name "*.gjf"))

for input_file in "${input_files[@]}"; do
  # Extract the job name from the input file
  job_name="${input_file%.*}"

  # Create a directory for the current job
  mkdir -p "$job_name"
  job_dir="$job_name"

  # Copy the input file into the job directory
  cp "$input_file" run.gaussian.sh "$job_dir/"

  # Change to the job directory
  cd "$job_dir"

  #mv "$input_file" POSCAR

  #module load vasp/6.3.2
  g09 < *.gjf > output.log

  for inf in *.chk
  do
  formchk ${inf}
  done

  #srun --mpi=pmi2 cp2k.popt -i "$input_file" -o "${input_file%.*}.out" &> "${input_file%.*}.log"
  wait
  cd ..
done
