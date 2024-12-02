#!/bin/bash
#SBATCH -J MYGPUJOB         # Job name
#SBATCH -o MYGPUJOB.o%j     # Name of job output file
#SBATCH -e MYGPUJOB.e%j     # Name of stderr error file
#SBATCH -p gpu              # Queue (partition) name for GPU nodes
#SBATCH -N 1                # Total # of nodes per instance
#SBATCH -n 1                # Total # of CPU cores (adjust as needed)
#SBATCH --gpus=1            # Request 1 GPU (adjust if more needed)
#SBATCH --mem=16G           # Memory (RAM) requested for gpu-mig
#SBATCH -t 01:00:00         # Run time (hh:mm:ss) for gpu-mig (adjust if needed)
#SBATCH --mail-type=all     # Send email at begin and end of job
#SBATCH --mail-user=isaac.n.malsky@jpl.nasa.gov

# Change to work directory
# NOTE: /scratch points to the scratch filesystem local
#       to the side of the cluster the job is running on.

# Dynamically locate Conda initialization script
CONDA_EXE=$(command -v conda)
CONDA_BASE=$(dirname $(dirname $CONDA_EXE))
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate nn

# Load CUDA module if needed (ensure compatibility with your environment)
module load cuda

# Run your Python script
python main.py
