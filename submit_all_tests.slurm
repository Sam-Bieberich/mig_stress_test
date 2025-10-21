#!/bin/bash
#SBATCH -J mig_tests_sequential
#SBATCH -o master_logs/slurm-%j.out
#SBATCH -e master_logs/slurm-%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -t 08:30:00
# Optional: set your allocation/account if required by the site
# SBATCH -A <your_allocation>

set -euo pipefail

# Move to the submission directory
cd "$SLURM_SUBMIT_DIR"

# Ensure logs dir exists
mkdir -p master_logs

# Load modules needed for PyTorch
if command -v module &> /dev/null; then
    module load gcc cuda || true
    module load python3 || true
fi

# If MIG is already configured by admins, the script will detect and skip setup.
# If MIG setup is not permitted on compute nodes (no sudo), our runner will log and continue.

# Run the sequential test runner in the foreground (Slurm manages lifecycle)
chmod +x run_all_tests_sequential.sh
bash run_all_tests_sequential.sh
