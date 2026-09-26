#!/bin/bash
#SBATCH --job-name=pvit_report
#SBATCH --partition=debug
#SBATCH --account=hpc
#SBATCH --qos=normal
#SBATCH --nice=100
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
set -euo pipefail
module load anaconda-2025.12-2
cd "$1/code"
python report.py "$1"
