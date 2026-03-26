#!/bin/bash
#SBATCH --job-name=drainage_density
#SBATCH --output=logs/drainage_density_%j.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:15:00

# 1. Environment Preparation
# Load the base Python module available on your cluster

export UV_CACHE_DIR="/projects/joma0457/.uv_cache"
export PATH="$HOME/.local/bin:$PATH"

echo "Running drainage density experiment"
uv run scripts/drainage_density.py
echo "Experiment complete."
