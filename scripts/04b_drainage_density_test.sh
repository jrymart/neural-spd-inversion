#!/bin/bash
#sbatch --job-name=drainage_density
#sbatch --output=logs/drainage_density_%j.log
#sbatch --qos=blanca-csdms
#sbatch --nodes=1
#sbatch --ntasks=1
#sbatch --cpus-per-task=1
#sbatch --mem=1g
#sbatch --time=00:15:00

# 1. environment preparation
# load the base python module available on your cluster

export uv_cache_dir="/projects/joma0457/.uv_cache"
export path="$home/.local/bin:$path"

echo "running drainage density experiment"
uv run scripts/04b_drainage_density_test.py
echo "experiment complete."
