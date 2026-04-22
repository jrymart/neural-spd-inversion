#!/bin/bash
#SBATCH --job-name=topo_var
#SBATCH --output=logs/topo_var.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2g
#SBATCH --time=00:30:00

# 1. environment preparation
# load the base python module available on your cluster

export uv_cache_dir="/projects/joma0457/.uv_cache"
export PATH="$HOME/.local/bin:$path"

echo "running drainage density experiment"
uv run scripts/04d_calc_topo_variation.py
echo "experiment complete."

