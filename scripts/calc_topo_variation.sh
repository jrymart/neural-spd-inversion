#!/bin/bash
#sbatch --job-name=topo_var
#sbatch --output=logs/topo_var.log
#sbatch --qos=blanca-csdms
#sbatch --nodes=1
#sbatch --ntasks=1
#sbatch --cpus-per-task=1
#sbatch --mem=2g
#sbatch --time=00:30:00

# 1. environment preparation
# load the base python module available on your cluster

export uv_cache_dir="/projects/joma0457/.uv_cache"
export path="$home/.local/bin:$path"

echo "running drainage density experiment"
uv run scripts/calc_topo_variation.py
echo "experiment complete."

