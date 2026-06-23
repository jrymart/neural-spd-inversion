#!/bin/bash
#SBATCH --job-name=model_run_timing
#SBATCH --output=logs/lem_timing.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=03:00:00

uv run scripts/06_rev_lem_timing.py

