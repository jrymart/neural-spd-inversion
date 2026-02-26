#!/bin/bash
#SBATCH --job-name=train-spd-net
#SBATCH --output=logs/array_runs/training_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=aa100,al40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # Data check usually benefits from more CPUs
#SBATCH --mem=32G            # Adjust based on your dataset size
#SBATCH --time=03:00:00
#SBATCH --array=0-159%10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jo.martin@colorado.edu
#SBATCH --requeue

# 2. Make sure uv is in your path
export UV_CACHE_DIR="/projects/joma0457/.uv_cache"
export PATH="$HOME/.local/bin:$PATH"

# 3. Run the script using 'uv run'
# This automatically handles the virtual environment and your src/ imports
echo "Starting training..."
uv run scripts/training.py
echo "Training complete."
