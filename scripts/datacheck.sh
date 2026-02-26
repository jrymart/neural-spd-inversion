#!/bin/bash
#SBATCH --job-name=data-check
#SBATCH --output=logs/datacheck_%j.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1    # Data check usually benefits from more CPUs
#SBATCH --mem=2G            # Adjust based on your dataset size
#SBATCH --time=00:10:00

# 1. Load your cluster's base Python (same as in your setup script)

# 2. Make sure uv is in your path
export UV_CACHE_DIR="/projects/joma0457/.uv_cache"
export PATH="$HOME/.local/bin:$PATH"

# 3. Run the script using 'uv run'
# This automatically handles the virtual environment and your src/ imports
echo "Starting data checkaration..."
uv run scripts/datacheck.py
echo "Data checkaration complete."
