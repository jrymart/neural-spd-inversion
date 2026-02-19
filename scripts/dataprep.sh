#!/bin/bash
#SBATCH --job-name=data-prep
#SBATCH --output=logs/dataprep_%j.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16    # Data prep usually benefits from more CPUs
#SBATCH --mem=64G            # Adjust based on your dataset size
#SBATCH --time=02:00:00

# 1. Load your cluster's base Python (same as in your setup script)

# 2. Make sure uv is in your path
export PATH="$HOME/.local/bin:$PATH"

# 3. Run the script using 'uv run'
# This automatically handles the virtual environment and your src/ imports
echo "Starting data preparation..."
uv run scripts/dataprep.py
echo "Data preparation complete."
