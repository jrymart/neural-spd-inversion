#!/bin/bash
#SBATCH --job-name=uv-init
#SBATCH --output=logs/setup_%j.log
#SBATCH --qos=blanca-csdms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:5:00

# 1. Environment Preparation
# Load the base Python module available on your cluster

export UV_CACHE_DIR="/projects/joma0457/.uv_cache"
export UV_INSTALL_DIR="$HOME/.local/bin"
export PATH="$UV_INSTALL_DIR:$PATH"

mkdir -p logs

echo "--- System Check ---"
echo "Host: $(hostname)"
echo "Date: $(date)"

# 2. Install/Update uv
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing to $UV_INSTALL_DIR..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the environment variables created by the installer
    source $HOME/.cargo/env
else
    echo "uv already installed: $(uv --version)"
fi

# 3. Build/Sync the Environment
# This reads your pyproject.toml and sets up the .venv
echo "--- Syncing Environment ---"
uv sync

# 4. Verification (The "Sanity Check")
# We use 'uv run' to ensure we are executing within the newly created .venv
echo "--- Verification ---"
uv run python <<EOF
import os
import sys
try:
    import neural_spd
    from neural_spd import config
    print(f"✅ Success: 'neural_spd' imported from {neural_spd.__file__}")
    print(f"✅ Config Check: Headless mode is {config.is_headless()}")
except ImportError as e:
    print(f"❌ Error: Could not import package. Check your src/ layout. Detail: {e}")
    sys.exit(1)
EOF

echo "--- Setup Complete ---"
