#!/usr/bin/env python3
"""
Generate SLURM batch scripts for running Python scripts on HPC clusters.

Usage:
    python generate_slurm.py <python_script> [output_script]
    
Examples:
    python generate_slurm.py train_model.py
    python generate_slurm.py process_data.py slurm_dataprep.sh

The script reads SLURM configuration from config.py and generates a batch script
that can be submitted with 'sbatch <output_script>'.
"""

import sys
from pathlib import Path

def generate_slurm_script(python_script, output_script=None):
    """Generate a SLURM batch script for a Python script."""
    
    # Import config here so error is clear if config.py is missing
    try:
        from config import (SLURM_JOB_NAME, SLURM_PARTITION, SLURM_NODES, 
                           SLURM_NTASKS, SLURM_CPUS_PER_TASK, SLURM_MEMORY, 
                           SLURM_TIME, SLURM_GPUS, SLURM_MAIL_TYPE, 
                           SLURM_MAIL_USER, SLURM_OUTPUT, SLURM_ERROR,
                           IN_HPC, DATA_PATH, WEIGHTS_PATH, RESULTS_PATH)
        if IN_HPC:
            from config import (SCRATCH_DATA_PATH, SCRATCH_WEIGHTS_PATH, 
                               SCRATCH_RESULTS_PATH, HPC_PROJECTS, HPC_SCRATCH)
    except ImportError as e:
        print(f"Error: Could not import SLURM settings from config.py")
        print(f"Make sure config.py exists and contains SLURM_* variables")
        print(f"Details: {e}")
        sys.exit(1)
    
    if output_script is None:
        output_script = f"slurm_{Path(python_script).stem}.sh"
    
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Build SLURM script content
    slurm_content = f"""#!/bin/bash
#SBATCH --job-name={SLURM_JOB_NAME}
#SBATCH --partition={SLURM_PARTITION}
#SBATCH --nodes={SLURM_NODES}
#SBATCH --ntasks={SLURM_NTASKS}
#SBATCH --cpus-per-task={SLURM_CPUS_PER_TASK}
#SBATCH --mem={SLURM_MEMORY}
#SBATCH --time={SLURM_TIME}"""

    # Add GPU request if specified
    if SLURM_GPUS > 0:
        slurm_content += f"\n#SBATCH --gres=gpu:{SLURM_GPUS}"
    
    slurm_content += f"""
#SBATCH --mail-type={SLURM_MAIL_TYPE}
#SBATCH --mail-user={SLURM_MAIL_USER}
#SBATCH --output={SLURM_OUTPUT}
#SBATCH --error={SLURM_ERROR}

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME" 
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo ""

# Load modules (uncomment and modify as needed for your cluster)
# module load python/3.9
# module load cuda/11.2  
# module load pytorch

# Activate conda environment (uncomment and modify as needed)
# source ~/.bashrc
# conda activate neural-spd-inversion"""

    # Add HPC file management if running on HPC
    if IN_HPC:
        slurm_content += f"""

# HPC File Management: Copy data to scratch for fast I/O
echo "Setting up scratch directories..."
mkdir -p {SCRATCH_DATA_PATH}
mkdir -p {SCRATCH_WEIGHTS_PATH}
mkdir -p {SCRATCH_RESULTS_PATH}

echo "Copying data to scratch filesystem..."
if [ -d "{DATA_PATH}" ]; then
    cp -r {DATA_PATH}/* {SCRATCH_DATA_PATH}/
    echo "Data copied to {SCRATCH_DATA_PATH}"
else
    echo "Warning: {DATA_PATH} not found"
fi

# Copy any existing weights to scratch
if [ -d "{WEIGHTS_PATH}" ]; then
    cp -r {WEIGHTS_PATH}/* {SCRATCH_WEIGHTS_PATH}/ 2>/dev/null || echo "No existing weights to copy"
fi

# Set environment variable to use scratch paths during job
export DATA_PATH={SCRATCH_DATA_PATH}
export WEIGHTS_PATH={SCRATCH_WEIGHTS_PATH}
export RESULTS_PATH={SCRATCH_RESULTS_PATH}"""
    
    slurm_content += f"""

# Run the Python script
echo "Running {python_script}..."
python {python_script}"""

    # Add cleanup and copy-back for HPC
    if IN_HPC:
        slurm_content += f"""

# Copy results back to projects filesystem
echo "Copying results back to projects filesystem..."
mkdir -p {WEIGHTS_PATH}
mkdir -p {RESULTS_PATH}
cp -r {SCRATCH_WEIGHTS_PATH}/* {WEIGHTS_PATH}/ 2>/dev/null || echo "No weights to copy back"
cp -r {SCRATCH_RESULTS_PATH}/* {RESULTS_PATH}/ 2>/dev/null || echo "No results to copy back"

# Clean up scratch space
echo "Cleaning up scratch space..."
rm -rf {SCRATCH_DATA_PATH} {SCRATCH_WEIGHTS_PATH} {SCRATCH_RESULTS_PATH}"""

    slurm_content += f"""

echo ""
echo "Job completed at: $(date)"
"""
    
    # Write the script
    with open(output_script, 'w') as f:
        f.write(slurm_content)
    
    # Make executable
    Path(output_script).chmod(0o755)
    
    print(f"✓ Generated SLURM script: {output_script}")
    print(f"✓ Submit with: sbatch {output_script}")
    print(f"✓ Edit config.py to modify SLURM settings")
    
    return output_script

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    python_script = sys.argv[1]
    output_script = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(python_script).exists():
        print(f"Error: Python script '{python_script}' not found")
        print(f"Make sure you've tangled your org notebooks or have the script ready")
        sys.exit(1)
    
    generate_slurm_script(python_script, output_script)

if __name__ == "__main__":
    main()