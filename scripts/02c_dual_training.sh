#!/bin/bash
#SBATCH --job-name=train-spd-net
#SBATCH --output=logs/array_runs/dual_training_%A_%a.log
#SBATCH --nodes=1
#SBATCH --partition=aa100,al40
#SBATCH --gres=gpu:1
#SBATCH --qos=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # Data check usually benefits from more CPUs
#SBATCH --mem=5G            # Adjust based on your dataset size
#SBATCH --time=00:10:00
#SBATCH --array=0-288%10
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jo.martin@colorado.edu
#SBATCH --requeue

# 2. Make sure uv is in your path
export UV_CACHE_DIR="/projects/joma0457/.uv_cache"
export PATH="$HOME/.local/bin:$PATH"

SEEDS=(0 10 20 30)
NOISE=("0" "0-1")
TYPES=("elevation" "slope" "curvature")

LABELS=("DoK" "KoD" "logDoK" "logKoD")

num_seeds=${#SEEDS[@]}
num_noise=${#NOISE[@]}
num_types=${#TYPES[@]}
num_labels=${#LABELS[@]}

label_idx=$(( SLURM_ARRAY_TASK_ID % num_labels ))
type1_idx=$(( (SLURM_ARRAY_TASK_ID / num_labels) % num_types ))
type2_idx=$(( (SLURM_ARRAY_TASK_ID / (num_labels * num_types)) % num_types ))
noise_idx=$(( (SLURM_ARRAY_TASK_ID / (num_labels * num_types * num_types)) % num_noise ))
seed_idx=$(( SLURM_ARRAY_TASK_ID / (num_labels * num_types * num_types * num_noise) ))

SELECTED_NOISE=${NOISE[$noise_idx]}
SELECTED_TYPE1=${TYPES[$type1_idx]}
SELECTED_TYPE2=${TYPES[$type2_idx]}


TASK_DATA="$SLURM_SCRATCH/neural-spd-inversion/data/$SELECTED_NOISE/"
mkdir -p "$TASK_DATA"

echo "Copying /projects/joma0457/neural-spd-inversion/data/$SELECTED_NOISE/$SELECTED_TYPE1 to $TASK_DATA"
echo "Copying /projects/joma0457/neural-spd-inversion/data/$SELECTED_NOISE/$SELECTED_TYPE2 to $TASK_DATA"

start_copy=$(date +%s)
cp -r "/projects/joma0457/neural-spd-inversion/data/$SELECTED_NOISE/$SELECTED_TYPE1" "$TASK_DATA/"
cp -r "/projects/joma0457/neural-spd-inversion/data/$SELECTED_NOISE/$SELECTED_TYPE2" "$TASK_DATA/"

cp "/projects/joma0457/neural-spd-inversion/data/model_runs.db" "$SLURM_SCRATCH/neural-spd-inversion/data"
cp "/projects/joma0457/neural-spd-inversion/data/model_stats.json" "$SLURM_SCRATCH/neural-spd-inversion/data"

end_copy=$(date +%s) # <--- ADD THIS LINE
echo "Copy completed in $((end_copy - start_copy)) seconds."

echo "--- VERIFICATION ---"
FILE_COUNT=$(ls -1 "$TASK_DATA/$SELECED_TYPE" | wc -l)
echo $(ls -l $TASK_DATA)
echo "Files found in local scratch: $FILE_COUNT"

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "ERROR: No files found in $TASK_DATA. Exiting job."
    exit 1
fi

export DATA_PATH="$SLURM_SCRATCH/neural-spd-inversion/data"
export DB_PATH="$SLURM_SCRATCH/neural-spd-inversion/data/model_runs.db"
export MODEL_STATS_PATH="$DATA_PATH/model_stats.json"
export SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

echo "Checking Python environment variables..."
uv run python -c "import os; print(f'PYTHON DATA_PATH: {os.getenv(\"DATA_PATH\")}')"
uv run python -c "import os; print(f'PYTHON DB_PATH: {os.getenv(\"DB_PATH\")}')"
uv run python -c "import os; print(f'PYTHON MODEL_STATS_PATH: {os.getenv(\"MODEL_STATS_PATH\")}')"
# 3. Run the script using 'uv run'
# This automatically handles the virtual environment and your src/ imports
echo "Starting training on noise: $SELECTED_NOISE and data: $SELECTED_TYPE..."
uv run scripts/02c_dual_training.py
echo "Training complete."
