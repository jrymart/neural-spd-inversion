import torch
import time
from neural_spd.train_peclet_model import PecletModelTrainer
from neural_spd.ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json
import os
import numpy as np

from pathlib import Path
from neural_spd.config import MODEL_STATS_PATH, DB_PATH, MODEL_DEM_PATH, MODEL_SLOPE_PATH, MODEL_ACC_PATH, MODEL_CURV_PATH, WEIGHTS_PATH, NN_SEEDS, NUM_EPOCHS, LEARNING_RATE, RETRAIN_MODELS, NOISE_LEVELS, DATA_TYPES, LABELS, DATA_PATH, LOG_PATH, IS_HEADLESS, CHECKPOINT_PATH, BATCH_SIZE
import os

# Use environment variables if set (for HPC scratch filesystem)
DATA_PATH = Path(os.getenv('DATA_PATH', DATA_PATH))
WEIGHTS_PATH = Path(os.getenv('WEIGHTS_PATH', WEIGHTS_PATH))
MODEL_STATS_PATH = DATA_PATH / "model_stats.json"
MODEL_STATS_PATH = Path(os.getenv('MODEL_STATS_PATH', MODEL_STATS_PATH))
DB_PATH = DATA_PATH / "model_runs.db"
DB_PATH = Path(os.getenv('DB_PATH', DB_PATH))
TIMING = os.getenv('TIMING', 'false').lower() == 'true'
from itertools import product

with open(MODEL_STATS_PATH, 'r') as f:
    statistics = json.load(f)

def train_neural_net(seed, noise, data_type, label, reload_from_checkpoint=True):
    if TIMING:
        start = time.perf_counter()
    label_key, label_query = label
    torch.manual_seed(seed)
    weights_path = WEIGHTS_PATH / f"n{str(noise).replace('.', '-')}_{data_type}_{seed}_{label_key}_weights.pt"
    log_path = LOG_PATH / f"n{str(noise).replace('.', '-')}_{data_type}_{seed}_{label_key}_training_log.json"
    dataset_path = DATA_PATH / str(noise).replace('.', '-') / data_type
    checkpoint_path = CHECKPOINT_PATH / f"n{str(noise).replace('.', '-')}_{data_type}_{seed}_{label_key}_checkpoint.pt"
    if not weights_path.exists() or RETRAIN_MODELS or TIMING:
        print(f"Training {weights_path}")
        label_stats = statistics[label_key]
        data_stats = statistics[str(noise).replace('.', '-')][data_type]
        trainer = PecletModelTrainer(DB_PATH,
                                    dataset_path,
                                    ThreeLayerCNNRegressor(),
                                    label_query,
                                    epochs=NUM_EPOCHS,
                                    learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE,
                                    **data_stats,
                                    **label_stats)
    
        trainer.train(checkpoint_path=checkpoint_path, reload_from_checkpoint=reload_from_checkpoint)
        if not weights_path.exists() or RETRAIN_MODELS:
            trainer.save_weights(weights_path)
            trainer.save_training_history(log_path)
    if TIMING:
        torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        end = time.perf_counter()
        elapsed_time = end - start
        print(f"Training time for {weights_path}: {elapsed_time} seconds")
    else:
        print(f"{weights_path} exists, skipping")

def train_neural_net_resolution(seed, noise, resolution_name, label, reload_from_checkpoint=True):
    """Train neural net for resolution experiment. Uses different naming to avoid conflicts."""
    from neural_spd.config import RESOLUTION_EXPERIMENTS, get_resolution_experiment_path

    label_key, label_query = label
    torch.manual_seed(seed)

    # Get resolution-specific paths
    data_path = get_resolution_experiment_path(noise, resolution_name) / "elevation"

    # Use backward-compatible naming for res5m baseline, new naming for others
    if resolution_name == 'res5m':
        # OLD naming convention: matches existing weights (n{noise}_elevation_{seed}_{label}_weights.pt)
        noise_str = str(noise).replace('.', '-')
        weights_path = WEIGHTS_PATH / f"n{noise_str}_elevation_{seed}_{label_key}_weights.pt"
        log_path = LOG_PATH / f"n{noise_str}_elevation_{seed}_{label_key}_training_log.json"
        checkpoint_path = CHECKPOINT_PATH / f"n{noise_str}_elevation_{seed}_{label_key}_checkpoint.pt"
    else:
        # NEW naming convention for new resolutions
        noise_str = str(noise).replace('.', '-')
        weights_path = WEIGHTS_PATH / f"{resolution_name}_n{noise_str}_{seed}_{label_key}_weights.pt"
        log_path = LOG_PATH / f"{resolution_name}_n{noise_str}_{seed}_{label_key}_training_log.json"
        checkpoint_path = CHECKPOINT_PATH / f"{resolution_name}_n{noise_str}_{seed}_{label_key}_checkpoint.pt"

    if not weights_path.exists() or RETRAIN_MODELS:
        print(f"Training {weights_path}")

        # Get statistics for this resolution experiment
        stats_key = f"{resolution_name}_{str(noise).replace('.', '-')}_elevation"
        data_stats = statistics[stats_key]
        label_stats = statistics[label_key]

        trainer = PecletModelTrainer(
            DB_PATH,
            data_path,  # Single elevation channel
            ThreeLayerCNNRegressor(channels=1),  # Single channel
            label_query,
            epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            **data_stats,
            **label_stats
        )
        trainer.train(checkpoint_path=checkpoint_path, reload_from_checkpoint=reload_from_checkpoint)
        trainer.save_weights(weights_path)
        trainer.save_training_history(log_path)
    else:
        print(f"{weights_path} exists, skipping")

runs = list(product(NN_SEEDS, NOISE_LEVELS, DATA_TYPES, LABELS.items()))
if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    train_neural_net(*runs[task_id])
else:
    for run in runs:
        train_neural_net(*run)

# Resolution experiment runs (only if environment variable set)
if os.getenv('RUN_RESOLUTION_EXPERIMENTS', 'false').lower() == 'true':
    from neural_spd.config import RESOLUTION_EXPERIMENTS, ACTIVE_RESOLUTION

    print("\n=== Running Resolution Experiments ===")
    resolution_runs = list(product(NN_SEEDS, NOISE_LEVELS, RESOLUTION_EXPERIMENTS.keys(), LABELS.items()))

    if IS_HEADLESS:
        # SLURM array job
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        train_neural_net_resolution(*resolution_runs[task_id])
    else:
        # Local: can filter by ACTIVE_RESOLUTION environment variable
        active_resolutions = [ACTIVE_RESOLUTION] if ACTIVE_RESOLUTION != 'all' else RESOLUTION_EXPERIMENTS.keys()

        for seed, noise, resolution_name, label in resolution_runs:
            if resolution_name in active_resolutions:
                train_neural_net_resolution(seed, noise, resolution_name, label)
