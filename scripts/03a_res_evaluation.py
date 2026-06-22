import torch
from neural_spd.train_peclet_model import PecletModelTrainer
from neural_spd.ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json
import os
import numpy as np

from pathlib import Path
from neural_spd.config import MODEL_STATS_PATH, DB_PATH, MODEL_DEM_PATH, MODEL_SLOPE_PATH, MODEL_ACC_PATH, MODEL_CURV_PATH, WEIGHTS_PATH, NN_SEEDS, NUM_EPOCHS, LEARNING_RATE, RETRAIN_MODELS, NOISE_LEVELS, DATA_TYPES, LABELS, DATA_PATH, LOG_PATH, IS_HEADLESS, CHECKPOINT_PATH, BATCH_SIZE, RESULTS_PATH
# Use environment variables if set (for HPC scratch filesystem)
DATA_PATH = Path(os.getenv('DATA_PATH', DATA_PATH))
WEIGHTS_PATH = Path(os.getenv('WEIGHTS_PATH', WEIGHTS_PATH))
MODEL_STATS_PATH = DATA_PATH / "model_stats.json"
MODEL_STATS_PATH = Path(os.getenv('MODEL_STATS_PATH', MODEL_STATS_PATH))
DB_PATH = DATA_PATH / "model_runs.db"
DB_PATH = Path(os.getenv('DB_PATH', DB_PATH))
from itertools import product

with open(MODEL_STATS_PATH, 'r') as f:
    statistics = json.load(f)

def eval_neural_net_resolution(seed, noise, resolution_name, label):
    """Evaluate neural net for resolution experiment. Matches training naming convention."""
    from neural_spd.config import RESOLUTION_EXPERIMENTS, get_resolution_experiment_path

    label_key, label_query = label
    torch.manual_seed(seed)

    data_path = get_resolution_experiment_path(noise, resolution_name) / "elevation"

    # Use same naming convention as training
    if resolution_name == 'res5m':
        # OLD naming convention: matches existing results
        noise_str = str(noise).replace('.', '-')
        weights_path = WEIGHTS_PATH / f"n{noise_str}_elevation_{seed}_{label_key}_weights.pt"
        csv_path = RESULTS_PATH / f"n{noise_str}_elevation_{seed}_{label_key}_results.csv"
    else:
        # NEW naming convention for new resolutions
        noise_str = str(noise).replace('.', '-')
        weights_path = WEIGHTS_PATH / f"{resolution_name}_n{noise_str}_{seed}_{label_key}_weights.pt"
        csv_path = RESULTS_PATH / f"{resolution_name}_n{noise_str}_{seed}_{label_key}_results.csv"

    if not os.path.exists(csv_path) or RETRAIN_MODELS:
        print(f"Evaluating {weights_path}")

        stats_key = f"{resolution_name}_{str(noise).replace('.', '-')}_elevation"
        data_stats = statistics[stats_key]
        label_stats = statistics[label_key]

        trainer = PecletModelTrainer(
            DB_PATH,
            data_path,
            ThreeLayerCNNRegressor(channels=1),
            label_query,
            epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            **data_stats,
            **label_stats
        )
        trainer.load_weights(weights_path)
        trainer.evaluate()
        trainer.test_df.to_csv(csv_path)
    else:
        print(f"{weights_path} already evaluated, skipping")

resolutions=["res7m", "res8m", "res10m"]
runs = list(product(NN_SEEDS, NOISE_LEVELS, resolutions, LABELS.items()))
if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    eval_neural_net_resolution(*runs[task_id])
else:
    for run in runs:
        eval_neural_net_resolution(*run)
