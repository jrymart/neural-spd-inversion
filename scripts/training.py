import torch
from train_peclet_model import PecletModelTrainer
from ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json
import os
import numpy as np

from pathlib import Path
from config import MODEL_STATS_PATH, DB_PATH, MODEL_DEM_PATH, MODEL_SLOPE_PATH, MODEL_ACC_PATH, MODEL_CURV_PATH, WEIGHTS_PATH, NN_SEEDS, NUM_EPOCHS, LEARNING_RATE, RETRAIN_MODELS, NOISE_LEVELS, DATA_TYPES, LABELS, DATA_PATH, LOG_PATH, IS_HEADLESS
import os

# Use environment variables if set (for HPC scratch filesystem)
DATA_PATH = Path(os.getenv('DATA_PATH', DATA_PATH))
WEIGHTS_PATH = Path(os.getenv('WEIGHTS_PATH', WEIGHTS_PATH))
MODEL_STATS_PATH = DATA_PATH / "model_stats.json"
DB_PATH = DATA_PATH / "model_runs.db"
from itertools import product

with open(MODEL_STATS_PATH, 'r') as f:
    statistics = json.load(f)

def train_neural_net(seed, noise, data_type, label):
    label_key, label_query = label
    torch.manual_seed(seed)
    weights_path = WEIGHTS_PATH / f"n{str(noise).replace('.', '-')}_{data_type}_{seed}_{label_key}_weights.pt"
    log_path = LOG_PATH / f"n{str(noise).replace('.', '-')}_{data_type}_{seed}_{label_key}_training_log.json"
    dataset_path = DATA_PATH / str(noise).replace('.', '-') / data_type
    if not weights_path.exists() or RETRAIN_MODELS:
        print(f"Training {weights_path}")
        label_stats = statistics[label_key]
        data_stats = statistics[str(noise).replace('.', '-')][data_type]
        trainer = PecletModelTrainer(DB_PATH,
                                    dataset_path,
                                    ThreeLayerCNNRegressor(),
                                    label_query,
                                    epochs=NUM_EPOCHS,
                                    learning_rate=LEARNING_RATE,
                                    **data_stats,
                                    **label_stats)
        trainer.train()
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

# weights_path = WEIGHTS_PATH / 'dem_jumbo_run_weights.pt'
# if not weights_path.exists() or RETRAIN_MODELS:
#     torch.manual_seed(NN_SEEDS[0])
#     # Use the first data type and label for jumbo model
#     first_label_key, first_label_query = list(LABELS.items())[0]
#     data_stats = statistics['0'][DATA_TYPES[0]]  # First data type, no noise
#     label_stats = statistics[first_label_key]
#     dataset_path = DATA_PATH / '0' / DATA_TYPES[0]
    
#     trainer = PecletModelTrainer(DB_PATH,
#                                 dataset_path,
#                                 JumboThreeLayerCNNRegressor(),
#                                 first_label_query,
#                                 epochs=NUM_EPOCHS,
#                                 learning_rate=LEARNING_RATE,
#                                 **data_stats,
#                                 **label_stats)
#     trainer.train()
#     trainer.save_weights(weights_path)
