from landlab_torch_tools import SineTopographyDataset
import json
from neural_spd.ThreeLayerCNNRegressor import ThreeLayerCNNRegressor
from neural_spd.config import MODEL_STATS_PATH, WEIGHTS_PATH, NN_SEEDS, NUM_EPOCHS, LEARNING_RATE, NOISE_LEVELS, DATA_TYPES, LABELS, IS_HEADLESS, RESULTS_PATH, TOPO_DERIVATIVES, DB_PATH
from neural_spd.train_peclet_model import PecletModelTrainer
import torch
import pandas as pd
import os
from itertools import product

with open(MODEL_STATS_PATH) as f:
    stats = json.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eval_neural_net_on_sine(seed, noise, data_type, label):
    label_key, label_query = label
    torch.manual_seed(seed)
    weights_path = WEIGHTS_PATH / f"n{str(noise).replace('.','-')}_{data_type}_{seed}_{label_key}_weights.pt"
    csv_path = RESULTS_PATH/ "sine" / f"n{str(noise).replace('.','-')}_{data_type}_{seed}_{label_key}_results.csv"
    if not os.path.exists(csv_path):
        print(f"evaluating {weights_path}")
        label_stats = stats[label_key]
        sineset = SineTopographyDataset(90, 290, 'y')
        if data_type != 'elevation':
            sineset.transform = TOPO_DERIVATIVES[data_type]
        sineset.normalize = None
        trainer = PecletModelTrainer(DB_PATH,
                            None,
                            ThreeLayerCNNRegressor().to(device),
                            label_query,
                            epochs = NUM_EPOCHS,
                            learning_rate = LEARNING_RATE,
                            train_dataset=sineset,
                            test_dataset=sineset,
                            **label_stats)
        trainer.load_weights(weights_path)
        trainer.evaluate()
        # denomalrize predicted labels in trainer.test_df as model was trained on normalized values
        trainer.test_df['predictions'] = trainer.test_df['predictions'] * label_stats['labels_std'] + label_stats['labels_mean']
        # convert to valley spacing by taking 290*5 and dividing by the labels
        trainer.test_df['true_labels'] = 290*5 / trainer.test_df['true_labels']
        trainer.test_df.to_csv(csv_path)
        

    else:
        print(f"{weights_path} already evaluated, skipping")

runs = list(product(NN_SEEDS, NOISE_LEVELS, DATA_TYPES, LABELS.items()))
if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    eval_neural_net_on_sine(*runs[task_id])
else:
    for run in runs:
        eval_neural_net_on_sine(*run)
