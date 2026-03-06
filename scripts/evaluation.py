import torch
from neural_spd.train_peclet_model import PecletModelTrainer
from neural_spd.ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json
import os
import numpy as np

from pathlib import Path
from neural_spd.config import MODEL_STATS_PATH, DB_PATH, MODEL_DEM_PATH, MODEL_SLOPE_PATH, MODEL_ACC_PATH, MODEL_CURV_PATH, WEIGHTS_PATH, NN_SEEDS, NUM_EPOCHS, LEARNING_RATE, RETRAIN_MODELS, NOISE_LEVELS, DATA_TYPES, LABELS, DATA_PATH, LOG_PATH, IS_HEADLESS, CHECKPOINT_PATH, BATCH_SIZE, RESULTS_DIR
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

def eval_neural_net(seed, noise, data_type, label):
    label_key, label_query = label
    torch.manual_seed(seed)
    weights_path = WEIGHTS_PATH / f"n{str(noise).replace('.','_')}_{data_type}_{seed}_{label_key}_weights.pt"
    csv_path = RESULTS_DIR / f"n{str(noise).replace('.','_')}_{data_type}_{seed}_{label_key}_results.csv"
    dataset_path = DATA_PATH / str(noise).replace('.','_') / data_type
    if not os.path.exists(dataset_path) or RETRAIN_MODELS:
        print(f"evaluating {weights_path}")
        trainer = PecletModelTrainer(DB_PATH,
                                    data['data_path'],
                                    ThreeLayerCNNRegressor(),
                                    LABEL_QUERY,
                                    epochs = NUM_EPOCHS,
                                    learning_rate = LEARNING_RATE,
                                    **statistics[data['type']],
                                    **statistics['labels'])
        trainer.load_weights(weights_path)
        trainer.evaluate()
        trainer.test_df.to_csv(csv_path)
    else:
        print(f"{weights_path} already evaluated, skipping")

runs = list(product(NN_SEEDS, NOISE_LEVELS, DATA_TYPES, LABELS.items()))
if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    eval_neural_net(*runs[task_id])
else:
    for run in runs:
        eval_neural_net(*run)
       
        # If data['type']=='dem':
        #     for second_data in data_types:
        #         weights_path = WEIGHTS_PATH / f"dem_{second_data['type']}_{seed}_weights.pt"
        #         if not weights_path.exists() or retrain:
        #             inputs_mean = np.stack([v['inputs_mean'] for k,v in statistics.items() if k in ('dem', second_data['type'])])[:, np.newaxis, np.newaxis]
        #             inputs_std= np.stack([v['inputs_std'] for k,v in statistics.items() if k in ('dem', second_data['type'])])[:, np.newaxis, np.newaxis]
        #             trainer = PecletModelTrainer(DB_PATH,
        #                                          [data['data_path'], second_data['data_path']],
        #                                          ThreeLayerCNNRegressor(channels=2),
        #                                          LABEL_QUERY,
        #                                          epochs = NUM_EPOCHS,
        #                                          learning_rate = LEARNING_RATE,
        #                                          inputs_mean = inputs_mean,
        #                                          inputs_std = inputs_std,
        #                                          **statistics['labels']
        #                                          )
        #             trainer.load_weights(weights_path)
        #             trainer.evaluate()
        #             csv_path = RESULTS_DIR / f"dem_{second_data['type']}_{seed}_results.csv}"
        #             trainer.test_df.to_csv(csv_path)

# weights_path = WEIGHTS_PATH / 'dem_jumbo_run_weights.pt'
# torch.manual_seed(NN_SEEDS[0])
# traier = PecletModelTrainer(DB_PATH,
#                             MODEL_DEM_PATH,
#                             JumboThreeLayerCNNRegressor,
#                             LABEL_QUERY,
#                             epochs = NUM_EPOCHS,
#                             learning_rate = LEARNING_RATE,
#                             **stats['dem'],
#                             **stats['labels'])
# trainer.load_weights(weights_path)
# trainer.evaluate()
# csv_path = RESULTS_DIR / 'dem_jumbo_run_results.pt')
# trainer.test_df.to_csv(csv_path)
