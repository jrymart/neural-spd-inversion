import torch
from neural_spd.train_peclet_model import PecletModelTrainer
from neural_spd.ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json
import os
import numpy as np
import time
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
TIMING = os.getenv('TIMING', 'false').lower() == 'true'

TIMING = True
with open(MODEL_STATS_PATH, 'r') as f:
    statistics = json.load(f)

def eval_neural_net(seed, noise, data_type1, data_type2, label):
    if TIMING:
        start = time.perf_counter()
    label_key, label_query = label
    torch.manual_seed(seed)
    weights_path = WEIGHTS_PATH / f"n{str(noise).replace('.', '-')}_{data_type1}_{data_type2}_{seed}_{label_key}_weights.pt"
    csv_path = RESULTS_PATH/ f"n{str(noise).replace('.','-')}_{data_type1}_{data_type2}_{seed}_{label_key}_results.csv"
    log_path = LOG_PATH / f"n{str(noise).replace('.', '-')}_{data_type1}_{data_type2}_{seed}_{label_key}_training_log.json"
    dataset_paths = [DATA_PATH / str(noise).replace('.', '-') / data_type for data_type in [data_type1, data_type2]]
    checkpoint_path = CHECKPOINT_PATH / f"n{str(noise).replace('.', '-')}_{data_type1}_{data_type2}_{seed}_{label_key}_checkpoint.pt"
    if not weights_path.exists() or RETRAIN_MODELS or TIMING:
        print(f"Training {weights_path}")
        label_stats = statistics[label_key]
        data_mean = np.array([statistics[str(noise).replace('.', '-')][data_type1]['inputs_mean'],
                                statistics[str(noise).replace('.', '-')][data_type2]['inputs_mean']])[:, np.newaxis, np.newaxis]
        data_std = np.array([statistics[str(noise).replace('.', '-')][data_type1]['inputs_std'],
                                statistics[str(noise).replace('.', '-')][data_type2]['inputs_std']])[:, np.newaxis, np.newaxis]
        trainer = PecletModelTrainer(DB_PATH,
                                    dataset_paths,
                                    ThreeLayerCNNRegressor(channels=2),
                                    label_query,
                                    epochs=NUM_EPOCHS,
                                    learning_rate=LEARNING_RATE,
                                    batch_size=BATCH_SIZE,
                                    inputs_mean=data_mean,
                                    inputs_std=data_std,
                                    **label_stats)
        trainer.load_weights(weights_path)
        trainer.evaluate()
        if not os.path.exists(csv_path) or RETRAIN_MODELS:
            trainer.test_df.to_csv(csv_path)
        if TIMING:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            elapsed_time = end - start
            print(f"Evaluation time for {weights_path}: {elapsed_time} seconds for {len(trainer.test_df)} samples")
    else:
        print(f"{weights_path} already evaluated, skipping")
types = ['elevation', 'slope', 'curvature']
runs = list(product(NN_SEEDS, NOISE_LEVELS, types, types, LABELS.items()))
if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    eval_neural_net(*runs[task_id])
else:
    for run in runs:
        eval_neural_net(*run)
        break

# Resolution experiment evaluation (only if environment variable set)
if os.getenv('RUN_RESOLUTION_EXPERIMENTS', 'false').lower() == 'true':
    from neural_spd.config import RESOLUTION_EXPERIMENTS, ACTIVE_RESOLUTION

    print("\n=== Evaluating Resolution Experiments ===")
    resolution_runs = list(product(NN_SEEDS, NOISE_LEVELS, RESOLUTION_EXPERIMENTS.keys(), LABELS.items()))

    if IS_HEADLESS:
        # SLURM array job
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        eval_neural_net_resolution(*resolution_runs[task_id])
    else:
        # Local: can filter by ACTIVE_RESOLUTION environment variable
        active_resolutions = [ACTIVE_RESOLUTION] if ACTIVE_RESOLUTION != 'all' else RESOLUTION_EXPERIMENTS.keys()

        for seed, noise, resolution_name, label in resolution_runs:
            if resolution_name in active_resolutions:
                eval_neural_net_resolution(seed, noise, resolution_name, label)
       
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
