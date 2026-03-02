import torch
from train_peclet_model import PecletModelTrainer
from ThreeLayerCNNRegressor import ThreeLayerCNNRegressor, JumboThreeLayerCNNRegressor
import json

from config import MODEL_STATS_PATH, DB_PATH, MODEL_DEM_PATH, LABEL_QUERY, OUTPUTS_TABLE, WEIGHTS_PATH, NN_SEEDS, MODEL_ACC_PATH, RESULTS_PATH, NUM_EPOCHS, LEARNING_RATE, SEEDS, NOISE, DATA_TYPES
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

if IS_HEADLESS:
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    eval_neural_net(*runs[task_id])
else:
    for run in runs:
        eval_neural_net(*run)
       
        # if data['type']=='dem':
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
