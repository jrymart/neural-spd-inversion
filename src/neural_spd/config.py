from pathlib import Path
import os
from neural_spd.topographic_derivatives import slope, curvature, flow_accumulation
import numpy as np
from itertools import product

def flowacc(x):
    return flow_accumulation(x, MODEL_RESOLUTION, FLOW_METHOD)

def log10flowacc(x):
    return np.log10(flow_accumulation(x, MODEL_RESOLUTION, FLOW_METHOD))

# REPROCESSING SETTINGS
REPROCESS_DATA = False
RECALCULATE_STATS = True
RETRAIN_MODELS = False

# ENVIRONMENT DETECTION
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if not os.environ.get('DISPLAY') or 'SLURM_JOB_ID' in os.environ:
    IS_HEADLESS = True
else:
    IS_HEADLESS = False

# MAIN PATH SETUP
if IN_COLAB:
    PROJECT_ROOT = Path("/content/neural-spd-inversion")
    USE_GOOGLE_DRIVE = True
    if USE_GOOGLE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        DRIVE_PATH = Path("/content/drive/MyDrive")
        DATA_PATH = DRIVE_PATH / "neural-spd-inversion" / "data"
        WEIGHTS_PATH = DRIVE_PATH / "neural-spd-inversion" / "weights"
        RESULTS_PATH = DRIVE_PATH / "neural-spd-inversion" / "results"
        PLOTS_PATH = DRIVE_PATH / "neural-spd-inversion" / "plots"
        LOG_PATH = DRIVE_PATH / "neural-spd-inversion" / "logs"
    else:
        DATA_PATH = PROJECT_ROOT / "data"
        WEIGHTS_PATH = PROJECT_ROOT / "weights" 
        RESULTS_PATH = PROJECT_ROOT / "results"
        PLOTS_PATH = PROJECT_ROOT / "plots"
        LOG_PATH = PROJECT_ROOT / "logs"
else:
    # Local development
    PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
    DATA_PATH = PROJECT_ROOT / "data"
    WEIGHTS_PATH = PROJECT_ROOT / "weights"
    RESULTS_PATH = PROJECT_ROOT / "results"
    PLOTS_PATH = PROJECT_ROOT / "plots"
    LOG_PATH = PROJECT_ROOT / "logs"

NOISE_LEVELS = [0,0.1]
TOPO_DERIVATIVES = {
                    'slope': slope,
                    'curvature': curvature,
                    'flowacc': flowacc,
                    'log10flowacc': log10flowacc}

MODEL_DEM_DIR = "elevation"
DATA_TYPES = [MODEL_DEM_DIR] + list(TOPO_DERIVATIVES.keys())
NOISE_PATHS = [DATA_PATH / f"noise_level_{str(nl).replace('.', '-')}" for nl in NOISE_LEVELS]
LABELS = {'DoK':"SELECT \"model_param.diffuser.D\"/ \"model_param.streampower.k\" FROM model_run_params",
          'KoD':"SELECT \"model_param.streampower.k\"/ \"model_param.diffuser.D\" FROM model_run_params",
          'logDoK':"SELECT log(\"model_param.diffuser.D\"/ \"model_param.streampower.k\") FROM model_run_params",
          'logKoD':"SELECT log(\"model_param.streampower.k\"/ \"model_param.diffuser.D\") FROM model_run_params"}

for noise, deriviative in product(NOISE_LEVELS, DATA_TYPES):
    dataset_path = DATA_PATH / str(noise).replace('.', '-') / deriviative
    dataset_path.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_PATH / "model_runs.db"
MODEL_ACC_DIR = "model_flowaccs"
MODEL_LOG_ACC_DIR = "model_log_flowaccs"
MODEL_SLOPE_DIR = "model_slopes"
MODEL_CURV_DIR = "model_curvatures"
TAR_DIR = "model_run_topography"
MODEL_DEM_PATH = DATA_PATH / MODEL_DEM_DIR / "model_run_topography"
MODEL_STATS_PATH = DATA_PATH / "model_stats.json"
MODEL_ACC_PATH = DATA_PATH / MODEL_ACC_DIR
MODEL_LOG_ACC_PATH = DATA_PATH / MODEL_LOG_ACC_DIR
MODEL_SLOPE_PATH = DATA_PATH / MODEL_SLOPE_DIR
MODEL_CURV_PATH = DATA_PATH / MODEL_CURV_DIR
CREATE_DIRS = True

if CREATE_DIRS:
    WEIGHTS_PATH.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_ACC_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SLOPE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_CURV_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)

# DOWNLOAD URLS AND HASHES
NPY_URL = "https://zenodo.org/records/15311644/files/model_run_topography.tar.gz"
NPY_HASH = "md5:44ce5b9c0106c0751c11bd72547d4546"
DB_URL = "https://zenodo.org/records/15311644/files/model_runs.db"
DB_HASH = "md5:59144b8f047ca68c69419141abdf5852"

# MODEL SETTINGS
MODEL_RESOLUTION = 5
MODEL_ARRAY_CROP = 5

FLOW_METHOD = 'FlowDirectorD8'

# DB SETTINGS
#LABEL_QUERY = "SELECT \"model_param.diffuser.D\"/ \"model_param.streampower.k\" FROM model_run_params"
PARAM_TABLE = "model_run_params"
OUTPUTS_TABLE = "model_run_outputs"
SPLIT_BY_FIELD = 'model_param.seed'
RUN_ID_FIELD = "model_run_id"

# Training settings
NN_SEEDS = [0, 10, 20, 30]
TRAINING_FRACTION = 0.8
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001

# SLURM configuration
SLURM_JOB_NAME = 'neural-spd-training'
SLURM_PARTITION = 'gpu'  # Change based on your cluster
SLURM_NODES = 1
SLURM_NTASKS = 1
SLURM_CPUS_PER_TASK = 4
SLURM_MEMORY = '32G'
SLURM_TIME = '24:00:00'
SLURM_GPUS = 1  # Set to 0 if no GPU needed
SLURM_MAIL_TYPE = 'END,FAIL'
SLURM_MAIL_USER = 'your-email@domain.com'  # Update with actual email
SLURM_OUTPUT = 'logs/slurm-%j.out'
SLURM_ERROR = 'logs/slurm-%j.err'
