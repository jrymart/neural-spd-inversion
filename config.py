from pathlib import Path
import os

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    PROJECT_ROOT = Path("/content/neural-spd-inversion")
else:
    PROJECT_ROOT = Path(__file__).parent.absolute()
USE_GOOGLE_DRIVE = True

if IN_COLAB and USE_GOOGLE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    DRIVE_PATH = Path("/content/drive/MyDrive")
    DATA_PATH = DRIVE_PATH / "neural-spd-inversion-data"
    WEIGHTS_PATH = DRIVE_PATH / "neural-spd-inversion-weights"
else:
    DATA_PATH = PROJECT_ROOT / "data"
    WEIGHTS_PATH = PROJECT_ROOT / "model_weights"

REPROCESS_DATA = False
DB_PATH = DATA_PATH / "model_runs.db"
NPY_URL = "https://zenodo.org/records/15311644/files/model_run_topography.tar.gz"
NPY_HASH = "md5:44ce5b9c0106c0751c11bd72547d4546"
DB_URL = "https://zenodo.org/records/15311644/files/model_runs.db"
DB_HASH = "md5:59144b8f047ca68c69419141abdf5852"
MODEL_RESOLUTION = 5
MODEL_DEM_DIR = "model_dems"
MODEL_ACC_DIR = "model_flowaccs"
MODEL_SLOPE_DIR = "model_slopes"
MODEL_CURV_DIR = "model_curvatures"
TAR_DIR = "model_run_topography"
MODEL_DEM_PATH = os.path.join(DATA_PATH, MODEL_DEM_DIR, TAR_DIR)
MODEL_ARRAY_CROP = 5
LABEL_QUERY = "SELECT \"model_param.diffuser.D\"/ \"model_param.streampower.k\" FROM model_run_params"
PARAM_TABLE = "model_run_params"
OUTPUTS_TABLE = "model_run_outputs"
SPLIT_BY_FIELD = 'model_param.seed'
TRAINING_FRACTION = 0.8
RUN_ID_FIELD = "model_run_id"
MODEL_STATS_PATH = os.path.join(DATA_PATH, "model_stats.json")
NN_SEEDS = [0, 10, 20, 30]
FLOW_METHOD = 'FlowDirectorD8'
MODEL_DEM_PATH = DATA_PATH / MODEL_DEM_DIR / "model_run_topography"
MODEL_ACC_PATH = DATA_PATH / MODEL_ACC_DIR
MODEL_SLOPE_PATH = DATA_PATH / MODEL_SLOPE_DIR
MODEL_CURV_PATH = DATA_PATH / MODEL_CURV_DIR
CREATE_DIRS = True

if CREATE_DIRS:
    MODEL_ACC_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_SLOPE_PATH.mkdir(parents=True, exist_ok=True)
    MODEL_CURV_PATH.mkdir(parents=True, exist_ok=True)
