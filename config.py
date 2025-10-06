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

DATA_PATH = PROJECT_ROOT / "data"
WEIGHTS_DIR = PROJECT_ROOT / "model_weights"
DB_PATH = DATA_PATH / "model_runs.db"
NPY_URL = "https://zenodo.org/records/15311644/files/model_run_topography.tar.gz"
NPY_HASH = "md5:44ce5b9c0106c0751c11bd72547d4546"
DB_URL = "https://zenodo.org/records/15311644/files/model_runs.db"
DB_HASH = " md5:59144b8f047ca68c69419141abdf5852"
MODEL_DEM_DIR = "model_dems"
MODEL_ACC_DIR = "model_flowaccs"
TAR_DIR = "model_run_topography"
MODEL_DEM_PATH = os.path.join(DATA_PATH, MODEL_DEM_DIR, TAR_DIR)
MODEL_ARRAY_CROP = 5
LABEL_QUERY = "\"model_param.diffuser.D\"/ \"model_param.streampower.k\""
OUTPUTS_TABLE = "model_run_outputs"
MODEL_STATS_PATH = os.path.join(DATA_PATH, "model_stats.json")
NN_SEEDS = [0, 10, 20, 30]
