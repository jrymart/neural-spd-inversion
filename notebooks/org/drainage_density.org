#+title: Drainage Density
#+PROPERTY: header-args:python :python /home/jo/micromamba/envs/neural-spd-inversion/bin/python :session drainagedensity

Now we create a dataset of varying drainage density and see how the trained network is able to infer it.

#+begin_src python
import sqlite3
from landlab_torch_tools import AdaptiveThresholdDataset
from ThreeLayerCNNRegressor import ThreeLayerCNNRegressor
from pathlib import Path
import json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
import numpy as np
#+end_src

#+RESULTS:
: None

#+begin_src python
db_path = "../../../peclet-prediction/model_runs.db"
acc_dataset_dir = Path("../../../peclet-prediction/model_run_flowacc")
weight_path = "../../weights/dem_10_weights.pt"
model_stat_path = "../../model_stats.json"
#+end_src

#+RESULTS:
: None

#+begin_src python
connection = sqlite3.connect(db_path)
cursor = connection.cursor()
cursor.execute('SELECT model_run_id, "model_param.diffuser.D"/"model_param.streampower.k" FROM model_run_params')
Dks = cursor.fetchall()
Dks.sort(key = lambda x: x[1])
low_Dk_run = Dks[int(len(Dks)/10)][0]
low_Dk_array = torch.unsqueeze(torch.tensor(np.load(acc_dataset_dir / f"{low_Dk_run}.npy")[5:-5,5:-5]), 0)
threshold_dataset = AdaptiveThresholdDataset(
    input_array = low_Dk_array,
    num_thresholds = 1000,
    percentile_range=(1,99),
    return_threshold=True
)
#+end_src

#+RESULTS:
: None

#+begin_src python
MODEL_STATS_PATH = "../../model_stats.json"
DEM_WEIGHT_PATH = "../../weights/dem_10_weights.pt"
with open(MODEL_STATS_PATH, 'r') as f:
    stats = json.load(f)
labels_mean = stats['labels']['labels_mean']
labels_std = stats['labels']['labels_std']
loader = DataLoader(threshold_dataset, 64, shuffle=False)
model = ThreeLayerCNNRegressor()
model.load_state_dict(torch.load(DEM_WEIGHT_PATH))
model.eval()
drainage_densities = []
thresholds = []
norm_labels = []
with torch.no_grad():
    for data, threshold in loader:
        data = data.float()
        drainage_density = (data.sum(axis=(1,2,3))*5)/(np.prod(data.shape)*5*5)
        norm_label = model(data)
        drainage_densities += drainage_density
        norm_labels += norm_label
        thresholds += threshold
labels = [l*labels_std+labels_mean for l in norm_labels]
#+end_src

#+RESULTS:
: None

#+begin_src python :results graphics file output :file dd_performance.png
plt.scatter(drainage_densities, labels)
#+end_src

#+RESULTS:
[[file:dd_performance.png]]
