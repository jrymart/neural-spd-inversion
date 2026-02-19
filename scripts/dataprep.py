import pooch
from neural_spd.config import DATA_PATH, NPY_URL, NPY_HASH, DB_URL, DB_HASH, MODEL_DEM_DIR

DATA_PATH.mkdir(exist_ok=True, parents=True)
model_dem_paths = pooch.retrieve(url=NPY_URL,
                           known_hash=NPY_HASH,
                           path=DATA_PATH,
                           processor=pooch.Untar(extract_dir=MODEL_DEM_DIR))
db_path = pooch.retrieve(url=DB_URL,
                           known_hash=DB_HASH,
                           fname="model_runs.db",
                           path=DATA_PATH)

import numpy as np
from pathlib import Path
from neural_spd.config import NOISE_LEVELS, DATA_PATH, MODEL_DEM_DIR, REPROCESS_DATA
rng = np.random.default_rng(seed=0)
model_dem_paths = [Path(path) for path in model_dem_paths]
old_dem_dir = model_dem_paths[0].parent
for noise in NOISE_LEVELS:
    new_dem_dir = DATA_PATH / str(noise).replace('.', '-') / MODEL_DEM_DIR
    if noise == 0:
        new_dem_dir.parent.mkdir(parents=True, exist_ok=True)
        if not new_dem_dir.exists():
                old_dem_dir.rename(new_dem_dir)
        else:
            import shutil
            for f in old_dem_dir.iterdir():
                shutil.move(str(f), str(new_dem_dir / f.name))
            old_dem_dir.rmdir()
        continue
    for dem_path in model_dem_paths:
        noise_path = new_dem_dir / dem_path.name
        if (not noise_path.exists()) or REPROCESS_DATA :
            dem_path = Path(dem_path)
            dem_array = np.load(dem_path)
            dem_noisy = dem_array + rng.normal(0, noise, dem_array.shape)
            noise_path = new_dem_dir / dem_path.name
            np.save(noise_path, dem_noisy)

def generate_derivative(args):
    noise, (derivative_name, derivative_function), dem_path = args
    derivative_path = DATA_PATH / str(noise).replace('.', '-') / derivative_name /dem_path.name
    if (not derivative_path.exists()) or REPROCESS_DATA:
        dem_path = DATA_PATH / str(noise).replace('.','-') / MODEL_DEM_DIR / dem_path.name
        dem_array = np.load(dem_path)
        derivative = derivative_function(dem_array)
        np.save(derivative_path, derivative)

import numpy as np
import os
from itertools import product
from concurrent.futures import ProcessPoolExecutor
from neural_spd.config import TOPO_DERIVATIVES, REPROCESS_DATA
#setup directories
for noise in NOISE_LEVELS:
        noise_str = str(noise).replace('.', '-')
        for deriv_name in TOPO_DERIVATIVES.keys():
            (DATA_PATH / noise_str / deriv_name).mkdir(parents=True, exist_ok=True)
            
derivatives = list(product(NOISE_LEVELS, TOPO_DERIVATIVES.items(), model_dem_paths))
num_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 1))
if __name__ == "__main__" and num_cpus>1:
    with ProcessPoolExecutor(max_workers=num_cpus) as executer:
        list(executer.map(generate_derivative, derivatives))
else:
    for derivative in derivatives:
        generate_derivative(derivative)

import numpy as np
def compute_partial_stats(args):
    paths, crop = args
    p_sum = 0.0
    p_sum_sq = 0.0
    p_count = 0
    for path in paths:
        data = np.load(path)[crop:-crop, crop:-crop]
        p_sum += np.sum(data)
        p_sum_sq +=  np.sum(np.square(data))
        p_count += data.size
    return p_sum, p_sum_sq, p_count

def get_parallel_statistics(all_paths, crop, num_cpus):
    chunks = np.array_split(all_paths, num_cpus)
    tasks = [(list(chunk), crop) for chunk in chunks]
    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        results = list(executor.map(compute_partial_stats, tasks))

    total_sum = sum(r[0] for r in results)
    total_sum_sq = sum(r[1] for r in results)
    total_count = sum(r[2] for r in results)

    mean = total_sum / total_count
    variance = (total_sum_sq / total_count) - np.square(mean)
    return {'inputs_mean': mean, 'inputs_std': np.sqrt(variance)}

def get_array_statistics(array_paths, crop):
    array_total_sum = 0.0
    array_total_sum_sq = 0.0
    array_total_count = 0
    for path in array_paths:
        data_array=np.load(path)[crop:-crop,crop:-crop]
        array_total_sum += np.sum(data_array)
        array_total_sum_sq += np.sum(np.square(data_array))
        array_total_count += data_array.size
    array_mean = array_total_sum / array_total_count
    variance = (array_total_sum_sq / array_total_count) - np.square(array_mean)
    array_std = np.sqrt(variance)
    return {'inputs_mean': array_mean, 'inputs_std': array_std}
import sqlite3
import json
from neural_spd.config import SPLIT_BY_FIELD, TRAINING_FRACTION, PARAM_TABLE, RUN_ID_FIELD, MODEL_DEM_PATH, MODEL_ARRAY_CROP, OUTPUTS_TABLE, MODEL_STATS_PATH, DB_PATH, RECALCULATE_STATS, DATA_TYPES

connection = sqlite3.connect(DB_PATH)
cursor = connection.cursor()
cursor.execute(f"SELECT DISTINCT \"{SPLIT_BY_FIELD}\" FROM {PARAM_TABLE}")
categories = [r[0] for r in cursor.fetchall()]
split = int((len(categories) * TRAINING_FRACTION))
train_categories = categories[:split]
train_filter = f"\"{SPLIT_BY_FIELD}\" IN ({', '.join([str(c) for c in train_categories])})"
cursor.execute(f"SELECT {RUN_ID_FIELD} FROM {PARAM_TABLE} WHERE {train_filter}")
train_run_ids = [r[0] for r in cursor.fetchall()]

if not RECALCULATE_STATS:
    try:
        with open(MODEL_STATS_PATH, 'r') as f:
            statistics = json.load(f)
    except FileNotFoundError:
        statistics = {}
else:
    statistics = {}
for noise in NOISE_LEVELS:
    noise = str(noise).replace('.', '-')
    if noise not in statistics:
        statistics[noise] = {}
    noise_stats = statistics[noise]
    for data_type in DATA_TYPES:
        if data_type not in noise_stats or RECALCULATE_STATS:
            dataset_path = DATA_PATH / noise / data_type 
            dataset_paths = [dataset_path / f"{name}.npy" for name in train_run_ids]
            if __name__ == '__main__' and num_cpus>1:
                noise_stats[data_type] = get_parallel_statistics(dataset_paths, MODEL_ARRAY_CROP, num_cpus)
            else:
                noise_stats[data_type] = get_array_statistics(dataset_paths,MODEL_ARRAY_CROP)

from neural_spd.config import LABELS
limit = connection.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
for label_name, label_query in LABELS.items():
    label_values = []
    for i in range(0, len(train_run_ids), limit):
        current_chunk_runs = train_run_ids[i:i+limit]
        # placeholders are a safe way to programatically construct an SQL query
        placeholders = ', '.join(['?']*len(current_chunk_runs))
        cursor.execute(f"{label_query} WHERE {RUN_ID_FIELD} IN ({placeholders})", current_chunk_runs)
        label_values += [l[0] for l in cursor.fetchall()]
    label_values = np.array(label_values, dtype=np.float64)
    statistics[label_name] = {'labels_mean': np.mean(label_values),
                        'labels_std' : np.std(label_values)}
with open(MODEL_STATS_PATH, 'w') as f:
    json.dump(statistics, f)

import matplotlib.pyplot as plt
import numpy as np
from neural_spd.config import PLOTS_PATH, IS_HEADLESS

# Pick a sample file for validation
sample_files = list((DATA_PATH / "0" / MODEL_DEM_DIR).glob("*.npy"))
sample_file = sample_files[0]
print(f"Validating with sample file: {sample_file.name}")

# Data types to validate (DEM + derivatives)
data_types = [MODEL_DEM_DIR] + list(TOPO_DERIVATIVES.keys())

# Create plot: rows = data types, columns = noise levels
fig, axes = plt.subplots(len(data_types), len(NOISE_LEVELS), figsize=(6*len(NOISE_LEVELS), 4*len(data_types)))
fig.suptitle('Data Processing Validation: Noise Levels vs Data Types', fontsize=16)

# Handle case where there's only one noise level or one data type
if len(NOISE_LEVELS) == 1:
    axes = axes.reshape(-1, 1)
if len(data_types) == 1:
    axes = axes.reshape(1, -1)

for row_idx, data_type in enumerate(data_types):
    for col_idx, noise_level in enumerate(NOISE_LEVELS):
        noise_str = str(noise_level).replace('.', '-')
        data_file = DATA_PATH / noise_str / data_type / sample_file.name
        
        if data_file.exists():
            data = np.load(data_file)
            
            # Choose colormap based on data type
            if 'flow_accumulation' in data_type:
                cmap = 'Blues'
            elif data_type == 'slope':
                cmap = 'Reds'
            elif data_type == 'curvature':
                cmap = 'RdBu'
            elif data_type == MODEL_DEM_DIR:
                cmap = 'terrain'
            else:
                cmap = 'viridis'
            
            # Plot the data
            im = axes[row_idx, col_idx].imshow(data, cmap=cmap)
            axes[row_idx, col_idx].axis('off')
            
            # Calculate differences if this isn't the first noise level
            if col_idx > 0:
                # Compare with noise level 0
                clean_file = DATA_PATH / "0" / data_type / sample_file.name
                if clean_file.exists():
                    clean_data = np.load(clean_file)
                    diff = data - clean_data
                    
                    mean_diff = np.mean(diff)
                    min_diff = np.min(diff)
                    max_diff = np.max(diff)
                    
                    title = f"{data_type}\nNoise={noise_level}\nΔ: μ={mean_diff:.3f}\nmin={min_diff:.3f}, max={max_diff:.3f}"
                else:
                    title = f"{data_type}\nNoise={noise_level}\n(Clean file missing)"
            else:
                title = f"{data_type}\nNoise={noise_level}\n(Reference)"
            
            axes[row_idx, col_idx].set_title(title, fontsize=10)
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row_idx, col_idx], shrink=0.8, aspect=20)
        
        else:
            axes[row_idx, col_idx].text(0.5, 0.5, f'{data_type}\nNoise={noise_level}\nFILE MISSING', 
                                      ha='center', va='center', transform=axes[row_idx, col_idx].transAxes,
                                      bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            axes[row_idx, col_idx].axis('off')

plt.tight_layout()
if not IS_HEADLESS:
    plt.show()
plt.savefig(PLOTS_PATH / "noise_validation.png")

print("✓ Validation plot complete!")
print("Check that:")
print("  - DEMs look like realistic topography") 
print("  - Slopes look like slope (bright = steep)")
print("  - Curvature shows ridges/valleys (red/blue)")
print("  - Flow accumulation shows drainage networks")
print("  - Noise versions show non-zero differences")
