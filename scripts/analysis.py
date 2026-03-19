import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
from itertools import product

d1 = Path("results")
dfs = []
nrmses = []
rmses = []
seeds = []
datas = []
count = []
targets = []
noise_levels = []
values = []
mean_perc_errs = []
mean_perc_errs_std = []

d1_files = list(os.listdir(d1))
files = d1_files 
ds = [d1]*len(d1_files)
for i, f in enumerate(files):
    d = ds[i]
    if not f[0]=='n':
        continue
    noise, data_type, seed, target, _ = f.split('_')
    if noise == 'n0':
        noise = 0
    elif noise == 'n0-1':
        noise = 0.1
    df = pd.read_csv(d / f)
    df['seed'] = [int(seed)]*len(df)
    df['data'] = [data_type]*len(df)
    df['target'] = [target]*len(df)
    df['noise'] = [noise]*len(df)
    dfs.append(df)
    perc_err = np.abs(df['true_labels'] - df['predictions'])/df['true_labels']
    mean_perc_err = np.mean(perc_err)
    mean_perc_err_std = np.std(perc_err)
    mean_perc_errs.append(mean_perc_err)
    mean_perc_errs_std.append(mean_perc_err_std)
    rmse = root_mean_squared_error(df['true_labels'], df['predictions'])
    rmses.append(rmse)
    nrmse = rmse/(np.ptp(df['true_labels']))
    nrmses.append(nrmse)
    seeds.append(seed)
    datas.append(data_type)
    targets.append(target)
    noise_levels.append(noise)

all_df = pd.concat(dfs)
rmse_df = pd.DataFrame({
    'data': datas,
    'seed': seeds,
    'target': targets,
    'noise': noise_levels,
    'nrmse': nrmses,
    'rmse': rmses,
    'mean_perc_err': mean_perc_errs,
    'mean_perc_err_std': mean_perc_errs_std,
})
# label_map = {'dem': 'Elevation',
#          'slope': 'Slope',
#          'curvature': 'Curvature',
#          'accumulation': 'Flow Accumulation',
#          'logaccumulation': '$\\log_{10}(\\text{Flow Accumulation})$',
#          'dem_dem': 'Elevation, Elevation',
#          'dem_slope': 'Elevation, Slope',
#          'dem_curvature': 'Elevation, Curvature',
#          'dem_accumulation': 'Elevation, Flow Accumulation'}
# rmse_df['labels'] = rmse_df['data'].map(label_map)
# all_df['labels'] = all_df['data'].map(label_map)
# dem_1part = rmse_df[(rmse_df['data'] == 'dem') & (rmse_df['parts'] == 1)]['nrmse'].mean()
# dem_2part = rmse_df[(rmse_df['data'] == 'dem_dem') & (rmse_df['parts'] == 2)]['nrmse'].mean()

# Calculate percentage improvements for all models
dem_baseline = rmse_df[(rmse_df['data'] == 'dem') & (rmse_df['target']=="DoK")]['nrmse'].mean()
for index, row in rmse_df.iterrows():
    nrmse = row['nrmse']
    row['improvement'] = (dem_baseline-nrmse)/dem_baseline*100

from pathlib import Path
analysis_dir = Path("analysis")
analysis_dir.mkdir(exist_ok=True)
rmse_path = analysis_dir / "overall_performance.csv"
rmse_df.to_csv(rmse_path)
test_path = analysis_dir / "all_test_performance.csv"
all_df.to_csv(test_path)
