import os
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import re
from itertools import product

def merge_and_calc_error(directory, test_type=None):
    dfs = []
    nrmses = []
    rmses = []
    datas = []
    count = []
    seeds = []
    targets = []
    noise_levels = []
    values = []
    mean_perc_errs = []
    mean_perc_errs_std = []
    test_types = []
    for f in os.listdir(directory):
        if not f[0]=='n':
            continue
        noise, data_type, seed, target, _ = f.split('_')
        if noise == 'n0':
            noise = 0
        elif noise == 'n0-1':
            noise = 0.1
        df = pd.read_csv(directory / f)
        df['seed'] = [int(seed)]*len(df)
        df['data'] = [data_type]*len(df)
        df['target'] = [target]*len(df)
        df['noise'] = [noise]*len(df)
        df['test_type'] = [test_type]*len(df)
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
        test_types.append(test_type)
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
        'test_type': test_types
    })
    return all_df, rmse_df

from pathlib import Path
all_df, rmse_df = merge_and_calc_error(Path("results"))
analysis_dir = Path("analysis")
analysis_dir.mkdir(exist_ok=True)
rmse_path = analysis_dir / "overall_performance.csv"
rmse_df.to_csv(rmse_path)
test_path = analysis_dir / "all_test_performance.csv"
all_df.to_csv(test_path)

shuffle_all, shuffle_rmse = merge_and_calc_error(Path("results/shuffle/shuffle"), "shuffle")
swap_all, swap_rmse = merge_and_calc_error(Path("results/shuffle/swap"), "swap")
mod_all_df = pd.concat([shuffle_all, swap_all])
mod_rmse_df = pd.concat([shuffle_rmse, swap_rmse])
mod_all_df.to_csv(analysis_dir / "all_mod_performance.csv")
mod_rmse_df.to_csv(analysis_dir / "all_mod_overall_performance.csv")
