import os
import sys
import numpy as np
import pandas as pd
import datetime as dt
from os.path import isfile, join

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from ts_source.utils.utils import add_timecol

def get_dists(row):
    return np.sqrt(row['dists_x'] ** 2) + np.sqrt(row['dists_y'] ** 2) + np.sqrt(row['dists_z'] ** 2)


def preprocess(dir_input, dir_output, test_indices, hz, time_col, colinds_colnames):
    print('\nPreprocessing data...')

    # get agg from hz
    agg = int(100 / hz)
    # load & store raw files by subject
    raw_csv_files = [os.path.join(dir_input, f) for f in os.listdir(dir_input)
                     if isfile(join(dir_input, f))
                     and os.path.splitext(f)[1] == '.csv']
    print(f'  loading raw data ({len(raw_csv_files)} subjects)...')
    subjects_raws = {}
    for f in raw_csv_files:
        subj = f.split('/')[-1].replace('.csv', '')
        subjects_raws[subj] = pd.read_csv(f)

    print('  splitting runs...')
    subjects_runs = {}
    for subj, data in subjects_raws.items():
        # get indices dividing runs (those w/zeros)
        cols_to_sum = [c for c in data.columns if c != '0']
        data['sum'] = data[cols_to_sum].sum(axis=1)
        zero_indices = data.index[data['sum'] == 0].tolist()
        # add final divider for last run (since it has no zero row)
        zero_indices.append(data.shape[0])
        # split data by run
        runinds_rundata = {}
        start_ind = 0
        for run_i, zero_ind in enumerate(zero_indices):
            # rename columns
            run_data = data[start_ind:zero_ind].rename(colinds_colnames, axis=1).astype(float)
            # add 'dists' column from x/y/z dists
            run_data['dists'] = run_data.apply(lambda row: get_dists(row), axis=1)
            # agg to desired granularity (Hz)
            run_data = run_data.groupby(run_data.index // agg).mean()

            runinds_rundata[run_i + 1] = run_data
            start_ind = zero_ind
        subjects_runs[subj] = runinds_rundata

    # split train/test
    print('  splitting train/test...')
    subjects_traintest = {}
    for subj, inds_data in subjects_runs.items():
        inds_data_train = {k: v for k, v in inds_data.items() if k not in test_indices}
        inds_data_test = {k: v for k, v in inds_data.items() if k in test_indices}
        subjects_traintest[subj] = {
            'train': pd.concat(inds_data_train.values(), axis=0),
            'test': pd.concat(inds_data_test.values(), axis=0),
        }

    print('  writing csvs...')
    for subj, traintest in subjects_traintest.items():
        # add timecol if needed
        if time_col not in traintest['train']:
            traintest['train'] = add_timecol(traintest['train'], time_col)
            traintest['test'] = add_timecol(traintest['test'], time_col)
        # write out
        path_train = os.path.join(dir_output, f"subj={subj}_train.csv")
        path_test = os.path.join(dir_output, f"subj={subj}_test.csv")

    return subjects_traintest


""" TEST """
# dir_input = "/Users/samheiserman/Desktop/PhD/Motion-Print/data/subjects_raw"
# dir_output = "/Users/samheiserman/Desktop/PhD/Motion-Print/data/subjects_preprocessed"
# test_indices = [1, 2, 3]
#
# subjects_traintest = preprocess(dir_input, dir_output, test_indices)
