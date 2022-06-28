import numpy as np
import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.model.lstm import train_lstm
from source.model.dtw import get_dtw_dist
from source.model.edr import get_edr_dist
from source.model.htm import train_htm

from numpy import array


def get_preds_online(X, model, window_size, n_steps=1):
    preds = []
    for _ in range(1, len(X)+1):
        window_adj = min(_, window_size)
        x = X[_-window_adj:_]
        y_hat = np.round_(model.predict_on_batch(x))  # predict on the "new" input
        preds.append(y_hat)
    return array(preds)


def train_save_models(subjects_traintest, config):
    if config['alg'] == 'lstm':
        subjects_models = train_lstm(subjects_traintest=subjects_traintest,
                                     window_size=config['window_size'],
                                     features=config['features'],
                                     data_cap=config['data_cap'],
                                     dir_models=config['dirs']['output_models'],
                                     n_epochs=config['lstm_n_epochs'])

    elif config['alg'] == 'htm':
        subjects_models = train_htm(subjects_traintest, config, config['dirs']['htm_config'])

    else:
        raise ValueError("train_models called with alg neither lstm or htm, found --> {alg}")
        print(f"\nNo models trained, since {alg} is a distance metric and not model-based")

    return subjects_models


def get_models_preds(subjects_traintest, subjects_models, test_mode, window_size, features, n_steps=1):
    subjstest_subjspreds = {}
    for subjtest, traintest in subjects_traintest.items():
        subjstest_subjspreds[subjtest] = {}
        X_array = array(traintest['test'][features])
        X_array = X_array[:len(X_array) - 1]
        X = X_array.reshape((X_array.shape[0], n_steps, X_array.shape[1]))
        for subjmod, mod in subjects_models.items():
            if test_mode == 'batch':
                subjstest_subjspreds[subjtest][subjmod] = mod.predict(X)
            else:
                subjstest_subjspreds[subjtest][subjmod] = get_preds_online(X, mod, window_size)

    return subjstest_subjspreds


def get_windowed_data(data, window_size):
    rows = []
    for _ in range(1, len(data)+1):
        window_adj = min(_, window_size)
        d = data[_-window_adj:_]
        rows.append(d)
    return array(rows)


def get_models_dists_pred(subjstest_subjspreds, subjects_traintest, features, test_mode, window_size):
    subjstest_subjsdists = {}
    for subjtest, subjspreds in subjstest_subjspreds.items():
        subjstest_subjsdists[subjtest] = {}
        # Get y_true by shifting test 1 time step
        y_true = array(subjects_traintest[subjtest]['test'][features].shift(-1))
        # Drop last row (since its NaN after shift)
        y_true = y_true[:len(y_true) - 1]
        # Calc dist
        for subjpred, preds in subjspreds.items():
            if test_mode == 'batch':
                subjstest_subjsdists[subjtest][subjpred] = get_diff(y_true, preds)
            else:
                subjstest_subjsdists[subjtest][subjpred] = get_diff_online(y_true, preds, window_size)
    return subjstest_subjsdists


def get_diff_online(y_true, preds, window_size):
    y_true = get_windowed_data(y_true, window_size)
    dist = 0
    for _ in range(len(y_true)):
        dist += get_diff(y_true[_], preds[_])
    return dist


def get_diff(y_true, preds):
    return abs(y_true - preds).sum()


def get_models_dists_dist(subjects_traintest, alg, window_size, features, test_mode):
    subjstest_subjsdists = {}
    for subjtest1, traintest in subjects_traintest.items():
        subjstest_subjsdists[subjtest1] = {}
        data1 = traintest['train'][features]
        for subjtest2, traintest in subjects_traintest.items():
            data2 = traintest['test'][features]
            if test_mode == 'batch':
                subjstest_subjsdists[subjtest1][subjtest2] = get_dist(data1, data2, alg)
            else:
                subjstest_subjsdists[subjtest1][subjtest2] = get_dist_online(data1, data2, window_size, alg)

    return subjstest_subjsdists


def get_dist_online(data1, data2, window_size, alg):
    dist = 0
    for _ in range(1, len(data2)+1):
        window_adj = min(_, window_size)
        data2_ = data2[_-window_adj:_]
        dist += get_dist(data1, data2_, alg)
    return dist


def get_dist(data1, data2, alg):
    algs_valid = ['dtw', 'edr']
    assert alg in algs_valid, f"Expected distance-based alg in -- {algs_valid}; found --> {alg}"
    if alg == 'dtw':
        dist = get_dtw_dist(data1.values, data2.values)
    else:  # alg = 'edr'
        dist = get_edr_dist(data1, data2)
    return dist
