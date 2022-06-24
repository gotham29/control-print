import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.model.lstm import train_lstm
from source.model.dtw import get_dtw_dist
from source.model.edr import get_edr_dist
from source.model.htm import train_htm

from numpy import array


def train_save_models(subjects_traintest, config):
    if config['alg'] == 'lstm':
        subjects_models = train_lstm(subjects_traintest, config['features'], config['data_cap'],
                                     config['dirs']['output_models'], n_steps=1,
                                     n_epochs=config['lstm_n_epochs'])
    elif config['alg'] == 'htm':
        subjects_models = train_htm(subjects_traintest, config, config['dirs']['htm_config'])

    else:
        raise ValueError("train_models called with alg neither lstm or htm, found --> {alg}")
        print(f"\nNo models trained, since {alg} is a distance metric and not model-based")

    return subjects_models


def get_models_preds(subjects_traintest, subjects_models, features):
    subjstest_subjspreds = {}
    for subjtest, traintest in subjects_traintest.items():
        subjstest_subjspreds[subjtest] = {}
        X_array = array(traintest['test'][features])
        X_array = X_array[:len(X_array) - 1]
        X = X_array.reshape((X_array.shape[0], 1, X_array.shape[1]))
        for subjmod, mod in subjects_models.items():
            subjstest_subjspreds[subjtest][subjmod] = mod.predict(X)
    return subjstest_subjspreds


def get_models_dists_pred(subjstest_subjspreds, subjects_traintest, features):
    subjstest_subjsdists = {}
    for subjtest, subjspreds in subjstest_subjspreds.items():
        subjstest_subjsdists[subjtest] = {}
        # Get y_true by shifting test 1 time step
        y_true = array(subjects_traintest[subjtest]['test'][features].shift(-1))
        # Drop last row (since its NaN after shift)
        y_true = y_true[:len(y_true) - 1]
        for subjpred, preds in subjspreds.items():
            subjstest_subjsdists[subjtest][subjpred] = abs(y_true - preds).sum()
    return subjstest_subjsdists


def get_models_dists_dist(subjects_traintest, alg, features):
    subjstest_subjsdists = {}
    for subjtest1, traintest in subjects_traintest.items():
        subjstest_subjsdists[subjtest1] = {}
        data1 = traintest['train'][features]
        for subjtest2, traintest in subjects_traintest.items():
            data2 = traintest['test'][features]
            subjstest_subjsdists[subjtest1][subjtest2] = get_dist(data1, data2, alg)
    return subjstest_subjsdists


def get_dist(data1, data2, alg):
    algs_valid = ['dtw', 'edr']
    assert alg in algs_valid, f"Expected distance-based alg in -- {algs_valid}; found --> {alg}"
    if alg == 'dtw':
        dist = get_dtw_dist(data1, data2)
    else:  # alg = 'edr'
        dist = get_edr_dist(data1, data2)
    return dist