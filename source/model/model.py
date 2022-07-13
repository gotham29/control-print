import numpy as np
import os
import sys
import time

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.model.lstm import train_lstm_batch, compile_lstm
from source.model.dtw import get_dtw_dist
from source.model.edr import get_edr_dist
from source.model.htm import train_save_htm_models, get_htm_dist
from source.model.arima import train_arima
from source.utils.utils import make_dir, save_data_as_pickle, load_pickle_object_as_data
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from numpy import array


def get_preds_online(X, model, window_size, n_steps=1):
    preds = []
    for _ in range(1, len(X) + 1):
        window_adj = min(_, window_size)
        x = X[_ - window_adj:_]
        y_hat = np.round_(model.predict_on_batch(x))  # predict on the "new" input
        preds.append(y_hat)
    return array(preds)


def get_models_preds(subjects_traintest, subjects_models, test_mode, window_size, features, dir_scalers, scale=False,
                     n_steps=1):
    subjstest_subjspreds = {}
    for subjtest, traintest in subjects_traintest.items():
        subjstest_subjspreds[subjtest] = {}
        X_array = array(traintest['test'][features])
        X_array = X_array[:len(X_array) - 1]

        """ SCALE DATA HERE """
        if scale:
            path_scaler = os.path.join(dir_scalers, f"{subjtest}.pkl")
            scaler = load_pickle_object_as_data(path_scaler)
            X_array = unscale_data(X_array, scaler)

        X = X_array.reshape((X_array.shape[0], n_steps, X_array.shape[1]))
        for subjmod, mod in subjects_models.items():
            if test_mode == 'batch':
                preds = mod.predict(X)
            else:
                preds = get_preds_online(X, mod, window_size)
            if scale:
                preds = unscale_data(preds, scaler)
            subjstest_subjspreds[subjtest][subjmod] = preds
    return subjstest_subjspreds


def get_windowed_data(data, window_size):
    rows = []
    for _ in range(1, len(data) + 1):
        window_adj = min(_, window_size)
        d = data[_ - window_adj:_]
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
    for _ in range(1, len(data2) + 1):
        window_adj = min(_, window_size)
        data2_ = data2[_ - window_adj:_]
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


def get_models_anomscores(subjects_traintest, subjects_models, features):
    subjstest_subjsanoms = {}
    for subjtest1, mod_dict in subjects_models.items():
        subjstest_subjsanoms[subjtest1] = {}
        for subjtest2, traintest in subjects_traintest.items():
            subjstest_subjsanoms[subjtest1][subjtest2] = get_htm_dist(mod_dict, traintest['test'])
    return subjstest_subjsanoms


def train_save_pred_models(subjects_traintest, config, alg, n_steps=1):
    print(f"\nTraining {len(subjects_traintest)} {alg.upper()} models...")

    make_dir(config['dirs']['output_models'])
    subjects_models = {}
    n_features = len(config['features'])
    counter, time_start = 1, time.time()

    for subj, traintest in subjects_traintest.items():

        X_array = array(traintest['train'][config['features']])  # BUG FIXED --> test
        y_array = array(traintest['train'][config['features']].shift(-1))  # BUG FIXED -->test
        # drop last row since NaN for y
        y_array = y_array[:len(y_array) - 1]
        X_array = X_array[:len(X_array) - 1]

        """ SCALE DATA HERE """
        if config['scaling']:
            X_array, scaler = scale_data(data=X_array, method=config['scaling'])
            y_array, scaler = scale_data(data=y_array, method=config['scaling'])

        X_conv = X_array.reshape((X_array.shape[0], n_steps, X_array.shape[1]))  # 1-->window_size

        # fit model -- CAN TRAIN IN BATCH BUT TEST IN ONLINE
        if alg == 'lstm':
            model = compile_lstm(config['lstm_config'], n_steps, n_features)
            model = train_lstm_batch(X_conv, y_array, config['data_cap'], config['lstm_config'], model)
        elif alg == 'arima':
            model = train_arima(X_array, config['arima_config'], config['data_cap'])

        # save model & scaler
        subjects_models[subj] = model
        path_mod = os.path.join(config['dirs']['output_models'], f"{subj}.pkl")
        save_data_as_pickle(model, path_mod)
        if config['scaling']:
            path_scaler = os.path.join(config['dirs']['output_scalers'], f"{subj}.pkl")
            save_data_as_pickle(scaler, path_scaler)

        # track time
        time_elapsed_mins = round((time.time() - time_start) / 60, 2)
        print(f"  Trained {counter} of {len(subjects_traintest)} models; elapsed minutes = {time_elapsed_mins}")
        counter += 1
    return subjects_models


def scale_data(data, method, scaler=None):
    if scaler is None:
        if method == 'minmax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()
        scaler = scaler.fit(data)
    scaled = scaler.transform(data)
    return scaled, scaler


def unscale_data(data, scaler):
    return scaler.inverse_transform(data)
