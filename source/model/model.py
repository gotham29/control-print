import numpy as np
import os
import sys
import time
import pandas as pd
import datetime as dt
from pandas import Series, concat

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.model.lstm import train_lstm, compile_lstm, forecast_lstm
from source.model.dtw import get_dtw_dist
from source.model.edr import get_edr_dist
from source.model.htm import train_save_htm_models, get_htm_dist
from source.model.arima import train_arima
from source.utils.utils import make_dir, save_data_as_pickle, load_pickle_object_as_data, add_timecol, sort_dict
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from numpy import array

from ts_source.pipeline.pipeline import run_pipeline
from ts_source.model.model import get_preds_rolling, get_model_lag, MODNAMES_LAGPARAMS, MODNAMES_MODELS


"""
def get_preds_online(test_x, model, batch_size, n_features):
    predictions = []
    for x in test_x:
        yhat = forecast_lstm(model, batch_size, x)
        # # invert scaling
        # yhat = invert_scale(scaler, X, yhat)
        # # invert differencing
        # yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        predictions.append(yhat)
    return array(predictions)
"""

def get_modname(model):
    mod_name = None
    for modname, mod in MODNAMES_MODELS.items():
        if isinstance(model, mod):
            mod_name = modname
    return mod_name


def get_models_preds(config, subjects_traintest, subjects_models, features, dir_scalers, scale=False, LAG_MIN=3):
    subjstest_subjspreds = {}
    for subjtest, traintest in subjects_traintest.items():
        subjstest_subjspreds[subjtest] = {}
        """
        # Prep data
        test_x, test_y, test_x_conv = prep_data(traintest['test'][features], time_lag, len(features) )
        # Scale data
        if scale:
            path_scaler = os.path.join(dir_scalers, f"{subjtest}.pkl")
            scaler = load_pickle_object_as_data(path_scaler)
            test_x = unscale_data(test_x, scaler)
        """
        # Get preds
        for subjmod, model in subjects_models.items():
            """
            if test_mode == 'batch':
                preds = mod.predict(test_x_conv, verbose=0)
            else:  # 'online'
                preds = get_preds_online(test_x_conv, mod, config['lstm_config']['batch_size'], len(config['features']))
            if scale:
                preds = unscale_data(preds, scaler)
            """
            mod_name = get_modname(model)
            features = model.training_series.components
            preds = get_preds_rolling(model=model,
                                        df=traintest['test'],
                                        features=features,
                                        LAG=max(LAG_MIN, get_model_lag(mod_name, model)),
                                        time_col=config['time_col'],
                                        forecast_horizon=config['forecast_horizon'])
            subjstest_subjspreds[subjtest][subjmod] = pd.DataFrame(preds, columns=list(features)) #modnames_preds[ config['alg'] ] #preds
    return subjstest_subjspreds


def get_windowed_data(data, window_size):
    rows = []
    for _ in range(1, len(data) + 1):
        window_adj = min(_, window_size)
        d = data[_ - window_adj:_]
        rows.append(d)
    return array(rows)


def get_models_dists_pred(subjstest_subjspreds, subjects_traintest, features):
    # Get dists (pred erres) between all subjs
    subjstest_subjsdists = {}
    for subjtest, subjspreds in subjstest_subjspreds.items():
        subjstest_subjsdists[subjtest] = {}
        """
        x_true, y_true, true_x_conv = prep_data(subjects_traintest[subjtest]['test'][features], time_lag, len(features))
        """
        subjtest_true = subjects_traintest[subjtest]['test']
        for subjpred, preds in subjspreds.items():
            subjtest_true_adj = subjtest_true.tail(len(preds))
            """
            if test_mode == 'batch':
                subjstest_subjsdists[subjtest][subjpred] = get_diff(subjtest_true_adj[features].values, preds[features].values)  #y_true, preds
            else:
                subjstest_subjsdists[subjtest][subjpred] = get_diff_online(subjtest_true_adj[features].values, preds[features].values, window_size)  #y_true, preds, window_size
            """
            subjstest_subjsdists[subjtest][subjpred] = get_diff(subjtest_true_adj[features].values,
                                                                preds[features].values)
    return subjstest_subjsdists

"""
def get_diff_online(true, pred, window_size):
    true = get_windowed_data(true, window_size)
    dist = 0
    for _ in range(len(pred)):  #true
        dist += get_diff(true[_], pred[_])
    return dist
"""

def get_diff(true, pred):
    return abs(true - pred).sum()


def get_models_dists_dist(subjects_traintest, alg, window_size, features, test_mode):
    feats_model = []
    for ftype, feats in features.items():
        feats_model += feats
    feats_model = list(set(feats_model))
    subjstest_subjsdists = {}
    for subjtest1, traintest in subjects_traintest.items():
        subjstest_subjsdists[subjtest1] = {}
        data1 = traintest['train'][feats_model]
        for subjtest2, traintest in subjects_traintest.items():
            data2 = traintest['test'][feats_model]
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


def get_models_anomscores(subjects_traintest, subjects_models):  #features
    subjstest_subjsanoms = {}
    for subjtest1, mod_dict in subjects_models.items():
        subjstest_subjsanoms[subjtest1] = {}
        for subjtest2, traintest in subjects_traintest.items():
            subjstest_subjsanoms[subjtest1][subjtest2] = get_htm_dist(mod_dict, traintest['test'])
    return subjstest_subjsanoms


def difference_df(df, lag):
    df_dict = {}
    for c in df:
        df_dict[f"{c} (lag={lag})"] = difference(df[c].values, lag)
    return pd.DataFrame(df_dict)


def difference(dataset, lag):
    diff = list()
    for i in range(lag, len(dataset)):
        value = dataset[i] - dataset[i - lag]
        diff.append(value)
    return Series(diff)


def timeseries_to_supervised(df, lag=1):
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = concat(columns, axis=1)
    return df


# def prep_data(df_y, time_lag, n_features):
#     """ Trainsform to --> stationary """
#     df_y = df_y.reset_index(inplace=False)
#     df_y = df_y.drop(columns=['index'], inplace=False)
#     timelags_dfdiffs = {}
#     # COLLECT TIME_LAG DFS
#     for tl in range(1, time_lag+1):
#         df_diff = difference_df(df=df_y, lag=tl)  #time_lag
#         df_nan = pd.DataFrame( {c:[np.nan for _ in range(tl)] for c in df_diff} )
#         df_diff = pd.concat([df_nan, df_diff], axis=0)
#         df_diff = df_diff.reset_index(inplace=False)
#         df_diff = df_diff.drop(columns=['index'], inplace=False)
#         timelags_dfdiffs[tl] = df_diff
#     df_diffs = list(timelags_dfdiffs.values())
#     # CONCAT TIME_LAG DFS
#     df_x = pd.concat(df_diffs, axis=1)
#     df_xy = pd.concat([df_x, df_y], axis=1)
#     # DROP NA
#     df_x = df_x.dropna(how='any', axis=0, inplace=False)
#     df_y = df_y.dropna(how='any', axis=0, inplace=False)
#     df_xy = df_xy.dropna(how='any', axis=0, inplace=False)
#     # RESHAPE
#     df_x_conv = array(df_x).reshape(df_x.shape[0], time_lag, n_features)  #df_x.shape[1]
#     return array(df_x), array(df_y), df_x_conv


def train_save_pred_models(subjects_traintest, config, alg):  #time_lag=1
    print(f"\nTraining {len(subjects_traintest)} {alg.upper()} models...")
    make_dir(config['dirs']['models'])
    subjects_models = {}
    features = []
    for ftype, fs in config['features'].items():
        features += [f for f in fs if f not in features]
    n_features = len(features)  #len(config['features'])
    counter, time_start = 1, time.time()
    for subj, traintest in subjects_traintest.items():
        print(f"  subj = {subj}")
        # train = traintest['train'][features]  #[config['features']]
        # train_x, train_y, train_x_conv = prep_data(train, time_lag, n_features)

        """
        print(f"    train_x_conv SHAPE = {train_x_conv.shape}")
        # Fit model
        if alg == 'lstm':
            model = compile_lstm(train_x_conv, config['lstm_config'], time_lag, n_features)
            model = train_lstm(train_x_conv, train_y, config['data_cap'], config['lstm_config'], model)
        # Save model & scaler
        subjects_models[subj] = model
        path_mod = os.path.join(config['dirs']['models'], f"{subj}.pkl")
        save_data_as_pickle(model, path_mod)
        if config['scaling']:
            path_scaler = os.path.join(config['dirs']['scalers'], f"{subj}.pkl")
            save_data_as_pickle(scaler, path_scaler)
        """
        # Train model (ts_source)
        config_ts = {k:v for k,v in config.items()}
        config_ts['train_models'] = True
        config_ts['modnames_grids'] = {k:v for k,v in config_ts['modnames_grids'].items() if k == alg}
        # if config_ts['time_col'] not in traintest['train']:
        #     traintest['train'] = add_timecol(traintest['train'], config_ts['time_col'])
        #     traintest['test'] = add_timecol(traintest['test'], config_ts['time_col'])

        output_dirs = {'data': os.path.join(config['dirs']['data'], subj),
                        'results': os.path.join(config['dirs']['results'], subj),
                        'models': os.path.join(config['dirs']['models'], subj),
                        'scalers': os.path.join(config['dirs']['scalers'], subj)}

        if os.path.exists(output_dirs['models']):
            print("  --> found")
            continue

        modnames_models, modname_best, modnames_preds       = run_pipeline(config=config_ts,
                                                                            data=traintest['train'],
                                                                            data_path=False,
                                                                            output_dir=False,
                                                                            output_dirs=output_dirs)
        subjects_models[subj] = modnames_models[modname_best]

        # Track time
        time_elapsed_mins = round((time.time() - time_start) / 60, 2)
        print(f"    Trained {counter} of {len(subjects_traintest)} models; elapsed minutes = {time_elapsed_mins}")
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
