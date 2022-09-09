import argparse
import os
import pickle
import pandas as pd
import datetime as dt
import yaml
from typing import List, Optional, Sequence, Tuple, Union

from htm_source.utils.utils import load_models as load_models_htm
from ts_source.utils.utils import (make_dir,
                                    load_darts,
                                    load_config,
                                    get_dirfiles,
                                    get_dir_data,
                                    MODNAMES_OBJTYPES,
                                    save_data_as_pickle,
                                    load_pickle_object_as_data)
from darts.models.forecasting.forecasting_model import ForecastingModel
from darts.models.forecasting.torch_forecasting_model import TorchForecastingModel


ALGS_KNOWN = ['htm', 'dtw', 'edr',
            'VARIMA', 'NBEATSModel', 'TCNModel',
            'TransformerModel', 'RNNModel', 'LightGBMModel' ]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cp', '--config_path', required=True,
                        help='path to config')
    return parser.parse_args()


# def make_dir(mydir):
#     if not os.path.exists(mydir):
#         os.mkdir(mydir)


# def get_dirfiles(dir_root, files_types=None):
#     subfiles = []
#     for path, subdirs, files in os.walk(dir_root):
#         for name in files:
#             subfiles.append(os.path.join(path, name))
#     if files_types is not None:
#         assert isinstance(files_types, list), f"files_types should be a list, found --> {type(files_types)}"
#         subfiles = [s for s in subfiles if s.split('.')[-1] in files_types]
#     return subfiles


# def load_config(yaml_path):
#     """
#     Purpose:
#         Load config from path
#     Inputs:
#         yaml_path
#             type: str
#             meaning: .yaml path to load from
#     Outputs:
#         cfg
#             type: dict
#             meaning: config (yaml) -- loaded
#     """
#     with open(yaml_path, 'r') as yamlfile:
#         cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
#     return cfg


# def save_data_as_pickle(data_struct, f_path):
#     """
#     Purpose:
#         Save data to pkl file
#     Inputs:
#         data_struct
#             type: any
#             meaning: data to save in pkl format
#         f_path
#             type: str
#             meaning: path to pkl file
#     Outputs:
#         True
#             type: bool
#             meaning: function ran
#     """
#     with open(f_path, 'wb') as handle:
#         pickle.dump(data_struct, handle)
#     return True


def validate_config(config):
    print('\nValidating Config...')
    # Ensure expected keys present and correct type
    keys_dtypes = {
        'alg': str,
        'test_mode': str,
        'data_cap': int,
        'hz': int,
        'window_size': int,
        'test_indices': list,
        'features': dict, #list,
        'train_models': bool,
        'dirs': dict,
        'algs_types': dict,
        'colinds_features': dict,
    }
    keys_missing = []
    keys_wrongdtypes = {}
    for k, dtype in keys_dtypes.items():
        if k not in config:
            keys_missing.append(k)
        continue
        if not isinstance(config[k], dtype):
            keys_wrongdtypes[k] = type(config[k])
    assert len(keys_missing) == 0, f"  Expected keys missing --> {sorted(keys_missing)}"
    for k, wrongdtype in keys_wrongdtypes.items():
        print(f"  {k} found type -->{wrongdtype}; expected --> {keys_dtypes[k]}")
    assert len(keys_wrongdtypes) == 0, "  wrong data types"

    # # Ensure LSTM stateful=True if test_mode=online
    # if config['test_mode'] == 'online':
    #     config['lstm_config']['stateful'] = True
    #     config['lstm_config']['shuffle'] = False
    #     config['lstm_config']['batch_size'] = 1
    # else:
    #     config['lstm_config']['stateful'] = False
    #     config['lstm_config']['reset_states'] = False

    # Ensure paths exist
    for dir_type, dir_ in config['dirs'].items():
        if dir_type == 'htm' and config['alg'] != 'htm':
            continue
        if dir_type not in ['models', 'results', 'scalers']:
            assert os.path.exists(dir_), f"{dir_type} not found! --> {dir_}"
        else:
            # make sub-folders for models & results (but skip for distance algs since no models learned)
            if dir_type == 'models' and config['algs_types'][config['alg']] == 'distance':
                continue
            if dir_type == 'output_scaling' and not config['scaling']:
                continue
            # make alg dir
            dir_alg = os.path.join(dir_, f"ALG={config['alg']}")
            make_dir(dir_alg)

            folder_sub = f"HZ={config['hz']};TESTS={config['test_indices']};FEATURES_PRED={config['features']['pred']};SCALING={config['scaling']};GRIDSEARCH={config['do_gridsearch']};"
            if config['algs_types'][config['alg']] == 'distance':
                folder_sub += f"TESTMODE={config['test_mode']};"
                if config['test_mode'] == 'online':
                    folder_sub += f"WINDOW={config['window_size']};"

            dir_sub = os.path.join(dir_alg, folder_sub)
            make_dir(dir_sub)
            config['dirs'][dir_type] = dir_sub

    # Ensure scaling is valid
    scaling_valid = [False, 'minmax', 'standard']
    assert config[
               'scaling'] in scaling_valid, f"scaling should be one of --> {scaling_valid}; found --> {config['scaling']}"

    # Ensure alg is valid
    assert config['alg'] in ALGS_KNOWN, f"alg should be one of --> {ALGS_KNOWN}; found --> {config['alg']}"

    # Ensure test_mode is valid
    modes_valid = ['batch', 'online']
    assert config[
               'test_mode'] in modes_valid, f"test_mode should be one of --> {modes_valid}; found --> {config['test_mode']}"

    # Ensure alg_types is valid
    types_valid = ['prediction', 'anomaly', 'distance']
    for alg, algtype in config['algs_types'].items():
        assert alg in ALGS_KNOWN, f"all algs in algs_types should be in --> {ALGS_KNOWN}; found --> {alg}"
        assert algtype in types_valid, f"all types in algs_types should be in --> {types_valid}; found --> {algtype}"

    # Ensure data_cap >= 100
    assert config['data_cap'] >= 100, f"  data_cap expected >= 1000, found --> {config['data_cap']}"

    # Ensure 1 <= hz <= 100
    assert 1 <= config['hz'] <= 100, f"  hz expected 1 - 100 found --> {config['hz']}"

    # Ensure 1 <= window_size <= 10000
    assert 1 <= config['window_size'] <= 10000, f"  window_size expected 1 - 10000 found --> {config['window_size']}"

    # # Ensure 1 <= time_lag <= 100
    # assert 1 <= config['time_lag'] <= 100, f"  time_lag expected 1 - 100 found --> {config['time_lag']}"

    # # Ensure 100 <= lstm_n_epochs <= 500
    # assert 100 <= config[
    #     'lstm_config'][
    #     'n_epochs'] < 1500, f"  lstm 'n_epochs' expected 100 - 500 found --> {config['lstm_config']['n_epochs']}"

    # # Ensure 1 <= lstm_n_layers <= 5
    # assert 1 <= config[
    #     'lstm_config'][
    #     'n_layers'] < 5, f"  lstm 'n_layers' expected 1 - 5 found --> {config['lstm_config']['n_layers']}"

    # # Ensure 2 <= n_units <= 200
    # assert 1 <= config[
    #     'lstm_config'][
    #     'n_units'] < 200, f"  lstm 'n_units' expected 1 - 200 found --> {config['lstm_config']['n_units']}"

    # # Ensure lstm 'activation' valid
    # valid_activations = ['relu']
    # assert config['lstm_config'][
    #            'activation'] in valid_activations, f"Invalid activation found --> {config['lstm_config']['activation']}\n  valids --> {valid_activations}"

    # # Ensure lstm 'optimizer' valid
    # valid_optimizers = ['adam']
    # assert config['lstm_config'][
    #            'optimizer'] in valid_optimizers, f"Invalid optimizer found --> {config['lstm_config']['optimizer']}\n  valids --> {valid_optimizers}"

    # # Ensure lstm 'loss' valid
    # valid_losses = ['mse', 'mean_squared_error']
    # assert config['lstm_config'][
    #            'loss'] in valid_losses, f"Invalid loss found --> {config['lstm_config']['loss']}\n  valids --> {valid_losses}"

    # Ensure test_indicies range from 1 - 15
    non_ints = []
    ints_over15 = []
    for v in config['test_indices']:
        if not isinstance(v, int):
            non_ints.append(v)
        continue
        if v > 15:
            ints_over15.append(v)
    assert len(non_ints) == 0, f"  Non-integers found in test_indices --> {sorted(non_ints)}"
    assert len(ints_over15) == 0, f"  Integers over 15 found in test_indices --> {sorted(ints_over15)}"

    # Ensure fields in colinds_features
    invalid_fields = []
    for feat_type, feats in config['features'].items(): #for f in config['features']:
        for f in feats:
            if f not in config['colinds_features'].values():
                invalid_fields.append(f)
    assert len(
        invalid_fields) == 0, f"  Invalid fields found --> {sorted(invalid_fields)}; \n  Valids --> {list(config['colinds_features'].values())}"

    print(f"  alg            = {config['alg']}")
    print(f"  train_models   = {config['train_models']}")
    print(f"  hz             = {config['hz']}")
    print(f"  test_indices   = {config['test_indices']}")
    print(f"  features       = {config['features']}")
    print(f"  scaling        = {config['scaling']}")
    if config['algs_types'][config['alg']] == 'distance':
        print(f"  test_mode      = {config['test_mode']}")
        print(f"  window_size    = {config['window_size']}")

    if config['alg'] == 'lstm':
        print(f"  batch_size     = {config['lstm_config']['batch_size']}")
    if config['data_cap'] < 1000:
        print(f"  data_cap       = {config['data_cap']}")
    if config['alg'] == 'htm':
        print(f"  use_sp         = {config['htm_config']['models_state']['use_sp']}")

    return config


def load_models(dir_:str,
                alg:str='htm',
                ftype:Optional[str]='pkl',
                search:Optional[str]='walk',
                **kwargs):
    """
    Purpose:
        Load ftype models for each feature from dir
    Inputs:
        dir_
            type: str
            meaning: path to dir where ftype models are loaded from
    Outputs:
        features_models
            type: dict
            meaning: model obj for each feature
    """
    sub_dirs = [f for f in os.listdir(dir_) if f!='.DS_Store']
    features_models = {}
    for sd in sub_dirs:
        sd_path = os.path.join(dir_, sd)
        if alg == 'htm':
            model = load_models_htm(sd_path) #load_pickle_object_as_data(path_)
        else:  #darts model
            filenames = get_dir_data(dir_=sd_path, ftype=ftype, search=search, rtype='filename')
            filenames_uni = list(set([f.split('.')[0] for f in filenames]))
            filename_model = filenames_uni[0]
            path_ = os.path.join(dir_, sd, f"{filename_model}.{ftype}")
            model = load_darts(path_, filename_model)
        features_models[sd] = model
    return features_models


# def load_darts(path_, darts_modeltype):
#     model = MODNAMES_OBJTYPES[darts_modeltype].load(path_)
#     return model


# def load_pickle_object_as_data(file_path):
#     """
#     Purpose:
#         Load data from pkl file
#     Inputs:
#         file_path
#             type: str
#             meaning: path to pkl file
#     Outputs:
#         data
#             type: pkl
#             meaning: pkl data loaded
#     """
#     with open(file_path, 'rb') as f_handle:
#         data = pickle.load(f_handle)
#     return data


def sort_dict(dict_, by='value', ascending=False):
    item_i = 1 if by=='value' else 0
    dict_sorted = {k: v for k, v in sorted(dict_.items(),
                                            key=lambda item: item[item_i],
                                            reverse=ascending)}
    return dict_sorted