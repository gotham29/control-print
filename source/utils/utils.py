import argparse
import os
import pickle

import yaml


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-cp', '--config_path', required=True,
                        help='path to config')
    return parser.parse_args()


def make_dir(mydir):
    if not os.path.exists(mydir):
        os.mkdir(mydir)


def get_dirfiles(dir_root, files_types=None):
    subfiles = []
    for path, subdirs, files in os.walk(dir_root):
        for name in files:
            subfiles.append(os.path.join(path, name))
    if files_types is not None:
        assert isinstance(files_types, list), f"files_types should be a list, found --> {type(files_types)}"
        subfiles = [s for s in subfiles if s.split('.')[-1] in files_types]
    return subfiles


def load_config(yaml_path):
    """
    Purpose:
        Load config from path
    Inputs:
        yaml_path
            type: str
            meaning: .yaml path to load from
    Outputs:
        cfg
            type: dict
            meaning: config (yaml) -- loaded
    """
    with open(yaml_path, 'r') as yamlfile:
        cfg = yaml.load(yamlfile, Loader=yaml.FullLoader)
    return cfg


def save_data_as_pickle(data_struct, f_path):
    """
    Purpose:
        Save data to pkl file
    Inputs:
        data_struct
            type: any
            meaning: data to save in pkl format
        f_path
            type: str
            meaning: path to pkl file
    Outputs:
        True
            type: bool
            meaning: function ran
    """
    with open(f_path, 'wb') as handle:
        pickle.dump(data_struct, handle)
    return True


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
        'features': list,
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

    # Ensure paths exist
    for dir_type, dir in config['dirs'].items():
        if dir_type == 'htm' and config['alg'] != 'htm':
            continue
        if dir_type not in ['output_models', 'output_results']:
            assert os.path.exists(dir), f"{dir_type} not found! --> {dir}"
        else:
            # make sub-folders for models & results (but skip for distance algs since no models learned)
            if dir_type == 'output_models' and config['algs_types'][config['alg']] == 'distance':
                continue
            # make alg dir
            dir_alg = os.path.join(dir, f"ALG={config['alg']}")
            make_dir(dir_alg)
            # make sub-dir for test_model (for results, models not needed since always batch trained)
            if dir_type == 'output_results':
                dir_alg = os.path.join(dir_alg, f"TESTMODE={config['test_mode']}")
                make_dir(dir_alg)
            # make sub-sub-dir for hz-test_indices-features combo
            subfolder = f"HZ={config['hz']};TESTS={config['test_indices']};FEATURES={config['features']}"
            dir_sub = os.path.join(dir_alg, subfolder)
            make_dir(dir_sub)
            # make sub-sub-sub-dir for window_size IF online test_mode AND not HTM
            if dir_type == 'output_results' and config['test_mode'] == 'online' and config['alg'] != 'htm':
                dir_sub = os.path.join(dir_sub, f"WINDOW={config['window_size']}")
                make_dir(dir_sub)
            config['dirs'][dir_type] = dir_sub

    # Ensure alg is valid
    algs_valid = ['lstm', 'htm', 'dtw', 'edr']
    assert config['alg'] in algs_valid, f"alg should be one of --> {algs_valid}; found --> {config['alg']}"

    # Ensure test_mode is valid
    modes_valid = ['batch', 'online']
    assert config[
               'test_mode'] in modes_valid, f"test_mode should be one of --> {modes_valid}; found --> {config['test_mode']}"

    # Ensure alg_types is valid
    types_valid = ['prediction', 'anomaly', 'distance']
    for alg, algtype in config['algs_types'].items():
        assert alg in algs_valid, f"all algs in algs_types should be in --> {algs_valid}; found --> {alg}"
        assert algtype in types_valid, f"all types in algs_types should be in --> {types_valid}; found --> {algtype}"

    # Ensure data_cap >= 100
    assert config['data_cap'] >= 100, f"  data_cap expected >= 1000, found --> {config['data_cap']}"

    # Ensure 1 <= hz <= 100
    assert 1 <= config['hz'] <= 100, f"  hz expected 1 - 100 found --> {config['hz']}"

    # Ensure 1 <= window_size <= 10000
    assert 1 <= config['window_size'] <= 10000, f"  window_size expected 1 - 10000 found --> {config['window_size']}"

    # Ensure 100 <= lstm_n_epochs <= 500
    assert 100 <= config[
        'lstm_config'][
        'n_epochs'] < 500, f"  lstm 'n_epochs' expected 100 - 500 found --> {config['lstm_config']['n_epochs']}"

    # Ensure 1 <= lstm_n_layers <= 5
    assert 1 <= config[
        'lstm_config'][
        'n_layers'] < 5, f"  lstm 'n_layers' expected 1 - 5 found --> {config['lstm_config']['n_layers']}"

    # Ensure 2 <= n_units <= 200
    assert 2 <= config[
        'lstm_config'][
        'n_units'] < 200, f"  lstm 'n_units' expected 2 - 200 found --> {config['lstm_config']['n_units']}"

    # Ensure lstm 'activation' valid
    valid_activations = ['relu']
    assert config['lstm_config'][
               'activation'] in valid_activations, f"Invalid activation found --> {config['lstm_config']['activation']}\n  valids --> {valid_activations}"

    # Ensure lstm 'optimizer' valid
    valid_optimizers = ['adam']
    assert config['lstm_config'][
               'optimizer'] in valid_optimizers, f"Invalid optimizer found --> {config['lstm_config']['optimizer']}\n  valids --> {valid_optimizers}"

    # Ensure lstm 'loss' valid
    valid_losses = ['mse']
    assert config['lstm_config'][
               'loss'] in valid_losses, f"Invalid loss found --> {config['lstm_config']['loss']}\n  valids --> {valid_losses}"

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
    for f in config['features']:
        if f not in config['colinds_features'].values():
            invalid_fields.append(f)
    assert len(
        invalid_fields) == 0, f"  Invalid fields found --> {sorted(invalid_fields)}; \n  Valids --> {list(config['colinds_features'].values())}"

    print(f"  alg = {config['alg']}")
    print(f"  train_models = {config['train_models']}")
    print(f"  test_mode = {config['test_mode']}")
    print(f"  window_size = {config['window_size']}")
    print(f"  hz = {config['hz']}")
    print(f"  data_cap = {config['data_cap']}")
    print(f"  features = {config['features']}")
    print(f"  test_indices = {config['test_indices']}")


def load_models(dir_models):
    """
    Purpose:
        Load pkl models for each feature from dir
    Inputs:
        dir_models
            type: str
            meaning: path to dir where pkl models are loaded from
    Outputs:
        features_models
            type: dict
            meaning: model obj for each feature
    """
    pkl_files = [f for f in os.listdir(dir_models) if '.pkl' in f]
    print(f"\nLoading {len(pkl_files)} models...")
    features_models = {}
    for f in pkl_files:
        pkl_path = os.path.join(dir_models, f)
        model = load_pickle_object_as_data(pkl_path)
        features_models[f.replace('.pkl', '')] = model
    print('  DONE')
    return features_models


def load_pickle_object_as_data(file_path):
    """
    Purpose:
        Load data from pkl file
    Inputs:
        file_path
            type: str
            meaning: path to pkl file
    Outputs:
        data
            type: pkl
            meaning: pkl data loaded
    """
    with open(file_path, 'rb') as f_handle:
        data = pickle.load(f_handle)
    return data
