import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.preprocess.preprocess import preprocess
from source.analyze.rank_score import write_rankscores
from source.model.model import train_save_pred_models, get_models_preds, get_models_dists_pred, get_models_dists_dist, \
    get_models_anomscores
from source.model.htm import train_save_htm_models
from source.utils.utils import load_config, load_models, validate_config, get_args


def run_pipeline(config):
    # VALIDATE CONFIG
    validate_config(config=config)

    # PREPROCESS DATA
    subjects_traintest = preprocess(dir_input=config['dirs']['input'],
                                    dir_output=config['dirs']['output_data'],
                                    test_indices=config['test_indices'],
                                    hz=config['hz'],
                                    colinds_colnames=config['colinds_features'])

    # GET MODELS
    alg_type = config['algs_types'][config['alg']]

    if alg_type in ['prediction', 'anomaly']:
        if not config['train_models']:  # LOAD
            subjects_models = load_models(dir_models=config['dirs']['output_models'])
        else:  # TRAIN
            if alg_type == 'prediction':
                subjects_models = train_save_pred_models(subjects_traintest=subjects_traintest,
                                                         config=config,
                                                         alg=config['alg'])
            else:  #alg_type == 'anomaly':
                subjects_models = train_save_htm_models(subjects_traintest=subjects_traintest,
                                                        config=config)

    # GET DISTANCES FROM ALL TEST SPLITS TO ALL MODELS
    if alg_type == 'prediction':
        print('\nGetting all model preds on all subjs test...')
        subjstest_subjspreds = get_models_preds(subjects_traintest=subjects_traintest,
                                                subjects_models=subjects_models,
                                                test_mode=config['test_mode'],
                                                window_size=config['window_size'],
                                                dir_scalers=config['dirs']['output_scalers'],
                                                scale=config['scaling'],
                                                features=config['features'])
        print('\nGetting all models dists from all subjs test...')
        subjstest_subjsdists = get_models_dists_pred(subjstest_subjspreds=subjstest_subjspreds,
                                                     subjects_traintest=subjects_traintest,
                                                     features=config['features'],
                                                     test_mode=config['test_mode'],
                                                     window_size=config['window_size'])
    elif alg_type == 'anomaly':
        subjstest_subjsdists = get_models_anomscores(subjects_traintest,
                                                     subjects_models,
                                                     config['features'])
    else:  # alg_type == 'distance'
        print('\nGetting all models dists from all subjs test...')
        subjstest_subjsdists = get_models_dists_dist(subjects_traintest,
                                                     config['alg'],
                                                     config['window_size'],
                                                     config['features'],
                                                     config['test_mode'], )
    # GET RANK SCORES
    print('\nGetting and saving all subjs rank scores...')
    label = f"alg={config['alg']}--hz={config['hz']}"
    if config['alg'] != 'htm':
        label += f"--window={config['window_size']}--mode={config['test_mode']}"
    write_rankscores(subjstest_subjsdists=subjstest_subjsdists,
                     dir_output=config['dirs']['output_results'])
    print('  DONE')


if __name__ == '__main__':
    config = load_config(get_args().config_path)
    run_pipeline(config)

""" TEST """
# config_path = "/Users/samheiserman/Desktop/PhD/Motion-Print/configs/run_pipeline.yaml"
# config = load_config(config_path)
# run_pipeline(config)
