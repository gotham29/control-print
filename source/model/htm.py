import numpy as np
import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir, load_models, save_data_as_pickle
from htm_source.pipeline.htm_stream_runner import run_stream
from htm_source.pipeline.htm_batch_runner import run_batch
from htm_source.config.config import load_config, save_config


def reset_htm_config(htm_config_orig):
    keys_keep = ['features',
                 'models_state',
                 'timesteps_stop']
    keys_keep_state = ['model_for_each_feature',
                       'save_outputs_accumulated',
                       'track_iter',
                       'track_tm']
    htm_config = {k: v for k, v in htm_config_orig.items() if k in keys_keep}
    htm_config['models_state'] = {k: v for k, v in htm_config['models_state'].items() if k in keys_keep_state}
    return htm_config_orig


def get_htm_dist(mod_dict, test, learn=False, predictor_config={'enable': False}):
    aScores = []
    htm_multimodels = False if len(mod_dict) == 1 else True
    for timestep, row in test.iterrows():
        features_data = dict(row)
        if htm_multimodels:
            modname = list(mod_dict.keys())[0]
            mod = mod_dict[modname]
            aScore, aLikl, prCount, stepsPreds = mod.run(features_data, timestep, learn, predictor_config)
        else:
            for modname, mod in mod_dict.items():
                aScore, aLikl, prCount, stepsPreds = mod.run(features_data, timestep, learn, predictor_config)
                aScores.append(aScore)
        aScores.append(aScore)
    return np.mean(aScores)


def train_save_htm_models(subjects_traintest, config):
    print(f"\nTraining {len(subjects_traintest)} HTM models...")
    htm_config_orig = reset_htm_config(config['htm_config'])
    subjects_models = {}
    for subj, traintest in subjects_traintest.items():
        models_dir = os.path.join(config['dirs']['output_models'], subj)
        make_dir(models_dir)
        htm_config_orig = reset_htm_config(htm_config_orig)
        subjects_models[subj], subj_outputs = run_batch(cfg=htm_config_orig,
                                                        config_path=None,
                                                        learn=True,
                                                        data=traintest['train'],
                                                        iter_print=1000,
                                                        features_models={})
        multiple_models = True if len(subjects_models[subj]) > 1 else False
        if multiple_models:
            for feat in config['features']:
                outpath = os.path.join(models_dir, f"{feat}.pkl")
                save_data_as_pickle(subjects_models[subj][feat], outpath)
        else:
            multi_feat = f"megamodel_features={len(config['features'])}"
            outpath = os.path.join(models_dir, f"{multi_feat}.pkl")
            save_data_as_pickle(subjects_models[subj][multi_feat], outpath)

    return subjects_models
