import os
import sys

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir, load_models


def train_htm(subjects_traintest, config, path_htm_config):
    dir_htm_src = config['dirs']['htm_src']
    dir_data = config['dirs']['output_data']
    dir_htm_pipeline = os.path.join(dir_htm_src, 'source', 'pipeline')
    if dir_htm_pipeline not in sys.path:
        sys.path.append(dir_htm_pipeline)

    paths_to_drop = [p for p in sys.path if 'Motion-Print' in p]
    for p in paths_to_drop:
        sys.path.remove(p)
    print(f'\nSys paths...')
    for p in sys.path:
        print(f"  --> {p}")

    # TRY CHANING CURRENT DIR
    curr_dir = os.getcwd()
    print(f"\ncurr_dir = {curr_dir}")
    os.chdir(dir_htm_pipeline)
    print(f"curr_dir = {os.getcwd()}")

    from htm_stream_runner import run_stream
    for subj, traintest in subjects_traintest.items():
        data_stream_dir = os.path.join(config['dirs']['output_data'], subj)
        outputs_dir = os.path.join(config['dirs']['output_results'], subj)
        models_dir = os.path.join(config['dirs']['output_models'], subj)
        make_dir(data_stream_dir)
        make_dir(outputs_dir)
        make_dir(models_dir)
        data_path = os.path.join(dir_data, f"{subj}.csv")
        run_stream(config_path=path_htm_config,
                   data_path=data_path,
                   data_stream_dir=data_stream_dir,
                   outputs_dir=outputs_dir,
                   models_dir=models_dir)
    subjects_models = load_models(models_dir)
    return subjects_models