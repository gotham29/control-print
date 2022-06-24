import os
import sys
import time

from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import array

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir, save_data_as_pickle


def train_lstm(subjects_traintest, features, data_cap, dir_models, n_steps=1, n_epochs=100):
    """ TODO --> Validate LSTM implementation """

    print(f"\nTraining {len(subjects_traintest)} LSTM models...")

    make_dir(dir_models)
    subjects_models = {}
    n_features = len(features)
    counter = 1
    time_start = time.time()

    for subj, traintest in subjects_traintest.items():
        X_array = array(traintest['test'][features])
        y_array = array(traintest['test'][features].shift(-1))
        # drop last row since NaN for y
        y_array = y_array[:len(y_array) - 1]
        X_array = X_array[:len(X_array) - 1]
        X = X_array.reshape((X_array.shape[0], 1, X_array.shape[1]))
        # y = y_array.reshape((y_array.shape[0], 1, y_array.shape[1]))

        """ SCALE DATA (?) """

        # build & compile model
        model = Sequential()
        model.add(LSTM(100, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
        model.add(LSTM(100, activation='relu'))
        model.add(Dense(n_features))
        # layers = [LSTM(2), Dense(n_features)]
        # model = Sequential(layers)
        model.compile(optimizer='adam', loss='mse')

        # fit model
        model.fit(X[:data_cap], y_array[:data_cap], epochs=n_epochs, verbose=0)

        # save model
        subjects_models[subj] = model
        path_mod = os.path.join(dir_models, f"{subj}.pkl")
        save_data_as_pickle(model, path_mod)

        # track time
        time_elapsed_mins = round((time.time() - time_start) / 60, 2)
        print(f"  Trained {counter} of {len(subjects_traintest)} models; elapsed minutes = {time_elapsed_mins}")
        counter += 1
    print('  DONE')

    return subjects_models
