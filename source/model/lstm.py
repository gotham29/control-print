import os
import sys
import time

from keras.layers import LSTM, Dense
from keras.models import Sequential
from numpy import array

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir, save_data_as_pickle


def forecast_lstm(model, batch_size, X):
    # print("\nforecast_lstm()")
    # print(f"  X shape = {X.shape}")
    # X = X.reshape(1, 1, len(X))
    # print(f"  X re-shaped = {X.shape}")
    X_re = X.reshape(1, X.shape[0], X.shape[1])
    yhat = model.predict(X_re, batch_size=batch_size, verbose=0)
    return yhat


def train_lstm(X, y, data_cap, config_lstm, model):
    if config_lstm['stateful']:
        for _ in range(config_lstm['n_epochs']):
            model.fit(X[:data_cap], y[:data_cap], epochs=1, batch_size=config_lstm['batch_size'], verbose=0, shuffle=config_lstm['shuffle'])
            if config_lstm['reset_states']:
                model.reset_states()
    else:
        model.fit(X[:data_cap], y[:data_cap], epochs=config_lstm['n_epochs'], batch_size=config_lstm['batch_size'], verbose=0, shuffle=config_lstm['shuffle'])
    # model.fit(X[:data_cap], y[:data_cap], epochs=config_lstm['n_epochs'], verbose=0)
    return model


def compile_lstm(X, config_lstm, n_steps, n_outputs):
    """
    * Random initial conditions for LSTM net cause different results each time a given config is trained

    Batch size of 1
        --> is required as we will be using walk-forward validation and making one-step forecasts
        --> means that the model will be fit using online training (vs batch or mini-batch)

    LSTM flavors to try:
        --> A Stateful LSTM
        --> A Stateless LSTM with the same configuration.
        --> A Stateless LSTM with shuffling during training.
        Expectations:
            --> The stateful LSTM will outperform the stateless LSTM
            --> The stateless LSTM without shuffling will outperform the stateless LSTM with shuffling
            --> Stateless and stateful LSTMs should produce near identical results when using the same batch size
            --> Resetting state after each training epoch results in better test performance.
            --> Seeding state in the LSTM by making predictions on the training dataset results in better test performance.
            --> Not resetting state between one-step predictions on the test set results in better test set performance.
    """
    model = Sequential()
    model.add(LSTM(config_lstm['n_units'], batch_input_shape=(config_lstm['batch_size'], X.shape[1], X.shape[2]), stateful=config_lstm['stateful']))
    model.add(Dense(n_outputs))
    model.compile(loss=config_lstm['loss'], optimizer=config_lstm['optimizer'])
    # model = Sequential()
    # for layer_i in range(config_lstm['n_layers']):
    #     if layer_i == 0:
    #         model.add(LSTM(units=config_lstm['n_units'],
    #                        batch_input_shape=(config_lstm['batch_size'], n_steps, n_features),
    #                        return_sequences=True,
    #                        stateful=config_lstm['stateful']))
    #     else:
    #         model.add(LSTM(units=config_lstm['n_units'],
    #                        batch_input_shape=(config_lstm['batch_size'], n_steps, n_features),
    #                        stateful=config_lstm['stateful']))
    # model.add(Dense(1)) #n_features
    # model.compile(optimizer=config_lstm['optimizer'], loss=config_lstm['loss'])
    return model


# def convert_data_for_lstm(X_array, n_steps):
#     return X_array.reshape((X_array.shape[0], n_steps, X_array.shape[1]))  # 1-->window_size

