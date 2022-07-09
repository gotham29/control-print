import os
import sys
import time

from numpy import array
from pmdarima.arima import auto_arima

_SOURCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.append(_SOURCE_DIR)

from source.utils.utils import make_dir, save_data_as_pickle


def train_arima(X_train, config_arima, data_cap):
    features_models = {}
    for feat in X_train:
        model = auto_arima(y=X_train[feat][:data_cap].values,
                           start_p=config_arima['start_p'],
                           start_P=config_arima['start_P'],
                           start_q=config_arima['start_q'],
                           start_Q=config_arima['start_Q'],
                           max_p=config_arima['max_p'],
                           max_d=config_arima['max_d'],
                           max_q=config_arima['max_q'],
                           max_P=config_arima['max_P'],
                           max_D=config_arima['max_D'],
                           max_Q=config_arima['max_Q'],
                           m=config_arima['m'],
                           max_order=config_arima['max_order'],
                           scoring=config_arima['scoring'])
        features_models[feat] = model
    return features_models