import lightgbm as lgb
from lightgbm import early_stopping
import torch.nn as nn
import numpy as np


# AgeをfillnaするときにMLPをつかう
class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden1=32, n_hidden2=32,
                 dropout1=0.4, dropout2=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(n_hidden2, n_out)
        )

    def forward(self, x):
        return self.net(x)


# 実際に学習&予想をするときにLightGBMをつかう
class LGBMClassifierWrapper:
    def __init__(self, params=None):
        if params is None:

            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "verbosity": -1,
                "boosting_type": "gbdt",
                "random_state": 42,
            }
        self.params = params
        self.model = None

    def fit(self, X_train, y_train, X_val=None, y_val=None,
            num_boost_round=100, early_stopping_rounds=10):
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        if X_val is not None and y_val is not None:
            X_val = np.asarray(X_val)
            y_val = np.asarray(y_val)
        # パラメータ型安全性
        num_boost_round = int(num_boost_round)
        early_stopping_rounds = int(early_stopping_rounds)

        lgb_train = lgb.Dataset(X_train, y_train)
        valid_sets = [lgb_train]
        valid_names = ["train"]

        fit_params = {
            "params": self.params,
            "train_set": lgb_train,
            "num_boost_round": num_boost_round,
            "valid_sets": valid_sets,
            "valid_names": valid_names
        }

        if X_val is not None and y_val is not None:
            lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)
            valid_sets.append(lgb_val)
            valid_names.append("valid")
            fit_params["valid_sets"] = valid_sets
            fit_params["valid_names"] = valid_names
            self.model = lgb.train(
                **fit_params,
                callbacks=[early_stopping(early_stopping_rounds)]
            )
        else:
            self.model = lgb.train(**fit_params)

    def predict(self, X):

        if self.model is None:
            raise ValueError("Model is not trained."
                             "Please call fit() before predict().")

        return (
            self.model.predict(
                X, num_iteration=self.model.best_iteration) > 0.5
        ).astype(int)
