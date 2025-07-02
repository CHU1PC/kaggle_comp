import torch
import json
import pandas as pd

from model import MLP
from config import FEATURES
from train_pred import train, age_train


def objective(trial, train_data, device):
    n_hidden1 = trial.suggest_int("n_hidden1", 8, 128)
    n_hidden2 = trial.suggest_int("n_hidden2", 8, 128)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.6)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_epoch = trial.suggest_int("n_epoch", 100, 500)

    model = MLP(n_in=len(FEATURES), n_out=1,
                n_hidden1=n_hidden1, n_hidden2=n_hidden2,
                dropout1=dropout1, dropout2=dropout2).to(device)

    acc = train(model=model, train_data=train_data, batch_size=batch_size,
                n_epoch=n_epoch, lr=lr, device=device,
                return_val_acc=True)
    return acc


def age_objective(trial, train_data, device):
    n_hidden1 = trial.suggest_int("n_hidden1", 8, 64)
    n_hidden2 = trial.suggest_int("n_hidden2", 8, 64)
    dropout1 = trial.suggest_float("dropout1", 0.1, 0.5)
    dropout2 = trial.suggest_float("dropout2", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    n_epoch = trial.suggest_int("n_epoch", 10, 200)

    # 特徴量数をダミー変数化後に合わせる
    age = train_data[["Age", "Pclass", "Sex", "Honorifics", "Family"]].copy()
    age_dummies = pd.get_dummies(age).astype(float)
    n_in = age_dummies.shape[1] - 1  # Age列を除いた数

    model = MLP(n_in=n_in, n_out=1,
                n_hidden1=n_hidden1, n_hidden2=n_hidden2,
                dropout1=dropout1, dropout2=dropout2).to(device)

    acc = age_train(model=model, train_data=train_data, batch_size=batch_size,
                    n_epoch=n_epoch, lr=lr, device=device,
                    return_val_loss=True)
    return acc


def save_model_and_params(model, params, model_path, param_path):
    torch.save(model.state_dict(), model_path)
    with open(param_path, "w") as f:
        json.dump(params, f)


def load_model_and_params(model_path, param_path, device):
    with open(param_path) as f:
        params = json.load(f)

    model = MLP(
        n_in=len(FEATURES), n_out=1,
        n_hidden1=params["n_hidden1"],
        n_hidden2=params["n_hidden2"],
        dropout1=params["dropout1"],
        dropout2=params["dropout2"]
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, params
