import os
import torch
import optuna
import json
import pandas as pd

from config import FEATURES, DATA_DIR
from all_preprocess import preprocess
from train_pred import train, predict, age_train, age_predict
from model import MLP
from utils import objective, age_objective

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(train_from_one=True):
    age_model_path = os.path.join(DATA_DIR, "age_mlp.pth")
    age_param_path = os.path.join(DATA_DIR, "age_mlp_hyper_param.json")
    model_path = os.path.join(DATA_DIR, "mlp.pth")
    param_path = os.path.join(DATA_DIR, "mlp_hyper_param.json")

    train_data, test_data = preprocess()

    # 年齢予測用の特徴量数を取得
    age = train_data[["Age", "Pclass", "Sex", "Honorifics", "Family"]].copy()
    age_dummies = pd.get_dummies(age).astype(float)
    n_age_in = age_dummies.shape[1] - 1  # Age列を除いた数

    if train_from_one:
        # ここは年齢の予想
        age_study = optuna.create_study(direction="minimize")
        age_study.optimize(lambda trial: age_objective(trial, train_data,
                                                       device),
                           n_trials=20)

        print("=" * 50, "\n\nBest AGE params: ", age_study.best_params)

        age_best = age_study.best_params
        with open(age_param_path, "w") as f:
            json.dump({
                "n_hidden1": age_best["n_hidden1"],
                "n_hidden2": age_best["n_hidden2"],
                "dropout1": age_best["dropout1"],
                "dropout2": age_best["dropout2"],
                "batch_size": age_best["batch_size"],
                "lr": age_best["lr"],
                "n_epoch": age_best["n_epoch"]
            }, f)

        age_model = MLP(
            n_in=n_age_in, n_out=1,
            n_hidden1=age_best["n_hidden1"],
            n_hidden2=age_best["n_hidden2"],
            dropout1=age_best["dropout1"],
            dropout2=age_best["dropout2"]
        ).to(device)

        age_train(age_model, train_data,
                  batch_size=age_best["batch_size"],
                  n_epoch=age_best["n_epoch"],
                  lr=age_best["lr"], device=device)

        train_data = age_predict(age_model, train_data, device)

        # ここからSurvivedかどうかの予想
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, train_data,
                                               device),
                       n_trials=30)

        print("-" * 50, "\n\nBest params: ", study.best_params)

        best = study.best_params
        with open(param_path, "w") as f:
            json.dump({
                "n_hidden1": best["n_hidden1"],
                "n_hidden2": best["n_hidden2"],
                "dropout1": best["dropout1"],
                "dropout2": best["dropout2"]
            }, f)

        model = MLP(
            n_in=len(FEATURES), n_out=1,
            n_hidden1=best["n_hidden1"],
            n_hidden2=best["n_hidden2"],
            dropout1=best["dropout1"],
            dropout2=best["dropout2"]
        ).to(device)

        train(model=model, train_data=train_data,
              batch_size=best['batch_size'], n_epoch=best['n_epoch'],
              lr=best['lr'], device=device)

    else:
        # 年齢予想モデルの有無
        if not (
            os.path.exists(age_model_path) and
            os.path.exists(age_param_path)
        ):
            raise FileExistsError("学習済み年齢予想モデルが存在しません")

        # 生存モデルの有無
        if not (
            os.path.exists(model_path) and
            os.path.exists(param_path)
        ):
            raise FileExistsError("学習済みモデルが存在しません")

        with open(age_param_path) as f:
            age_params = json.load(f)

        with open(param_path) as f:
            params = json.load(f)

        age_model = MLP(
            n_in=n_age_in, n_out=1,
            n_hidden1=age_params["n_hidden1"],
            n_hidden2=age_params["n_hidden2"],
            dropout1=age_params["dropout1"],
            dropout2=age_params["dropout2"]
        ).to(device)
        age_model.load_state_dict(torch.load(age_model_path,
                                             map_location=device))

        train_data = age_predict(age_model, train_data, device)

        model = MLP(
                    n_in=len(FEATURES), n_out=1,
                    n_hidden1=params["n_hidden1"],
                    n_hidden2=params["n_hidden2"],
                    dropout1=params["dropout1"],
                    dropout2=params["dropout2"]
                ).to(device)

    predict(model, test_data, device)


if __name__ == "__main__":
    main()
