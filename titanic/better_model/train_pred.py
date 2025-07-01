import os
import pandas as pd
from sklearn.model_selection import train_test_split
from all_preprocess import preprocess
from config import DATA_DIR, FEATURES
from model import LGBMClassifierWrapper
import joblib  # LightBGMはtorchで作られたmodelはないためそれを保存するためのimport
import optuna
from sklearn.model_selection import StratifiedKFold


def objective(trial, train_data):
    train_data = train_data
    x = train_data.drop("Survived", axis=1)
    y = train_data["Survived"]

    # Optunaでパラメータをサジェスト
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
        "random_state": 42,
    }

    N_SPLITS = 5  # 5分割交差検証
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

    val_accuracies = []

    for train_idx, val_idx in skf.split(x, y):
        x_train, y_train = x.iloc[train_idx], y.iloc[train_idx]
        x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

        model = LGBMClassifierWrapper(params=params)
        model.fit(x_train.to_numpy(), y_train.to_numpy(),
                  X_val=x_val.to_numpy(), y_val=y_val.to_numpy(),
                  num_boost_round=2000,       # 学習ラウンド数を増やす
                  early_stopping_rounds=20)  # 少し増やす

        val_pred = model.predict(x_val.to_numpy())
        val_acc = (val_pred == y_val.to_numpy()).mean()
        val_accuracies.append(val_acc)

    # 全ての分割での平均精度を返す
    return sum(val_accuracies) / len(val_accuracies)


def run_optuna(device, n_trials=30):
    train_data, _ = preprocess(device=device)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, train_data),
                   n_trials=n_trials)
    print("-" * 50, "\n\n\n")
    print("Best params:", study.best_params)
    print("Best val acc:", study.best_value)
    return study.best_params


def train_lgbm(device=None, return_train_acc=False, lgbm_params=None):
    train_data, _ = preprocess(device)
    x = train_data.drop("Survived", axis=1).to_numpy()
    y = train_data["Survived"].to_numpy()
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    if lgbm_params is None:
        # デフォルトパラメータを設定
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
        }
    else:
        # Optunaのパラメータに、固定のパラメータを追加/上書き
        params = lgbm_params.copy()  # 元の辞書をコピー
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"

    print("最終モデルを学習します...")
    print("使用するパラメータ:", params)

    model = LGBMClassifierWrapper(params=params)

    model.fit(x_train, y_train, X_val=x_val, y_val=y_val,
              num_boost_round=1000, early_stopping_rounds=10)
    joblib.dump(model, os.path.join(DATA_DIR, "lgbm.pkl"))
    if return_train_acc:
        # 注意：この精度は訓練データに対するものであり、汎化性能ではない
        train_pred = model.predict(x)
        train_acc = (train_pred == y).mean()
        return train_acc


def predict_lgbm(device=None):
    import joblib
    _, test_data = preprocess(device)
    x_test = test_data[FEATURES].values
    model = joblib.load(os.path.join(DATA_DIR, "lgbm.pkl"))
    preds = model.predict(x_test)
    passenger_ids = test_data["PassengerId"].values
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print("submission.csv を出力しました")
    return submission
