import os
import pandas as pd
from sklearn.model_selection import train_test_split
from all_preprocess import preprocess
from config import DATA_DIR, FEATURES
from model import LGBMClassifierWrapper
import joblib  # LightBGMはtorchで作られたmodelはないためそれを保存するためのimport


def train_lgbm(device=None, return_val_acc=False):
    train_data, _ = preprocess(device)
    x = train_data.drop("Survived", axis=1).to_numpy()
    y = train_data["Survived"].to_numpy()

    # trainとvalを8:2で分ける
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    model = LGBMClassifierWrapper()
    model.fit(x_train, y_train, X_val=x_val, y_val=y_val,
              num_boost_round=100, early_stopping_rounds=10)

    # モデル保存
    joblib.dump(model, os.path.join(DATA_DIR, "lgbm.pkl"))

    # バリデーション精度
    val_pred = model.predict(x_val)
    val_acc = (val_pred == y_val).mean()

    if return_val_acc:
        return val_acc


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
