import os
from config import DATA_DIR
from train_pred import train_lgbm, predict_lgbm


def main(train_data=True):
    model_path = os.path.join(DATA_DIR, "lgbm.pkl")
    if train_data:
        train_lgbm()
    else:
        if not os.path.exists(model_path):
            raise FileExistsError("学習済みモデルが存在しません")
    predict_lgbm()


if __name__ == "__main__":
    main()
