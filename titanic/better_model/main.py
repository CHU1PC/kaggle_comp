import os
import torch
from config import DATA_DIR
from train_pred import train_lgbm, predict_lgbm, run_optuna


def main(train_data=True, use_optuna=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(DATA_DIR, "lgbm.pkl")
    if train_data:
        if use_optuna:
            best_params = run_optuna(device=device, n_trials=30)
            train_lgbm(device=device, lgbm_params=best_params)
        else:
            train_lgbm()
    else:
        if not os.path.exists(model_path):
            raise FileExistsError("学習済みモデルが存在しません")
    predict_lgbm()


if __name__ == "__main__":
    # Optunaを使いたい場合は引数をTrueに
    main(train_data=True, use_optuna=True)
