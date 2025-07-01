import os
import torch
import optuna

from config import FEATURES, DATA_DIR
from train_pred import train, predict
from model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    n_epoch = trial.suggest_int("n_epoch", 100, 500)
    acc = train(batch_size=batch_size, n_epoch=n_epoch, lr=lr, device=device,
                return_val_acc=True)
    return acc


def main(train_data=False):
    model_path = os.path.join(DATA_DIR, "mlp.pth")
    if train_data:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=30)
        print("-" * 50, "\n\nBest params: ", study.best_params)

        best = study.best_params
        train(batch_size=best['batch_size'], n_epoch=best['n_epoch'],
              lr=best['lr'], device=device)

    else:
        if not os.path.exists(model_path):
            raise FileExistsError("学習済みモデルが存在しません")

    model = MLP(n_in=len(FEATURES), n_out=1).to(device)
    predict(model, device)


if __name__ == "__main__":
    main()
