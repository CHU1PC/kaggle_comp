import os
import torch

from config import DATA_DIR
from test_pred import train, predict
from model import Logistic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(train_data=True):
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
    model_path = os.path.join(DATA_DIR, "logistic.pth")
    if train_data:
        if os.path.exists(model_path):
            os.remove(model_path)
        train(batch_size=16, n_epoch=100, lr=0.001, device=device)

    model = Logistic(n_in=len(features), n_out=1).to(device)
    predict(model, device)


if __name__ == "__main__":
    main()
