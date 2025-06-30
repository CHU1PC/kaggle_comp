import os
import torch

from config import DATA_DIR, FEATURES
from train_pred import train, predict
from model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(train_data=True):
    model_path = os.path.join(DATA_DIR, "mlp.pth")
    if train_data:
        if os.path.exists(model_path):
            os.remove(model_path)
        train(batch_size=32, n_epoch=1000, lr=0.001, device=device)

    model = MLP(n_in=len(FEATURES), n_out=1).to(device)
    predict(model, device)


if __name__ == "__main__":
    main()
