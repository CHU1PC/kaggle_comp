import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from all_preprocess import preprocess
from config import DATA_DIR, FEATURES
from model import MLP


def train(device, batch_size=16, n_epoch=20, lr=0.001):
    train_data, _ = preprocess(device)

    x_train = torch.tensor(train_data.drop("Survived", axis=1).values,
                           dtype=torch.float32)

    y_train = torch.tensor(train_data["Survived"].values,
                           dtype=torch.float32).reshape(-1, 1)

    x_dataset = TensorDataset(
        x_train,
        y_train
    )

    train_dataloader = DataLoader(
        x_dataset, batch_size, shuffle=True
    )

    model = MLP(n_in=x_train.shape[1], n_out=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    num_epochs = n_epoch
    model.train()
    for epoch in range(num_epochs):
        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 100 == 0:
            preds = (torch.sigmoid(output) >= 0.5).int()
            acc = accuracy(preds, yb.int())
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                  f"Accuracy: {acc:.4f}")

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "mlp.pth"))


def predict(model, device):
    _, test_data = preprocess(device)

    x_test = torch.tensor(test_data[FEATURES].values,
                          dtype=torch.float32).to(device)

    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "mlp.pth"),
                                     map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(x_test)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int().cpu().numpy().flatten()

    passenger_ids = test_data["PassengerId"].values
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print("submission.csv を出力しました")
    return submission


def accuracy(y_pred, y):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return (y == y_pred).mean()
