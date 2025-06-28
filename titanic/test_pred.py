import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from preprocess import preprocess
from config import DATA_DIR, FEATURES
from model import Logistic


def train(device, batch_size=16, n_epoch=20, lr=0.001):
    train_data, _ = preprocess()

    x_train = torch.tensor(train_data[FEATURES].values, dtype=torch.float32)

    y_train = torch.tensor(train_data["Survived"].values,
                           dtype=torch.float32).reshape(-1, 1)

    x_dataset = TensorDataset(
        x_train,
        y_train
    )

    train_dataloader = DataLoader(
        x_dataset, batch_size, shuffle=True
    )

    model = Logistic(n_in=x_train.shape[1], n_out=1).to(device)

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
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "logistic.pth"))


def predict(model, device):
    _, test_data = preprocess()

    x_test = torch.tensor(test_data[FEATURES].values,
                          dtype=torch.float32).to(device)

    model.load_state_dict(torch.load(os.path.join(DATA_DIR, "logistic.pth"),
                                     map_location=device))
    model.eval()

    with torch.no_grad():
        outputs = model(x_test)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5).int().cpu().numpy().flatten()

    if "PassengerId" in test_data.columns:
        passenger_ids = test_data["PassengerId"].values
    else:
        # 必要に応じてIDを生成
        passenger_ids = range(892, 892 + len(test_data))

    # DataFrame作成
    submission = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": preds
    })
    submission.to_csv(os.path.join(DATA_DIR, "submission.csv"), index=False)
    print("submission.csv を出力しました")
    return submission
