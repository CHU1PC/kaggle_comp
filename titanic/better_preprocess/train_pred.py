import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from config import DATA_DIR, FEATURES


def train(model, train_data, batch_size=16, n_epoch=20, lr=0.001,
          return_val_acc=False, device=None):
    train_data = train_data.copy()
    x = train_data.drop("Survived", axis=1).to_numpy()
    y = train_data["Survived"].to_numpy()

    x_train, x_val, y_train, y_val = \
        train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = torch.tensor(x_train, dtype=torch.float32)
    # BCEWithLogitsLossが2次元を必要とするためreshapeする
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    # BCEWithLogitsLossが2次元を必要とするためreshapeする
    y_val = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1)

    train_dataset = TensorDataset(
        x_train,
        y_train
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True
    )

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(n_epoch):
        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_output = model(x_val.to(device))
        val_pred = (torch.sigmoid(val_output) >= 0.5).int()
        val_acc = accuracy(val_pred, y_val.int())

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "mlp.pth"))

    if return_val_acc:
        return val_acc


def predict(model, test_data, device):
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


def age_train(model, train_data, batch_size=16, n_epoch=20, lr=0.001,
              return_val_loss=False, device=None):
    train_data = train_data.copy()
    age = train_data[["Age", "Pclass", "Sex", "Honorifics", "Family"]].copy()
    age_dummies = pd.get_dummies(age).astype(float)
    known_age = age_dummies[age_dummies["Age"].notna()].to_numpy()

    x = torch.tensor(known_age[:, 1:], dtype=torch.float32)
    y = torch.tensor(known_age[:, 0], dtype=torch.float32)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2,
                                                      random_state=42)

    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(n_epoch):
        model.train()
        for xb, yb in train_dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(x_val.to(device)).squeeze()
        val_loss = criterion(val_pred, y_val.to(device)).item()

    torch.save(model.state_dict(), os.path.join(DATA_DIR, "age_mlp.pth"))

    if return_val_loss:
        return val_loss


def age_predict(model, train_data, device):

    # age_param_path = os.path.join(DATA_DIR, "age_mlp_hyper_param.json")
    # with open(age_param_path) as f:
    #     params = json.load(f)

    age_model_path = os.path.join(DATA_DIR, "age_mlp.pth")

    age = train_data[["Age", "Pclass", "Sex", "Honorifics", "Family"]].copy()
    age_dummies = pd.get_dummies(age).astype(float)
    null_age = age_dummies[age_dummies["Age"].isna()].to_numpy()

    null_x = null_age[:, 1:]
    X_null = torch.tensor(null_x, dtype=torch.float32).to(device)

    model.load_state_dict(torch.load(age_model_path, map_location=device))
    model.eval()
    with torch.no_grad():
        pred_age = model(X_null).cpu().numpy().flatten()
        null_idx = train_data[train_data["Age"].isna()].index
        train_data.loc[null_idx, "Age"] = pred_age

    return train_data


def accuracy(y_pred, y):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    return (y == y_pred).mean()
