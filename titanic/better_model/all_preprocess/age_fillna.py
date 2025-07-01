import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import MLP


def pred_ages(train_data, test_data, device, batch_size=16, n_epoch=20,
              lr=0.001):
    # train+testを結合
    all_data = pd.concat([train_data, test_data],
                         sort=False).reset_index(drop=True)
    age_feat = all_data[["Age", "Pclass", "Sex", "Honorifics"]].copy()
    age_dummies = pd.get_dummies(age_feat).astype(float)

    n_train = len(train_data)
    # train部分でMLP学習
    known_age = age_dummies.iloc[:n_train][
            age_dummies.iloc[:n_train]["Age"].notna()
        ].to_numpy()
    x = torch.tensor(known_age[:, 1:], dtype=torch.float32)
    y = torch.tensor(known_age[:, 0], dtype=torch.float32)
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP(x.shape[1], 1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for _ in range(n_epoch):
        model.train()
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb).squeeze()
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

    # train/test両方のAge欠損を補完
    for idx, data in enumerate([train_data, test_data]):
        if idx == 0:
            null_idx = age_dummies.iloc[:n_train][
                    age_dummies.iloc[:n_train]["Age"].isnull()
                ].index
            null_x = age_dummies.iloc[null_idx, 1:].to_numpy()
            X_null = torch.tensor(null_x, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                pred_age = model(X_null).cpu().numpy().flatten()
                data.loc[data["Age"].isna(), "Age"] = pred_age
        else:
            null_idx = age_dummies.iloc[n_train:][
                    age_dummies.iloc[n_train:]["Age"].isnull()
                ].index
            null_x = age_dummies.iloc[null_idx, 1:].to_numpy()
            X_null = torch.tensor(null_x, dtype=torch.float32).to(device)
            model.eval()
            with torch.no_grad():
                pred_age = model(X_null).cpu().numpy().flatten()
                data.loc[data["Age"].isna(), "Age"] = pred_age

    return train_data, test_data
