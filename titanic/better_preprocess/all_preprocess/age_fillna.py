import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from model import MLP


def pred_ages(train_data, test_data, device, batch_size=16, n_epoch=20,
              lr=0.001):

    age = train_data[["Age", "Pclass", "Sex", "Honorifics"]].copy()
    age_dummies = pd.get_dummies(age).astype(float)
    known_age = age_dummies[age_dummies["Age"].notna()].to_numpy()
    null_age = age_dummies[age_dummies["Age"].isna()].to_numpy()

    # age意外とageだけに分ける
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

    null_x = null_age[:, 1:]
    X_null = torch.tensor(null_x, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_age = model(X_null).cpu().numpy().flatten()
        null_idx = train_data[train_data["Age"].isna()].index
        train_data.loc[null_idx, "Age"] = pred_age

    return train_data, test_data
