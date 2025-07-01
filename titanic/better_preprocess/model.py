import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_hidden1=32, n_hidden2=32,
                 dropout1=0.4, dropout2=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(n_hidden1, n_hidden2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(n_hidden2, n_out)
        )

    def forward(self, x):
        return self.net(x)
