import torch.nn as nn


class Logistic(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)

    def forward(self, x):
        y = self.fc(x)
        return y
