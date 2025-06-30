import os
import pandas as pd

from config import DATA_DIR
from preprocess import preprocess

train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# print(train_data.info())
train_data, _ = preprocess()

print(train_data.corrwith(train_data["Survived"]))
