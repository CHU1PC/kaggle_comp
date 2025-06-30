import os
import pandas as pd

from config import DATA_DIR

train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))

# print(train_data.info())
train_data = train_data[["Survived", "Pclass", "Sex",
                         "Age", "SibSp", "Parch", "Fare"]]
train_data["Sex"] = train_data["Sex"].replace({"male": 0, "female": 1})
train_data["Age"] = train_data["Age"].fillna(train_data["Age"].mean())


print(train_data.corrwith(train_data["Survived"]))
