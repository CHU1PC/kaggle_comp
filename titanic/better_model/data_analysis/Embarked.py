import os
import pandas as pd

train_data = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "train.csv"))

embarked_close = \
    train_data[(train_data["Pclass"] == 1) & (train_data["Survived"] == 1) &
               (train_data["SibSp"] == 0) & (train_data["Parch"] == 0) &
               (train_data["Cabin"].str.get(0) == "B")
               ].drop(["Name", "Ticket", "Cabin"], axis=1)

embarked_close["Sex"] = embarked_close["Sex"].\
    replace({"male": 0, "female": 1}).astype(float)

embarked_close = embarked_close.groupby("Embarked").mean()

print(embarked_close)
