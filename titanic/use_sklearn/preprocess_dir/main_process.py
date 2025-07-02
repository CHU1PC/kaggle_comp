import os
import sys
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR  # noqa


def preprocess(get_id=True):

    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    data = pd.concat([train_data, test_data], axis=0)

    # trainとtestがindexの行とする
    PassengerId = pd.concat([train_data["PassengerId"],
                             test_data["PassengerId"]],
                            axis=0, keys=["train", "test"]
                            ).reset_index(level=1, drop=True)

    data["Deck"] = data["Cabin"].fillna("Unknown")
    data["Deck"] = data["Deck"].str.get(0)  # 1文字目だけを抽出

    # Embarkedをなにで埋めるかを調べるためにvalue_counts()を行い多かったCを採用
    # print("=" * 50)
    # print(data[(data["Deck"].str.startswith("B")) &
    #            (data["Sex"] == "female")]["Embarked"].value_counts())
    # print("=" * 50)
    data["Embarked"] = data["Embarked"].fillna("S")

    # na以外のAgeの分布
    age_dist = data["Age"].dropna()
    nan_idx = data["Age"].isna()
    # Ageの分布からランダムに取り出す
    data.loc[nan_idx, "Age"] = np.random.choice(age_dist, size=nan_idx.sum(),
                                                replace=True)  # Trueにして復元処理に

    data["Fare"] = data["Fare"].fillna(data["Fare"].mean())

    # Ticket numberを重複数で分ける
    # value_counts()でとってきて対応関係をTieket全部に適応させる
    data["TicketGroup"] = data["Ticket"].map(data["Ticket"].value_counts())
    data["TicketGroup"] = np.select([data["TicketGroup"].isin([2, 3, 4, 8]),
                                     data["TicketGroup"].isin([1, 5, 6, 7]),
                                     data["TicketGroup"] >= 9],
                                    [0, 1, 2], default=-1)

    data = data[["Survived", "Pclass", "Sex", "Age", "Fare",
                 "Embarked", "Deck", "TicketGroup"]]

    data = pd.get_dummies(data)
    # print(data.shape)

    if get_id:
        return data, PassengerId
    else:
        return data


# data, passenger = preprocess(get_id=True)
# print(data["Survived"].value_counts())
