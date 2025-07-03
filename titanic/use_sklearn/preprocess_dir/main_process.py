import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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
    data["Embarked"] = data["Embarked"].fillna("C")

    # 敬称を作る
    data["Hono"] = data["Name"].apply(
            lambda x: x.split(",")[1].split(".")[0].strip())
    # 対応するkeyとvalueを決める
    Hono_dict = {}
    Hono_dict.update(dict.fromkeys(
        ["Capt", "Col", "Major", "Dr", "Rev"], "Officer"))
    Hono_dict.update(dict.fromkeys(
        ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    Hono_dict.update(dict.fromkeys(
        ['Mme', 'Ms', 'Mrs'], 'Mrs'))
    Hono_dict.update(dict.fromkeys(
        ['Mlle', 'Miss'], 'Miss'))
    Hono_dict.update(dict.fromkeys(
        ['Mr'], 'Mr'))
    Hono_dict.update(dict.fromkeys(
        ['Master', 'Jonkheer'], 'Master'))
    data["Hono"] = data["Hono"].map(Hono_dict)

    # # na以外のAgeの分布
    # age_dist = data["Age"].dropna()
    # nan_idx = data["Age"].isna()
    # # Ageの分布からランダムに取り出す
    # data.loc[nan_idx, "Age"] = np.random.choice(age_dist, size=nan_idx.sum(),
    #                                             replace=True)  # Trueにして復元処理に
    # ageのfillnaをモデルで予測したものを入れるようにする
    age = data[["Age", "Pclass", "Sex", "Hono"]]
    age_dummies = pd.get_dummies(age)

    known_age = age_dummies[age_dummies["Age"].notna()].to_numpy()
    na_age = age_dummies[age_dummies["Age"].isna()].to_numpy()

    age_x = known_age[:, 1:]
    age_y = known_age[:, 0]

    rf = RandomForestRegressor()
    rf.fit(age_x, age_y)
    pred_Age = rf.predict(na_age[:, 1:])
    data.loc[(data["Age"].isna()), "Age"] = pred_Age

    # Ticket numberを重複数で分ける
    # value_counts()でとってきて対応関係をTieket全部に適応させる
    data["TicketGroup"] = data["Ticket"].map(data["Ticket"].value_counts())
    data["TicketGroup"] = np.select([data["TicketGroup"].isin([2, 3, 4, 8]),
                                     data["TicketGroup"].isin([1, 5, 6, 7]),
                                     data["TicketGroup"] >= 9],
                                    [0, 1, 2], default=-1)

    fare_median = data[(data["Embarked"] == "S") &
                       (data["Pclass"] == 3)]["Fare"].median()
    data["Fare"] = data["Fare"].fillna(fare_median)

    # 同乗している家族の数
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1
    data["FamilyLabel"] = np.select([data["FamilySize"].isin([2, 3, 4]),
                                     data["FamilySize"].isin([5, 6, 8, 1]),
                                     data["FamilySize"] >= 8],
                                    [0, 1, 2], default=-1)

    # 苗字による特徴
    data["Surname"] = data["Name"].apply(lambda x: x.split(",")[0].strip())
    data["Surname_count"] = data["Surname"].map(data["Surname"].value_counts())

    # 苗字に重複がある人を、女性または子ども、大人かつ男性に分ける
    Female_Child_Group = \
        data.loc[(data["Surname_count"] >= 2) &
                 ((data['Age'] <= 12) | (data['Sex'] == 'female'))]
    Male_Adult_Group = \
        data.loc[(data['Surname_count'] >= 2) &
                 (data['Age'] > 12) & (data['Sex'] == 'male')]

    # 女・子供グループにおける苗字ごとの生存率平均の個数を比較
    Female_Child_mean = \
        Female_Child_Group.groupby('Surname')['Survived'].mean()
    Female_Child_mean_count = pd.DataFrame(Female_Child_mean.value_counts())
    Female_Child_mean_count.columns = ['GroupCount']

    # 男（大人）グループにおける苗字ごとの生存率平均の個数を比較
    Male_Adult_mean = Male_Adult_Group.groupby('Surname')['Survived'].mean()
    Male_Adult_mean_count = pd.DataFrame(Male_Adult_mean.value_counts())
    Male_Adult_mean_count.columns = ['GroupCount']

    Dead_List = set(Female_Child_mean[
            Female_Child_mean.apply(lambda x: x == 0)].index)
    Survived_List = set(Male_Adult_mean[
            Male_Adult_mean.apply(lambda x: x == 1)].index)

    train = data.loc[data["Survived"].notna()]
    test = data.loc[data["Survived"].isna()]

    # 女・子供グループで全員死亡した苗字の人→６０歳の男性、敬称はMrに。
    # 男（大人）グループで全員生存した苗字の人→５才の女性、敬称はMissに。
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)),
             'Sex'] = 'male'
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)),
             'Age'] = 60
    test.loc[(test['Surname'].apply(lambda x: x in Dead_List)),
             'Title'] = 'Mr'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)),
             'Sex'] = 'female'
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)),
             'Age'] = 5
    test.loc[(test['Surname'].apply(lambda x: x in Survived_List)),
             'Title'] = 'Miss'

    # 再びデータを結合
    data = pd.concat([train, test])

    # 最終的に使う要素
    data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked',
                 'Hono', 'FamilyLabel', 'Deck', 'TicketGroup']]

    data = pd.get_dummies(data)
    # print(data.shape)

    if get_id:
        return data, PassengerId
    else:
        return data


data, passenger = preprocess(get_id=True)
# print(data["Survived"].value_counts())
