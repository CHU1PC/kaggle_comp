import os
import pandas as pd

from config import DATA_DIR, FEATURES

pd.set_option('future.no_silent_downcasting', True)


# Survived, Pclass, Sex, Age
# Ageだけfillna
def preprocess(features=FEATURES):
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_data = train_data[["Survived"] + features]
    test_data = test_data[["PassengerId"] + features]

    # 欠損値補完
    for col in ["Age", "Fare"]:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(test_data[col].mean())
    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    test_data["Embarked"] = test_data["Embarked"].fillna("S")
    train_data["Cabin"] = train_data["Cabin"].fillna("T")
    test_data["Cabin"] = test_data["Cabin"].fillna("T")

    # Sexの数値化
    train_data["Sex"] = train_data["Sex"].\
        replace({"male": 0, "female": 1}).astype(int)
    test_data["Sex"] = test_data["Sex"].\
        replace({"male": 0, "female": 1}).astype(int)

    # Embarkedの数値化
    train_data["Embarked"] = \
        train_data["Embarked"].replace({"S": 0, "C": 1, "Q": 2}).astype(int)
    test_data["Embarked"] = \
        test_data["Embarked"].replace({"S": 0, "C": 1, "Q": 2}).astype(int)

    # Cabinの数値化
    train_data["Cabin"] = \
        train_data["Cabin"].replace({r"^A.*": 0, r"^B.*": 1, r"^C.*": 2,
                                     r"^D.*": 3, r"^E.*": 4, r"^F.*": 5,
                                     r"^G.*": 6, r"^T.*": -1}, regex=True)
    test_data["Cabin"] = \
        test_data["Cabin"].replace({r"^A.*": 0, r"^B.*": 1, r"^C.*": 2,
                                    r"^D.*": 3, r"^E.*": 4, r"^F.*": 5,
                                    r"^G.*": 6, r"^T.*": -1}, regex=True)

    return train_data, test_data


# train, test = preprocess()

# print(train.dtypes)
