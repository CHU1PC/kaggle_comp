import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import DATA_DIR, FEATURES  # noqa
from .age_fillna import pred_ages  # noqa


pd.set_option('future.no_silent_downcasting', True)


def preprocess(device, features=FEATURES):
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    """敬称一覧
    Mr：男 , Master：男の子, Jonkheer：オランダ貴族(男),
    Mlle：マドモワゼル (フランス未婚女性), Miss：未婚女性、女の子, Mme：マダム(フランス既婚女性), Ms：女性(未婚・既婚問わず),
    Mrs：既婚女性,
    Don：男(スペイン), Sir：男(イギリス), the Countess：伯爵夫人, Dona：既婚女性(スペイン),
    Lady：既婚女性(イギリス),
    Capt：船長, Col：大佐, Major：軍人, Dr：医者, Rev：聖職者や牧師
    """
    # ,で区切って[1]をとることでMr, Mrs, Miss以降などを取り出し.で区切りそれだけをとる
    train_data["Honorifics"] = train_data["Name"].\
        apply(lambda x: x.split(",")[1].split(".")[0].strip())
    test_data["Honorifics"] = test_data["Name"].\
        apply(lambda x: x.split(",")[1].split(".")[0].strip())

    Honorifics_Dict = {}
    Honorifics_Dict.update(dict.fromkeys(
        ['Capt', 'Col', 'Major', 'Dr', 'Rev'], 0))
    Honorifics_Dict.update(dict.fromkeys(
        ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 1))
    Honorifics_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 2))
    Honorifics_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 3))
    Honorifics_Dict.update(dict.fromkeys(['Mr'], 4))
    Honorifics_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 5))

    train_data['Honorifics'] = train_data['Honorifics'].\
        map(Honorifics_Dict).astype(float)
    test_data["Honorifics"] = test_data["Honorifics"].\
        map(Honorifics_Dict).astype(float)

    # 欠損値補完
    train_data["Embarked"] = train_data["Embarked"].fillna("S")
    test_data["Embarked"] = test_data["Embarked"].fillna("S")
    train_data["Cabin"] = train_data["Cabin"].fillna("T")
    test_data["Cabin"] = test_data["Cabin"].fillna("T")
    train_data, test_data = pred_ages(train_data, test_data, device)

    # 家族の合計人数
    train_data["Family_size"] = train_data[["SibSp", "Parch"]].sum(axis=1) + 1
    test_data["Family_size"] = test_data[["SibSp", "Parch"]].sum(axis=1) + 1

    # 要素の選定
    train_data = train_data[["Survived"] + features]
    test_data = test_data[["PassengerId"] + features]

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
    for data in [train_data, test_data]:
        data["Cabin"] = \
            data["Cabin"].replace({r"^A.*": 0, r"^B.*": 1, r"^C.*": 2,
                                   r"^D.*": 3, r"^E.*": 4, r"^F.*": 5,
                                   r"^G.*": 6, r"^T.*": -1}, regex=True)

    return train_data, test_data
