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

    print(test_data["Name"].isna().sum())
    Honorifics_Dict = {}
    Honorifics_Dict.update(dict.fromkeys(
        ['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer'))
    Honorifics_Dict.update(dict.fromkeys(
        ['Don', 'Sir', 'the Countess', 'Dona', 'Lady'], 'Royalty'))
    Honorifics_Dict.update(dict.fromkeys(['Mme', 'Ms', 'Mrs'], 'Mrs'))
    Honorifics_Dict.update(dict.fromkeys(['Mlle', 'Miss'], 'Miss'))
    Honorifics_Dict.update(dict.fromkeys(['Mr'], 'Mr'))
    Honorifics_Dict.update(dict.fromkeys(['Master', 'Jonkheer'], 'Master'))
    train_data['Honorifics'] = train_data['Honorifics'].map(Honorifics_Dict)
    test_data["Honorifics"] = test_data["Honorifics"].map(Honorifics_Dict)
    train_data = train_data.drop("Name", axis=1)
    test_data = test_data.drop("Name", axis=1)

    return train_data, test_data


train, test = preprocess()

# print(train.dtypes)
