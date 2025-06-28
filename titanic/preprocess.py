import os
import pandas as pd

from config import DATA_DIR, FEATURES

pd.set_option('future.no_silent_downcasting', True)


# Survived, Pclass, Sex, Age
# Ageだけfillna
def preprocess():
    train_data = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test_data = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

    train_data = train_data[["Survived"] + FEATURES]
    test_data = test_data[["PassengerId"] + FEATURES]

    # 欠損値補完
    for col in ["Age", "Fare"]:
        train_data[col] = train_data[col].fillna(train_data[col].mean())
        test_data[col] = test_data[col].fillna(test_data[col].mean())

    # Sexの数値化
    train_data["Sex"] = train_data["Sex"].replace({"male": 0, "female": 1})
    test_data["Sex"] = test_data["Sex"].replace({"male": 0, "female": 1})

    # スケーリング（例：標準化）
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    train_data[FEATURES] = scaler.fit_transform(train_data[FEATURES])
    test_data[FEATURES] = scaler.transform(test_data[FEATURES])

    return train_data, test_data
