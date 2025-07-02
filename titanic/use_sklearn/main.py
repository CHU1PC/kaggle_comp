import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV

from config import DATA_DIR
from preprocess_dir import preprocess


def main():
    data, PassengerId = preprocess(get_id=True)

    train_data = data[data["Survived"].notna()]
    test_data = data[data["Survived"].isna()].drop("Survived", axis=1)

    x = train_data.to_numpy()[:, 1:]  # Survived以外
    y = train_data.to_numpy()[:, 0].astype(int)  # Survived
    print(np.unique(y, return_counts=True))

    clf = RandomForestClassifier(random_state=10, max_features="sqrt")
    pipe = Pipeline([("classify", clf)])
    param_test = {"classify__n_estimators": list(range(20, 30, 1)),
                  "classify__max_depth": list(range(3, 10, 1))}

    grid = GridSearchCV(estimator=pipe, param_grid=param_test,
                        scoring='accuracy', cv=10)
    grid.fit(x, y)
    print(grid.best_params_, grid.best_score_, sep="\n")

    pred = grid.predict(test_data)

    submission = pd.DataFrame({"PassengerId": PassengerId["test"],
                               "Survived": pred.astype(np.int32)})

    submission.to_csv(os.path.join(DATA_DIR, "submission3.csv"))
    print("submission.csv を出力しました")


if __name__ == "__main__":
    main()
