import os
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.ensemble import RandomForestRegressor
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

    select = SelectKBest(k=20)
    clf = RandomForestClassifier(random_state=10, warm_start=True,
                                 n_estimators=26,
                                 max_depth=6,
                                 max_features="sqrt")
    pipeline = make_pipeline(select, clf)
    pipeline.fit(x, y)

    cv_score = model_selection.cross_val_score(pipeline, x, y, cv=10)
    print(f"CV score : Mean - {np.mean(cv_score):.7g} | "
          f"Std - {np.std(cv_score):.7g}")

    predictions = pipeline.predict(test_data)
    submission = pd.DataFrame({"PassengerId": PassengerId["test"],
                               "Survived": predictions.astype(np.int32)})
    submission.to_csv(os.path.join(DATA_DIR, "submission4.csv"), index=False)
    print("submission.csv を出力しました")


if __name__ == "__main__":
    main()
