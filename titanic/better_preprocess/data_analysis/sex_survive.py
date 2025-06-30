import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_comp.titanic.standard.preprocess import preprocess  # noqa

train_data, test_data = preprocess()

# 性別ごとの生存率を計算
sex_survived = train_data.groupby("Sex")["Survived"].mean()
sex_labels = ["male", "female"]

# Sexが0/1の場合、ラベルを対応させる
if set(sex_survived.index) == {0, 1}:
    x = [0, 1]
    labels = ["male", "female"]
else:
    x = sex_survived.index.tolist()
    labels = x  # type: ignore

plt.bar(x, sex_survived.values, tick_label=labels)
plt.xlabel("Sex")
plt.ylabel("Survived rate")
plt.title("Survival Rate by Sex")
plt.show()
