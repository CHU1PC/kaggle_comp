import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_comp.titanic.standard.preprocess import preprocess  # noqa

train_data, test_data = preprocess()

# Survived == 1 (生存) と Survived == 0 (死亡) のデータに分ける
survived_fare = train_data[train_data['Survived'] == 1]['Fare']
not_survived_fare = train_data[train_data['Survived'] == 0]['Fare']

# ヒストグラムを重ねて描画
# alphaで透明度を指定すると重なりが見やすい
plt.hist(survived_fare, bins=30, alpha=0.5,
         label='Survived', density=True)
plt.hist(not_survived_fare, bins=30, alpha=0.3,
         label='Not Survived', density=True)

# グラフの装飾
plt.xlabel("Fare")
plt.ylabel("Density")  # density=Trueで正規化（割合表示）
plt.title("Fare Distribution by Survival Status")
plt.legend()  # 凡例を表示
plt.grid(True)
plt.show()
