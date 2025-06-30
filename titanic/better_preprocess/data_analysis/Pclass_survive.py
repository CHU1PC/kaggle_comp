import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from kaggle_comp.titanic.standard.preprocess import preprocess  # noqa

train_data, test_data = preprocess()


pclass = train_data.groupby("Pclass")["Survived"].mean()
print(pclass.head())
plt.bar(pclass.index, pclass.values)
plt.xlabel("Pclass")
plt.ylabel("Survived rate")
plt.title("Survival Rate by Pclass")
plt.xticks(pclass.index)
plt.show()
