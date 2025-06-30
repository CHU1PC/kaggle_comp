import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train_data = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "train.csv"))

family_size = train_data[["SibSp", "Parch"]].sum(axis=1) + 1
train_data["family_size"] = family_size
survived_rate = train_data.groupby("family_size")["Survived"].mean()
# print(family_size.shape)
plt.bar(survived_rate.index, survived_rate.values)
plt.xlabel("family size")
plt.ylabel("Survival rate")
plt.xticks(np.arange(1, family_size.max() + 1))
plt.show()
