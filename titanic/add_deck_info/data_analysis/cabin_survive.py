import os
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "train.csv"))

cabin_survive = \
    train_data.groupby(
        train_data["Cabin"].str.get(0)
    )["Survived"].mean()

plt.bar(cabin_survive.index, cabin_survive)
plt.xlabel("Cabin initial")
plt.show()
