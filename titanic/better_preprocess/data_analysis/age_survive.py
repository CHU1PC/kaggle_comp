import os
import sys
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess import preprocess  # noqa

train_data, test_data = preprocess()

survived_age = train_data[train_data["Survived"] == 1]["Age"]
not_survived_age = train_data[train_data["Survived"] == 0]["Age"]

plt.hist(survived_age, bins=30, alpha=0.5,
         label="Survived", density=True)
plt.hist(not_survived_age, bins=30, alpha=0.3,
         label="Not Survived", density=True)

plt.xlabel("Age")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
