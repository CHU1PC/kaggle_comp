import os
import sys

import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from preprocess import preprocess  # noqa

train, _ = preprocess()

hono_survive = train.groupby("Honorifics")["Survived"].mean()

plt.bar(hono_survive.index, hono_survive.to_numpy())
plt.xlabel("honorifics")
plt.ylabel("Survived rate")
plt.title("Survival Rate by honorifics")
plt.xticks(hono_survive.index)
plt.show()
