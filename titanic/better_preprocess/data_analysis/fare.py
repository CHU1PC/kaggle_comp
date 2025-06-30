import os
import pandas as pd

train_data = pd.read_csv(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "train.csv"))


print(train_data.info())
# fare has no nan value
