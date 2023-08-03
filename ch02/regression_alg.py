import pandas as pd
import numpy as np
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print("Data shape: ", data.shape)

X, y = mglearn.datasets.load_extended_boston()
print("X.shape: ", X.shape)

mglearn.plots.plot_knn_regression(n_neighbors=1)

