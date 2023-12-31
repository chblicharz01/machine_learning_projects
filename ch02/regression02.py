import mglearn
import numpy as np 
import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

X,y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)

print("test set predictions:\n", reg.predict(X_test))
print("Test set R^2: {:.2f}".format(reg.score(X_test, y_test)))