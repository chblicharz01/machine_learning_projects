import pandas as pd
import numpy as np
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

mglearn.plots.plot_knn_classification(n_neighbors=1)
X, y = mglearn.datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_train, y_train)

print("Test set predictions: ", clf.predict(X_test))
print("Test score accuracy: {:.2f}".format(clf.score(X_test, y_test)))