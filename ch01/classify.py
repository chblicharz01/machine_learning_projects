#importations
import mglearn
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#load the dataset
iris_dataset = load_iris()


#dataset testing
'''
#print("Keys of iris_dataset:\n", iris_dataset.keys())
print(iris_dataset['DESCR'][:193] + "\n...")
print('Feature names:\n', iris_dataset['feature_names'])
print('Type of data:', type(iris_dataset['data']))
print("Shape of data: ", iris_dataset['data'].shape)

print("First five rows of data:\n", iris_dataset['data'][:150])

print("Type of target: ", type(iris_dataset['target']))
print("Shape of target: ", iris_dataset['target'].shape)

print('Target:\n', iris_dataset['target'])
'''

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state = 0)

print("X_train shape: ", X_train.shape)
print("y_train shape: ", y_train.shape)

print("X_test shape: ",X_test.shape)
print("y_test shape: ", y_test.shape)



#graphing - this still doesn't work for now
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

'''
display(pd.plotting.scatter_matrix(iris_dataframe, c=y_train,
                           figsize=(15,15), marker='o',hist_kwds={'bins':20}, s = 60,
                           alpha=.8,cmap=mglearn.cm3))
'''


#K-neighbor testing
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

#finding a new iris in the wild
X_new = np.array([[5, 2.9, 1, .2 ]])
print("X_new.shape:", X_new.shape)



prediction = knn.predict(X_new)
print("Prediction:", prediction)


print("Predicted target name: ", iris_dataset['target_names'][prediction])
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

#evaluating this model
y_pred = knn.predict(X_test)
print("Test set predictions:\n", y_pred)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score 2: {:.2f}".format(knn.score(X_test, y_test)))
