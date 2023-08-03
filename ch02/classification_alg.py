from sklearn.datasets import load_breast_cancer
import numpy as np
import mglearn
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
print("Shape of cancer data: ", cancer.data.shape)

print('Sample counts per class:\n', 
	{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print("\nFeature names:\n", cancer.feature_names)

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=66)

training_accuracy = []
test_accuracy = []

neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	clf.fit(X_train, y_train)
	training_accuracy.append(clf.score(X_train, y_train))
	test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()