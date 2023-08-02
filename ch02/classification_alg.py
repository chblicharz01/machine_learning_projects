from sklearn.datasets import load_breast_cancer
import numpy as np
import mglearn
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
print("Shape of cancer data: ", cancer.data.shape)

print('Sample counts per class:\n', 
	{n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))})

print("\nFeature names:\n", cancer.feature_names)

