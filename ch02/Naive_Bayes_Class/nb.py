import numpy as np

#BermoulliNB example

#X is the array of test data 
X = np.array([[0,1,0,1],
	[1,0,1,1],
	[0,0,0,1],
	[1,0,1,0]])
#y is the array that determines the class of each test data point
y=np.array([0,1,0,1])

#counts is the output list of each unique class
counts = {}
#for each unique label in y...
for label in np.unique(y):
	#where the label is equal to the class, add to that labels quantity count
	counts[label] = X[y==label].sum(axis=0)
#print out the totals
print("Feature counts:\n", counts)

