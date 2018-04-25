import numpy as np

import Orange

X = np.array([[5,6,7],[0,0,0],[1,9,0],[3,2,1],[2,2,2],[5,1,9]])
Y = np.array([1,1,3,1,2,2])

l_labels = ['a','b','c']
classv = Orange.data.DiscreteVariable(name='classv',values=l_labels)
l_attr = []


l_attr.append(Orange.data.ContinuousVariable(name='f1', compute_value=None))
l_attr.append(Orange.data.ContinuousVariable(name='f2', compute_value=None))
l_attr.append(Orange.data.ContinuousVariable(name='f3', compute_value=None))
mydomain = Orange.data.Domain(attributes=l_attr,class_vars=classv)


table = Orange.data.Table.from_numpy(mydomain,X,Y=Y)

learner = Orange.classification.CN2Learner()

classifier = learner(table)
