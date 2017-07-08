#!/home/gabriel/pythonenvs/v3.5/bin/python

import Orange
from sklearn import datasets

# Read some data

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

## Creating an Orange "Domain" to work with the data and the tags
# Define each attribute
att1 = Orange.data.ContinuousVariable(name='att1', compute_value=None)
att2 = Orange.data.ContinuousVariable(name='att2', compute_value=None)
att3 = Orange.data.ContinuousVariable(name='att3', compute_value=None)
classv = Orange.data.DiscreteVariable(name='classv', compute_value=None)

# Create domain based on the above attributes
mydomain = Orange.data.Domain(attributes=[att1,att2,att3],class_vars=classv)

## Loading data and tags in ndarray format into a an Orange.Table
table = Orange.data.Table.from_numpy(mydomain,data,Y=tags)

# construct the learning algorithm and use it to induce a classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_classifier = cn2_learner(table)

# All rule-base classifiers can have their rules printed out like this:
#for r in cn2_classifier.rules:
#    print(Orange.classification.rules.rule_to_string(r))
