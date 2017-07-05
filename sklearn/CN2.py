#!/usr/bin/python

import Orange
from sklearn import datasets
# Read some data
titanic = Orange.data.Table("titanic")

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

orangetable = Orange.data.Table(data)

# construct the learning algorithm and use it to induce a classifier
cn2_learner = Orange.classification.rules.CN2Learner()
cn2_classifier = cn2_learner(titanic)

# All rule-base classifiers can have their rules printed out like this:
for r in cn2_classifier.rules:
        print Orange.classification.rules.rule_to_string(r)

for ejemplo in orangetable:
    print(ejemplo)
