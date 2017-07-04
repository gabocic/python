#!/usr/bin/python

from sklearn.cluster import Birch
from sklearn import datasets
from sklearn.preprocessing import scale
import numpy as np

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

## Creating BIRCH object to process the dataset
## **********************************************
# To avoid the computation of global clustering, for every call of partial_fit the user is advised
# * To set n_clusters=None initially
# * Train all data by multiple calls to partial_fit.
# * Set n_clusters to a required value using brc.set_params(n_clusters=n_clusters).
# * Call partial_fit finally with no arguments, i.e brc.partial_fit() which performs the global clustering
birch = Birch(branching_factor=50, n_clusters=None, threshold=0.5,compute_labels=True)

## Compute mean-shift clustering against the original data set
birch.fit(data)
print(len(np.unique(birch.labels_)))
print(birch.labels_)

## Preprocess data set and repeat mean-shift clustering
scaleddata = scale(data)
birch.fit(scaleddata)
print(birch.labels_)
