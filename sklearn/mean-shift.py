#!/usr/bin/python

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import datasets
from sklearn.preprocessing import scale

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

## Creating Mean-Shift object to process the dataset
## **********************************************
# The bandwidth parameter dictates the size of the region to search through
# This parameter can be set manually, but can be estimated using the provided estimate_bandwidth function, which is called if the bandwidth is not set.
bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=50) # <-- n_samples needs to be calculated based on the dataset size!
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)

## Compute mean-shift clustering against the original data set
ms.fit(data)
print(ms.labels_)

## Save cluster centers
data_cluster_centers = ms.cluster_centers_
print(data_cluster_centers)

## Preprocess data set and repeat mean-shift clustering
scaleddata = scale(data) ## <-- this scale function makes all points to be part of the same cluster!!
ms.fit(scaleddata)
print(ms.labels_)
scaleddata_cluster_centers = ms.cluster_centers_
print(scaleddata_cluster_centers)
