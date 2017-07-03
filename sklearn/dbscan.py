#!/usr/bin/python

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file
from mpl_toolkits.mplot3d import Axes3D

## LOADING DATASET FROM FILE
## *************************
## dataspmat is a scipy matrix storing all the data points. 
## and tags is a numpy.ndarray storing tags for each data point
dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

## Converting from SciPy Sparse matrix to numpy ndarray
data = dataspmat.toarray()

## Creating DBSCAN object to process the dataset
## **********************************************
# There are two parameters to the algorithm, min_samples and epsilon, which define formally what we mean when we say "dense". Higher min_samples or lower eps indicate 
# higher density necessary to form a cluster.
db = DBSCAN(eps=0.3, min_samples=10)

## Compute k-means clustering against the original data set
db.fit(data)

## Preprocess data set and repeat DBSCAN clustering
scaleddata = StandardScaler().fit_transform(data)
db.fit(scaleddata)

print(db.labels_)

## PLOTTING RESULTS
## ******************************

#fig, ax = plt.subplots() # <-- 2D
fig = plt.figure() # <-- 3D
ax = fig.add_subplot(111, projection='3d') # <-- 3D

fig.suptitle("Original and scaled data points with their correspondant K-Means centroids", fontsize=10)

## Plot the original data points
#ax.plot(data[:, 0], data[:, 1],'k.', markersize=2) # <-- 2D
ax.plot(data[:, 0], data[:, 1], data[:, 2],'k.', markersize=2) # <-- 3D

## Plot the scaled data points
#ax.plot(scaleddata[:, 0], scaleddata[:, 1],'k.', markersize=2,color='r') # <-- 2D
ax.plot(scaleddata[:, 0], scaleddata[:, 1], scaleddata[:, 2],'k.', markersize=2,color='r') # <-- 3D

## Graph and axis formatting
ax.set_aspect('equal')
ax.grid(True, which='both')

# set the x-spine
ax.spines['left'].set_position('zero')

# turn off the right spine/ticks
ax.spines['right'].set_color('none')
#ax.yaxis.tick_left()

# set the y-spine
ax.spines['bottom'].set_position('zero')

# turn off the top spine/ticks
ax.spines['top'].set_color('none')
ax.xaxis.tick_bottom()

plt.show()

