#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.preprocessing import scale
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


## Extracting amount of groups by determining unique tags number
n_digits = len(np.unique(tags))

## Creating KMeans object to process the dataset
## **********************************************
# k-means++ initialization scheme is specified (use the init='kmeans++' parameter), which has been implemented in scikit-learn. This initializes the centroids to be (generally) 
# distant from each other, leading to probably better results than random initialization (init=random and PCA-based are two other possible choices). The parameter n_job specify 
# the amount of processors to be used (default: 1). A value of -1 uses all available processors, with -2 using one less, and so on.
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, n_jobs=2)

## Compute k-means clustering against the original data set
kmeans.fit(data)

## Save centroids for plotting
datacentroids=kmeans.cluster_centers_

## Preprocess data set and repeat k-means clustering
scaleddata = scale(data)
kmeans.fit(scaleddata)
scaleddatacentroids=kmeans.cluster_centers_



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

# Plot the centroids as a green X for both, the original and the scaled data set
centroids = datacentroids
#ax.scatter(centroids[:, 0], centroids[:, 1], # <-- 2D
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
            marker='x', s=169, linewidths=3,
            color='g', zorder=10)

centroids = scaleddatacentroids
#ax.scatter(centroids[:, 0], centroids[:, 1], # <-- 2D
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
            marker='x', s=169, linewidths=3,
            color='g', zorder=10)

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

