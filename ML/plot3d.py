#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

generator = np.random

n_features=3
p0=generator.rand(1,n_features)
p1=generator.rand(1,n_features)
points=[p0[0],p1[0]]

all_coords = np.zeros((n_features, 2))

k=0
for valores in zip(*points):
    all_coords[k:k+1,:] = valores
    k+=1


A=np.vstack((all_coords[:all_coords.shape[0]-1,:],np.ones(2))).T

k=np.linalg.lstsq(A, all_coords[all_coords.shape[0]-1:all_coords.shape[0],:][0])[0]

xy = np.array([[2, 1, 1]])

print(xy.T)
p2 = np.dot(k,xy.T)

print(p2)


## PLOTTING RESULTS
## ******************************

fig = plt.figure() # <-- 3D
ax = fig.add_subplot(111, projection='3d') # <-- 3D

# SubTitle
fig.suptitle("Original and scaled data points with their correspondant K-Means centroids", fontsize=10)

## Plot the original data points
ax.scatter(*all_coords, color='g') # <-- 3D
ax.scatter(2,1,p2,color='g') # <-- 3D
#ax.scatter(p1,'.', markersize=5,color='r') # <-- 3D
#ax.plot(all_coords[:all_coords.shape[0]-1,:],np.dot(A, np.transpose([k])),'-', markersize=5,color='r') # <-- 3D
ax.plot3D(*all_coords,markersize=5,color='r') # <-- 3D

## Plot the scaled data points
#ax.plot(scaleddata[:, 0], scaleddata[:, 1], scaleddata[:, 2],'k.', markersize=2,color='r') # <-- 3D

# Plot the centroids as a green X for both, the original and the scaled data set
#centroids = datacentroids
#ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
#            marker='x', s=169, linewidths=3,
#            color='g', zorder=10)

#centroids = scaleddatacentroids
#ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
#            marker='x', s=169, linewidths=3,
#            color='g', zorder=10)

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

