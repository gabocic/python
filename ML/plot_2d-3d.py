#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

generator = np.random

n_features=3
p0 = np.array(generator.random_integers(low=(0), high=10, size=(n_features)))
p1 = np.array(generator.random_integers(low=(0), high=10, size=(n_features)))

d0 = np.array(p1 - p0)


## PLOTTING RESULTS
## ******************************

if n_features == 2:
    fig,ax = plt.subplots()
else:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') # <-- 3D

# SubTitle
fig.suptitle("Dataset linear points", fontsize=10)

## Plot the original data points
for a in range(1,9):
    lins = p0+a*d0
    print(lins)
    ax.scatter(*lins,color='b') # <-- 3D
ax.scatter(*p0, color='g') # <-- 3D
ax.scatter(*p1, color='g') # <-- 3D

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
