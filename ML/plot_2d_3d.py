#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#generator = np.random

#n_features=3
#p0 = generator.randn(n_features)
#p1 = generator.randn(n_features)
#d0 = np.array(p1 - p0)

#points = np.zeros((15,n_features))
#for a in range(1,15):
#    lins = p0+a*d0
#    # Add some noise 
#    lins += np.random.normal(size=lins.shape) * 0.4
#    points[a:a+1,:] = lins
#print(points)

def plot_2d_3d(points,p0,p1,rpoints):
    n_features = points.shape[1]
    if n_features == 2:
        fig,ax = plt.subplots()
        ax.scatter(points[:,0],points[:,1],color='b')
        ax.scatter(rpoints[:,0],rpoints[:,1],color='r')
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
        ax.scatter(points[:,0],points[:,1],points[:,2],color='b')
        ax.scatter(rpoints[:,0],rpoints[:,1],rpoints[:,2],color='r')

    # SubTitle
    fig.suptitle("Dataset linear points", fontsize=10)

    ax.scatter(*p0, color='g', marker='x', s=90)
    ax.scatter(*p1, color='g', marker='x', s=90)

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