#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_3d(p0,p1,rpoints):
    n_features = rpoints.shape[1]
    if n_features == 2:
        fig,ax = plt.subplots()
        ax.scatter(rpoints[:,0],rpoints[:,1],color='r')
        #ax.scatter(points[:,0],points[:,1],color='b')
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
        ax.scatter(rpoints[:,0],rpoints[:,1],rpoints[:,2],color='r')
        #ax.scatter(points[:,0],points[:,1],points[:,2],color='b')

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
