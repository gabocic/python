#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
import numpy as np
from numpy.linalg import lstsq
from sklearn import datasets
from numpy.linalg import norm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plot_2d_3d import plot_2d_3d

class TooFewPoints(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def analyze_dataset():
    dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

    ## Converting from SciPy Sparse matrix to numpy ndarray
    data = dataspmat.toarray()

    print(data)

    # Average for each set of coordinates
    datamean = data.mean(axis=0)

    # Singular Value Descomposition
    ## Any given matrix can be factorize as the product of three matrixes: A = U E V*
    ## The matrix V[n_features x n_features] is unitary and its first row is a vector which corresponds to the direction of the line that fit the data points
    U,E,V = np.linalg.svd(data - datamean)

    ## Parametric line: r-> = ro + kv->
    # multiplying the direction vector by several different factors to generate points
    linepts = V[0] * np.mgrid[-50:50:4j][:, np.newaxis]
    # adding the datamean point to the points generated previously to obtain the line that better fit all points
    linepts += datamean

    #Calculate the norm of the vector formed by the edges of the "box" containing the data points
    boxnorm = norm(np.amax(data,axis=0) - np.amin(data,axis=0))
    print("boxnmorm")
    print(boxnorm)

## Plotting
#########################################

    n_features = data.shape[1]
    if n_features == 2:
        fig,ax = plt.subplots()
        ax.scatter(*data.T,color='r')
        ax.scatter(*datamean,color='g',s=10)
        ax.plot(*linepts.T,'-', color='b')
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
        # For each point, find the distance to the line 
        for row in data:
            d = norm(np.cross(row-datamean, V[0]))/norm(V[0])
            print(d)

            if d < 0.025*boxnorm:
                ax.scatter(*row,color='y')
            else:
                ax.scatter(*row,color='r')

        #ax.scatter3D(data[:,0],data[:,1],data[:,2],color='r')
        ax.scatter3D(*datamean,color='g',s=10)
        ax.plot3D(*linepts.T,'-', color='b')
    else:
        sys.exit()


    # SubTitle
    fig.suptitle("Dataset linear points", fontsize=10)

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

    


analyze_dataset()
