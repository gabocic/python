#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from numpy.linalg import lstsq
from sklearn import datasets

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


    A=np.concatenate((data[:,:data.shape[1]-1],np.ones((data.shape[0],1))),axis=1)
    print(A)
    B = data.T
    print(B)
    print(B[B.shape[0]-1:B.shape[0],:])

    k=np.linalg.lstsq(A, B[B.shape[0]-1:B.shape[0],:][0])[0]

    print(k)

    n_features = 3
    if n_features == 2:
        fig,ax = plt.subplots()
        ax.scatter(points[:,0],points[:,1],color='b')
        ax.scatter(rpoints[:,0],rpoints[:,1],color='r')
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
        ax.scatter(data[:,0],data[:,1],data[:,2],color='r')
        ax.plot(B[:B.shape[0]-1,:],np.dot(A, np.transpose([k])),'-', markersize=5,color='g')

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
