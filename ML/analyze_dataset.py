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

    print(data)
    print("")
    print("")
    print("")
    print("")

    A=np.concatenate((data[:,:data.shape[1]-1],np.ones((data.shape[0],1))),axis=1)

    print("MATRIX A")
    print("**********")
    print(A)
    print("")
    print("")
    B = data.T
    
    print("MATRIX B")
    print("**********")
    print(B)
    print("")
    print("")

    print("B[B.shape[0]-1:B.shape[0],:]")
    print("**********")
    print(B[B.shape[0]-1:B.shape[0],:])
    print("")
    print("")

    k=np.linalg.lstsq(A, B[B.shape[0]-1:B.shape[0],:][0])[0]

    print(k)

    datamean = data.mean(axis=0)
    uu, dd, vv = np.linalg.svd(data - datamean)

    n_features = data.shape[1]
    if n_features == 2:
        fig,ax = plt.subplots()
        ax.scatter(*data.T,color='r')
        ax.scatter(*datamean,color='g',s=10)
        linepts = vv[0] * np.mgrid[-50:50:4j][:, np.newaxis]

        ## r-> = ro + kv->
        linepts += datamean
        ax.plot(*linepts.T,'-', color='b')
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
        ax.scatter3D(data[:,0],data[:,1],data[:,2],color='r')
        ax.scatter3D(*datamean,color='g',s=10)
        #ax.scatter3D(0,0,0,color='b')
        #linep = k.T * np.mgrid[-7:7:4j][:, np.newaxis]
        #Zz = A * k 
        #linep = np.concatenate((data[:,:data.shape[1]-1],Zz),axis=1)
        #ax.plot3D(*linep.T,'-', color='g')
        #ax.plot3D(k[0],k[1],k[2],'-', color='g')
        #l = np.zeros((1,3))
        #r = np.array(vv[0])
        #p = np.vstack((r,l))
        linepts = vv[0] * np.mgrid[-50:50:4j][:, np.newaxis]

        ## r-> = ro + kv->
        linepts += datamean
        ax.plot3D(*linepts.T,'-', color='b')

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
