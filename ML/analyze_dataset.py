#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
import numpy as np
from numpy.linalg import lstsq
from sklearn import datasets
from numpy.linalg import norm
from sympy.solvers import solve
from sympy import Symbol

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
    n_features = data.shape[1]

    # Average for each set of coordinates
    datamean = data.mean(axis=0)

    # Singular Value Descomposition
    ## Any given matrix can be factorize as the product of three matrixes: A = U E V*
    ## The matrix V[n_features x n_features] is unitary and its first row is a vector which corresponds to the direction of the line that fit the data points
    U,E,V = np.linalg.svd(data - datamean)

    #Calculate the norm of the vector formed by the edges of the "box" containing the data points
    #boxnorm = norm(np.amax(data,axis=0) - np.amin(data,axis=0))
   
    # What I'm trying to do here is to calculate the norm of the fitting line that intersects the "box" where the points are contained.
    # The challenge is to find the two parameters that would become the two ends of the line
   

    maxcords=np.amax(data,axis=0)
    mincords=np.amin(data,axis=0)

    # High point
    # Calculating parameters as MaxCor_x / V_x,  MaxCor_y / V_y, etc
    
    # For every V and maxcords components
    higherthan=[]
    lowerthan=[]
    for idx in range(0,n_features):
        print("maxcords["+idx.__str__()+"]: "+maxcords[idx].__str__())
        print("V[0]["+idx.__str__()+"]: "+V[0][idx].__str__())
        if V[0][idx] >= 0 and maxcords[idx] >= 0:
            lowerthan.append((maxcords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] >= 0 and maxcords[idx] < 0:
            lowerthan.append((maxcords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] < 0 and maxcords[idx] >= 0:
            higherthan.append((maxcords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] < 0 and maxcords[idx] < 0:
            higherthan.append((maxcords[idx]-datamean[idx])/V[0][idx])
        print("higherthan:",higherthan)    
        print("lowerthan:",lowerthan)    

    if len(lowerthan) > 0 and len(higherthan) > 0:
        if abs(min(lowerthan)) < abs(max(higherthan)):
            h_par=min(lowerthan)
        else:
            h_par=max(higherthan)
        #h_par=min(lowerthan)
        #if h_par < max(higherthan):
        #    print("High point - Impossible parameter")
        #    sys.exit()
        print("Max higherthan:",max(higherthan))
        print("Min lowerthan:",min(lowerthan))
    elif len(lowerthan) > 0:
        print("Min lowerthan:",min(lowerthan))
        h_par=min(lowerthan)
    elif len(higherthan) > 0:
        h_par=max(higherthan)
        print("Max higherthan:",max(higherthan))
    print("h_par: ",h_par)


    # Low point
    # Calculating parameters as MaxCor_x / V_x,  MaxCor_y / V_y, etc

    higherthan=[]
    lowerthan=[]
    for idx in range(0,n_features):
        print("mincords["+idx.__str__()+"]: "+mincords[idx].__str__())
        print("V[0]["+idx.__str__()+"]: "+V[0][idx].__str__())
        if V[0][idx] >= 0 and mincords[idx] >= 0:
            higherthan.append((mincords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] >= 0 and mincords[idx] < 0:
            higherthan.append((mincords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] < 0 and mincords[idx] >= 0:
            lowerthan.append((mincords[idx]-datamean[idx])/V[0][idx])
        if V[0][idx] < 0 and mincords[idx] < 0:
            lowerthan.append((mincords[idx]-datamean[idx])/V[0][idx])
        print("higherthan:",higherthan)    
        print("lowerthan:",lowerthan)    

    if len(lowerthan) > 0 and len(higherthan) > 0:
        if abs(min(lowerthan)) < abs(max(higherthan)):
            l_par=min(lowerthan)
        else:
            l_par=max(higherthan)
        #l_par=max(higherthan)
        #l_par=min(lowerthan)
        #if l_par > min(lowerthan):
        #    print("Low point - Impossible parameter")
        #    sys.exit()
        print("Min lowerthan:",min(lowerthan))
        print("Max higherthan:",max(higherthan))
    elif len(lowerthan) > 0:
        l_par=min(lowerthan)
        print("Min lowerthan:",min(lowerthan))
    elif len(higherthan) > 0:
        l_par=max(higherthan)
        print("Max higherthan:",max(higherthan))
    print("l_par: ",l_par)
    print("datamean",datamean)

    print("norm: ",norm(h_par-l_par))
    dthres = (norm(h_par-l_par)) * 0.09

    ## Parametric line: r-> = ro + kv->
    # multiplying the direction vector by several different constants to generate points
    #linepts = V[0] * np.mgrid[-50000:50000:4j][:, np.newaxis]
    linepts = V[0] * np.mgrid[l_par:h_par:2j][:, np.newaxis]
    # adding the datamean point to the points generated previously to obtain the line that better fit all points
    linepts += datamean
    print(linepts)


    #Calculate the distance of each point to the line
    pdist=[]
    for row in data:
        A = datamean
        B = datamean + V[0]
        P = row
        pa = P - A
        ba = B - A
        t = np.dot(pa,ba)/np.dot(ba,ba)
        d = norm(pa - t*ba)
        pdist.append(d)
    pdist = np.asarray(pdist)
    #print(pdist)


#    sys.exit()

    # Separate "linear" points from "non linear"
    linp=[]
    nlinp=[]
    u = 0
    for d in pdist:
        if d < dthres:
            linp.append(data[u,:])
        else:
            nlinp.append(data[u,:])
        u+=1

    l_linp = len(linp)
    l_nlinp = len(nlinp)

    print("Percentage of linear points:",100*(l_linp/(l_linp+l_nlinp)))
    print("Percentage of Non linear points:",100*(l_nlinp/(l_linp+l_nlinp)))

## Plotting
#########################################

    if n_features == 2:
        fig,ax = plt.subplots()
    elif n_features == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d') # <-- 3D
    else:
        sys.exit()
    
    # Plot each point using a different color if distance from the line is within the threshold
    for point in linp:
        ax.scatter(*point,color='y')
    for point in nlinp:
        ax.scatter(*point,color='r')
    
    ax.scatter(*datamean,color='g',s=10)
    ax.plot(*linepts.T,'-', color='b')
    

    # SubTitle
    #ax.suptitle("Dataset linear points", fontsize=10)

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
