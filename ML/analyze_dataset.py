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


def analyze_dataset(debug=0):

    def logger(var=None,message=None,dbg_level=2):
        if dbg_level <= debug:
            print(message,var)

    dataspmat,tags = datasets.load_svmlight_file('dataset.svl')

    ## Converting from SciPy Sparse matrix to numpy ndarray
    data = dataspmat.toarray()

    logger(message="\n Loaded data: \n *******************",var=None,dbg_level=2)
    logger(message=None,var=data,dbg_level=2)
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

    logger(message="maxcords",var=maxcords,dbg_level=2)
    #logger(maxcords,2)
    #logger("mincords",2)
    #logger(mincords,2)
    #logger("Direction vector",2)
    #logger(V[0],2)
    #logger("Mean point",2)
    #logger(datamean,2)

    l_lambdas=[]
    l_hlpoints=[]

    # High point
    # Calculating parameters as MaxCor_x / V_x,  MaxCor_y / V_y, etc
    
    test=(maxcords-datamean)/V[0]
    print("test",test)
    
    # For every V and maxcords components, find which plane the line crosses
    for idx in range(0,n_features):
        # Calculate lambda for this component to be equal the max component in the dataset
        v_lambda = (maxcords[idx]-datamean[idx]) / V[0][idx]
        r_lambda = datamean + v_lambda * V[0]
        #logger("r_lambda",r_lambda)

        # Check if the for the above lambda, the remaining components are within the plane 
        p=0
        ok_count=0
        for r_lambda_comp in r_lambda:
            if p == idx:
                pass
            else:
                if r_lambda_comp >= mincords[p] and r_lambda_comp <= maxcords[p]:
                    ok_count +=1
                    
            p+=1
        if ok_count == n_features -1:
            l_lambdas.append(v_lambda)
            l_hlpoints.append(r_lambda)
            max_lambda = v_lambda
            high_point = r_lambda
            print("max_lambda",max_lambda)
            print("high_point",high_point)
            

    # Low point
    # Calculating parameters as MaxCor_x / V_x,  MaxCor_y / V_y, etc
    print("")
    print("Low point")
    print("*****************")
    print("")

    # For every V and mincords components, find which plane the line crosses
    for idx in range(0,n_features):
        # Calculate lambda for this component to be equal the max component in the dataset
        v_lambda = (mincords[idx]-datamean[idx]) / V[0][idx]
        r_lambda = datamean + v_lambda * V[0]
        print("r_lambda",r_lambda)

        # Check if the for the above lambda, the remaining components are within the plane 
        p=0
        ok_count=0
        for r_lambda_comp in r_lambda:
            if p == idx:
                pass
            else:
                if r_lambda_comp >= mincords[p] and r_lambda_comp <= maxcords[p]:
                    ok_count +=1
                    
            p+=1
        if ok_count == n_features -1:
            l_lambdas.append(v_lambda)
            l_hlpoints.append(r_lambda)
            min_lambda = v_lambda
            low_point = r_lambda
            print("min_lambda",min_lambda)
            print("low_point",low_point)

    #dthres = (norm(high_point-low_point)) * 0.025
    dthres = (norm(l_hlpoints[0]-l_hlpoints[1])) * 0.05
    #dthres = 1

    ## Parametric line: r-> = ro + kv->
    # multiplying the direction vector by several different constants to generate points
    #linepts = V[0] * np.mgrid[-5000:5000:2j][:, np.newaxis]
    #linepts = V[0] * np.mgrid[min_lambda:max_lambda:2j][:, np.newaxis]
    linepts = V[0] * np.mgrid[l_lambdas[0]:l_lambdas[1]:2j][:, np.newaxis]
    linepts += datamean
    #linepts = V[0] * np.mgrid[l_par:h_par:2j][:, np.newaxis]
    # adding the datamean point to the points generated previously to obtain the line that better fit all points
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
    sys.exit()

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

analyze_dataset(debug=2)
