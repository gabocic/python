#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import DBSCAN
from time import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
from plot_2d_3d import plot_2d_3d
from scipy.misc import derivative

def dbscan_clustering(data,plot,p_n_jobs):

    #def f(x):
    #    print('Z',z)
    #    return z[0]*x**3 + z[1]*x**2 + z[2]*x + z[3]


    ## Creating DBSCAN object to process the dataset
    ## **********************************************
    # There are two parameters to the algorithm, min_samples and epsilon, which define formally what we mean when we say "dense". Higher min_samples or lower eps indicate 
    # higher density necessary to form a cluster.

    l_NearestNeighborsAlg = ['auto', 'ball_tree', 'kd_tree', 'brute']
    NearestNeighborsAlg = 'auto'

    # Defining min_samples as 10% of dataset size
    min_samples = data.shape[0] * 0.1

    # Estimating eps
    nbrs = NearestNeighbors(n_neighbors=int(min_samples-1), algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)
    sorteddist = np.sort(distances,axis=None)

    k = sorteddist.shape[0] -1
    minvar1 = 0
    minvar2 = 0
    while k >= 0:
        print(k,'-',k-9)
        variance = (sorteddist[k] - sorteddist[k-9])/10
        print(variance)
        if k == sorteddist.shape[0]-1:
            minvar1 = variance
            minvarran1 = (k,k-9)
            minvar2 = variance
            minvarran2 = (k,k-9)
        else:
            if variance < minvar1:
                minvar2 = minvar1
                minvarran2 = minvarran1
                minvar1 = variance
                minvarran1 = (k,k-9)
        k = k-10
    print('min variance',minvar1,minvarran1)
    print('min variance',minvar2,minvarran2)
    
    #z = np.polyfit(np.arange(sorteddist.shape[0]),sorteddist, 3)
    #print(z)
    #for valor in range(0,sorteddist.shape[0]):
    #    deriv = derivative(f, valor, dx=1e-6)
    #    print(deriv)
    #    break
    
    #print(sorteddist)
    f = open('workfile', 'w')
    for valor in sorteddist:
        #print(valor)
        f.write(valor.__str__()+'\n')
    f.close()

    dbscan = DBSCAN(eps=80000, min_samples=10,metric='euclidean',algorithm=NearestNeighborsAlg,n_jobs=p_n_jobs)

    # Initial time mark
    t0 = time()

    ## Compute dbscan clustering against the original data set
    dbscan.fit(data)

    # Calculate process time
    elap_time = (time() - t0)

    return dbscan,elap_time

