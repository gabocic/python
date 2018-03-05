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

    # Detecting elbow
    dx = int(0.01 * sorteddist.shape[0])
    minvar=0
    minidxul=0
    print('dx',dx)
    k = sorteddist.shape[0] -1
    l_ranupplim = []
    while k >= 0:
        variance = ((sorteddist[k] - sorteddist[k-(dx-1)])/sorteddist[-1])/(dx/sorteddist.shape[0])
        print(k,'-',k-(dx-1))
        print('variance',variance)
        print('')
        #if k == sorteddist.shape[0]-1:
        #    minvar = variance
        #    minidxul = k
        #else:
        #    if variance !=0 and variance < minvar:
        #        minvar = variance
        #        minidxul=k

        if 0.5 <= variance <= 0.6:
            print(k,'-',k-(dx-1))
            print(variance)
            print('')
            l_ranupplim.append(k-(dx-1))
            #l_ranupplim.append(k)
        k = k-dx

    # Get the max upper limit for the winning ranges and use the middle value as eps
    print('minidxul',minidxul)
    #winidx = minidxul - int((dx-1)/2)
    winidx = max(l_ranupplim) - (dx-1)
    #winidx = max(l_ranupplim)
    print(winidx)
    v_eps = sorteddist[winidx]
    print('v_eps',v_eps)

    
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
    
    #v_eps = input("eps:")
    #v_eps = float(v_eps)
    dbscan = DBSCAN(eps=v_eps, min_samples=10,metric='euclidean',algorithm=NearestNeighborsAlg,n_jobs=p_n_jobs)

    # Initial time mark
    t0 = time()

    ## Compute dbscan clustering against the original data set
    dbscan.fit(data)

    # Calculate process time
    elap_time = (time() - t0)

    return dbscan,elap_time

