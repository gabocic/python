#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import DBSCAN
from time import time
from sklearn.neighbors import NearestNeighbors
import numpy as np
from plot_2d_3d import plot_2d_3d
from scipy.misc import derivative
from bisect import bisect_right
from parameters import *

def dbscan_clustering(data,plot,p_n_jobs):


    ## Creating DBSCAN object to process the dataset
    ## **********************************************
    # There are two parameters to the algorithm, min_samples and epsilon, which define formally what we mean when we say "dense". Higher min_samples or lower eps indicate 
    # higher density necessary to form a cluster.

    l_NearestNeighborsAlg = ['auto', 'ball_tree', 'kd_tree', 'brute']
    NearestNeighborsAlg = 'auto'

    # Defining min_samples as 10% of dataset size
    min_samples = data.shape[0] * dbs_min_samples_per_cluster_perc

    # Estimating eps
    nbrs = NearestNeighbors(n_neighbors=int(min_samples-1), algorithm='auto',n_jobs=p_n_jobs).fit(data)
    distances, indices = nbrs.kneighbors(data)
    print('Distances')
    print(distances)
    print('Indices')
    print(indices)

    sorteddist = np.sort(distances,axis=None)
    print('sorted distance')
    print(sorteddist)

    # Generating a list of eps around the mid value
    l_eps=[]
    midval=sorteddist[int(sorteddist.shape[0]/2)]
    l_eps.append(midval)
    for j in np.arange(1.01, 1.1, 0.01):
        l_eps.append(midval*j)
        l_eps.append(midval/j)
 
    # If all distances for the range are zero, just use the smallest value
    if sum(l_eps) == 0:
        l_eps=[sorteddist[bisect_right(sorteddist,0)]]

    # Running DBSCAN for each of the above eps until at least "min_clusters" is found
    winning_dbscan = None
    winning_elap_time = None

    for v_eps in l_eps:
        dbscan = DBSCAN(eps=v_eps, min_samples=min_samples,metric='euclidean',algorithm=NearestNeighborsAlg,n_jobs=p_n_jobs)
        
        # Initial time mark
        t0 = time()

        ## Compute dbscan clustering against the original data set
        dbscan.fit(data)

        # Calculate process time
        elap_time = (time() - t0)
        
        clusternum = len(np.unique([ label for label in dbscan.labels_ if label > -1]))
        if clusternum >=min_clusters:
            #print('cluster #',clusternum,'eps',v_eps)
            winning_dbscan = dbscan
            winning_elap_time = elap_time
            break

    return winning_dbscan,winning_elap_time

