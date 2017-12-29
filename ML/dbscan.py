#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import DBSCAN
from time import time

def dbscan_clustering(data,plot,p_n_jobs):

    ## Creating DBSCAN object to process the dataset
    ## **********************************************
    # There are two parameters to the algorithm, min_samples and epsilon, which define formally what we mean when we say "dense". Higher min_samples or lower eps indicate 
    # higher density necessary to form a cluster.

    l_NearestNeighborsAlg = ['auto', 'ball_tree', 'kd_tree', 'brute']
    NearestNeighborsAlg = 'auto'
    dbscan = DBSCAN(eps=0.98, min_samples=3,metric='euclidean',algorithm=NearestNeighborsAlg,n_jobs=p_n_jobs)

    # Initial time mark
    t0 = time()

    ## Compute dbscan clustering against the original data set
    dbscan.fit(data)

    # Calculate process time
    elap_time = (time() - t0)

    return dbscan,elap_time

