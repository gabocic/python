#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import Birch
import numpy as np
from time import time

def birch_clustering(data,plot,p_n_clusters,p_n_jobs):

    ## Creating BIRCH object to process the dataset
    ## **********************************************
    # To avoid the computation of global clustering, for every call of partial_fit the user is advised
    # * To set n_clusters=None initially
    # * Train all data by multiple calls to partial_fit.
    # * Set n_clusters to a required value using brc.set_params(n_clusters=n_clusters).
    # * Call partial_fit finally with no arguments, i.e brc.partial_fit() which performs the global clustering
    birch = Birch(branching_factor=50, n_clusters=p_n_clusters, threshold=0.5,compute_labels=True)

    # Initial time mark
    t0 = time()

    ## Compute Birch clustering
    birch.fit(data)

    # Calculate process time
    elap_time = (time() - t0)

    return birch,elap_time
