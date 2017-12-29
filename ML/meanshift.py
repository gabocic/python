#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import MeanShift, estimate_bandwidth
from time import time

def meanshift_clustering(data,plot,p_n_jobs):

    ## Creating Mean-Shift object to process the dataset
    ## **********************************************
    # The bandwidth parameter dictates the size of the region to search through
    # This parameter can be set manually, but can be estimated using the provided estimate_bandwidth function, which is called if the bandwidth is not set.
    bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=50) # <-- n_samples needs to be calculated based on the dataset size!
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True,n_jobs=p_n_jobs)

    # Initial time mark
    t0 = time()

    ## Compute mean-shift clustering against the original data set
    ms.fit(data)

    # Calculate process time
    elap_time = (time() - t0)

    return ms,elap_time

