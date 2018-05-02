#!/home/gabriel/pythonenvs/v3.5/bin/python

from time import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plot_2d_3d import plot_2d_3d
from common import split_data_in_clusters
from collections import Counter
import numpy as np
from procmetrics import wb_index

def k_means_clustering(data,plot,p_init,p_n_init,p_n_jobs):

    ## Creating KMeans object to process the dataset
    ## **********************************************
    # k-means++ initialization scheme is specified (use the init='kmeans++' parameter), which has been implemented in scikit-learn. This initializes the centroids to be (generally) 
    # distant from each other, leading to probably better results than random initialization (init=random and PCA-based are two other possible choices). The parameter n_job specify 
    # the amount of processors to be used (default: 1). A value of -1 uses all available processors, with -2 using one less, and so on.

    # We need to run this for different number of clusters and determine which one has better parameters
  
    estimators = []
    wbindexes = []
    elaptimes = []

    l_clus_range=3

    if p_init == 'PCA-based':
        h_clus_range=data.shape[1]
    else:
        h_clus_range=10

    flag_sel=0 #flag to detect single element clusters

    for n_clusters in range(l_clus_range,h_clus_range+1):
        if p_init == 'PCA-based':
            pca = PCA(n_components=n_clusters).fit(data)
            centinit=pca.components_
            p_n_init=1
        else:
            centinit=p_init

        kmeans = KMeans(init=centinit, n_clusters=n_clusters, n_init=p_n_init, n_jobs=p_n_jobs)
        
        # Initial time mark
        t0 = time()

        ## Compute k-means clustering against the original data set
        kmeans.fit(data)

        # Calculate process time
        elap_time = (time() - t0)

        # Split data in clusters
        clusters,sin_ele_clus,cleandata,cleanlabels,l_outliers,cluster_cnt = split_data_in_clusters(kmeans,data)
        #for singleclus in clusters:
        #    print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))

        
        ### As per the "Investigation of Internal Validity Measures for K-Means Clustering" paper, the Sum-of-squares method was found to be the most effective for predicting the 'best' number of clusters        ### To calculate the Sum-of-Squares index we need to calculate the Sum-of-Squares 'within the clusters' (SSW) and the Sum-of-squares 'between the clusters' (SSB)

        wb = wb_index(clusters,data)

        ## Store the WB indexes, estimators and elapsed times
        wbindexes.append(wb)
        estimators.append(kmeans)
        elaptimes.append(elap_time)

    ## Find the winner "K" by looking for the minimum WB index
    minwbidx = wbindexes.index(min(wbindexes))
    #print(min(wbindexes))

    ## Save centroids for plotting
    centroids=kmeans.cluster_centers_

    # Plot values
    if plot == 1:    
        element_list=[]
        element={'type':'dot','value':centroids.T,'color':'g','marker':'x','size':90}
        element_list.append(element)
        plot_2d_3d(element_list,3)

    return estimators[minwbidx],elaptimes[minwbidx]
