#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import Birch
import numpy as np
from time import time
from sklearn.metrics import calinski_harabaz_score,silhouette_score
from procmetrics import dunn_index,wb_index
from common import split_data_in_clusters
from collections import Counter


def birch_clustering(data,plot,p_n_jobs):

    ## Creating BIRCH object to process the dataset
    ## **********************************************
    # To avoid the computation of global clustering, for every call of partial_fit the user is advised
    # * To set n_clusters=None initially
    # * Train all data by multiple calls to partial_fit.
    # * Set n_clusters to a required value using brc.set_params(n_clusters=n_clusters).
    # * Call partial_fit finally with no arguments, i.e brc.partial_fit() which performs the global clustering


    def clustering_metrics():
        clus_metrics={}
        clus_metrics['wb_index'] = wb_index(clusters,cleandata)
        clus_metrics['dunn_index'] = dunn_index(clusters)
        clus_metrics['calinski_harabaz_score'] = calinski_harabaz_score(cleandata, cleanlabels)
        clus_metrics['silhouette_score'] = silhouette_score(cleandata, cleanlabels,metric='euclidean',sample_size=None)
        clus_metrics['sin_ele_clus'] = sin_ele_clus
        return clus_metrics


    # We need to run this for different number of clusters and determine which one has better parameters
    all_metrics={}

    # #0: silhouette_score  #1:calinski_harabaz_score #2: dunn_index  #3: sin_ele_clus  #4: wb_index 
    metrics_winners=[0,0,0,0,0]
    metrics_win_val=[0,0,0,0,0]

    est_and_elap = {}

    l_clus_range=3
    h_clus_range=10

    flag_sel=0 #flag to detect single element clusters
    for n_clusters in range(l_clus_range,h_clus_range+1):
        birch = Birch(branching_factor=50, n_clusters=n_clusters, threshold=0.5,compute_labels=True)

        # Initial time mark
        t0 = time()

        ## Compute Birch clustering
        birch.fit(data)

        # Calculate process time
        elap_time = (time() - t0)


        # Split data in clusters
        clusters,sin_ele_clus,cleandata,cleanlabels,l_outliers,cluster_cnt = split_data_in_clusters(birch,data)
        #for singleclus in clusters:
        #    print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))

        ## Store the estimators and elapsed times
        est_and_elap[n_clusters]={'estimator':birch,'elaptime':elap_time}
        
        # Calculate metrics. To solve ties, I'm prioritizing runs with less amount of clusters
        clus_metrics = clustering_metrics()
        
        # Find the highest metric values. To solve ties, I'm prioritizing runs with less amount of clusters
        if n_clusters > l_clus_range:
            if clus_metrics['silhouette_score'] > metrics_win_val[0]:
                metrics_winners[0] = n_clusters
                metrics_win_val[0] = clus_metrics['silhouette_score']
            
            if clus_metrics['calinski_harabaz_score'] > metrics_win_val[1]:
                metrics_winners[1] = n_clusters
                metrics_win_val[1] = clus_metrics['calinski_harabaz_score']
            
            if clus_metrics['dunn_index'] > metrics_win_val[2]:
                metrics_winners[2] = n_clusters
                metrics_win_val[2] = clus_metrics['dunn_index']
            
            if clus_metrics['sin_ele_clus'] < metrics_win_val[3]:
                metrics_winners[3] = n_clusters
                metrics_win_val[3] = clus_metrics['sin_ele_clus']

            if clus_metrics['wb_index'] < metrics_win_val[4]:
                metrics_winners[4] = n_clusters
                metrics_win_val[4] = clus_metrics['wb_index']

            if clus_metrics['sin_ele_clus'] > 0:
                flag_sel=1

        else:
            metrics_winners[0] = n_clusters
            metrics_win_val[0] = clus_metrics['silhouette_score']
            metrics_winners[1] = n_clusters
            metrics_win_val[1] = clus_metrics['calinski_harabaz_score']
            metrics_winners[2] = n_clusters
            metrics_win_val[2] = clus_metrics['dunn_index']
            metrics_winners[3] = n_clusters
            metrics_win_val[3] = clus_metrics['sin_ele_clus']
            metrics_winners[4] = n_clusters
            metrics_win_val[4] = clus_metrics['wb_index']
        
        # Save metrics 
        #all_metrics[n_clusters] = clus_metrics
        
    #print(metrics_winners)
    #print(metrics_win_val)

    # If no iteration generated single element clusters, do not consider this metric 
    if flag_sel == 0:
        del metrics_winners[3]
        del metrics_win_val[3]
    
    most_common,num_most_common = Counter(metrics_winners).most_common(1)[0] 
    #print(most_common,num_most_common)


    #for metric_group in all_metrics:
    #    print(all_metrics[metric_group])


    return est_and_elap[most_common]['estimator'],est_and_elap[most_common]['elaptime']
