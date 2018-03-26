#!/home/gabriel/pythonenvs/v3.5/bin/python

from time import time
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plot_2d_3d import plot_2d_3d
from metrics import clustering_metrics
from common import split_data_in_clusters
from collections import Counter

def k_means_clustering(data,plot,p_init,p_n_clusters,p_n_init,p_n_jobs):

    ## Creating KMeans object to process the dataset
    ## **********************************************
    # k-means++ initialization scheme is specified (use the init='kmeans++' parameter), which has been implemented in scikit-learn. This initializes the centroids to be (generally) 
    # distant from each other, leading to probably better results than random initialization (init=random and PCA-based are two other possible choices). The parameter n_job specify 
    # the amount of processors to be used (default: 1). A value of -1 uses all available processors, with -2 using one less, and so on.

    # We need to run this for different number of clusters and determine which one has better parameters
  
    all_metrics={}

    # #0: silhouette_score  #1:calinski_harabaz_score #2: dunn_index  #3: sin_ele_clus
    metrics_winners=[0,0,0,0]
    metrics_win_val=[0,0,0,0]

    l_clus_range=3
    h_clus_range=10
    flag_sel=0 #flag to detect single element clusters
    for n_clusters in range(l_clus_range,h_clus_range+1):
        if p_init == 'PCA-based':
            pca = PCA(n_components=p_n_clusters).fit(data)
            p_init=pca.components_
            p_n_init=1

        kmeans = KMeans(init=p_init, n_clusters=n_clusters, n_init=p_n_init, n_jobs=p_n_jobs)
        
        # Initial time mark
        t0 = time()

        ## Compute k-means clustering against the original data set
        kmeans.fit(data)

        # Calculate process time
        elap_time = (time() - t0)

        # Split data in clusters
        clusters,sin_ele_clus,cleandata,cleanlabels = split_data_in_clusters(kmeans,data)
        for singleclus in clusters:
            print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))

        
        # Calculate metrics. To solve ties, I'm prioritizing runs with less amount of clusters
        clus_metrics = clustering_metrics(cleanlabels, 'k-means_'+n_clusters.__str__()+'clus',cleandata, elap_time, None, clusters,sin_ele_clus)
        
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
        
        # Save metrics 
        all_metrics[n_clusters] = clus_metrics
        
        print(metrics_winners)
        print(metrics_win_val)

        # If no iteration generated single element clusters, do not consider this metric 
    if flag_sel == 0:
        del metrics_winners[3]
        del metrics_win_val[3]
    ocurrences = Counter(metrics_winners)
    print(ocurrences)


#    for metric_group in all_metrics:
#        print(all_metrics[metric_group])

    ## Save centroids for plotting
    centroids=kmeans.cluster_centers_

    # Plot values
    if plot == 1:    
        element_list=[]
        element={'type':'dot','value':centroids.T,'color':'g','marker':'x','size':90}
        element_list.append(element)
        plot_2d_3d(element_list,3)

    return kmeans,elap_time
