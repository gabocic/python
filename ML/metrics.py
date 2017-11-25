#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics
import numpy as np
from common import get_intra_cluster_distances
from numpy.linalg import norm

def clustering_metrics(estimator, name, data, time, sample_size):


    def dunn_index(estimator,data):

        def get_inter_cluster_distances(i, j, clusters):
            distances = []
            for cluster_i_element in clusters[i]:
                for cluster_j_element in clusters[j]:
                    distances.append(norm(cluster_i_element-cluster_j_element))
            return distances

        # Split data into the different clusters
        clusters={}
        it = np.nditer(estimator.labels_, flags=['f_index'])
        while not it.finished:
            clusterid = int(it[0])
            if clusterid in clusters: 
                clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
            else:
                clusters[clusterid] = np.array([data[it.index,:]])
            it.iternext()

        # Calculates the maximum internal distance.
        l_micd = []
        for c in clusters:
            ## For each cluster, calculates the distances between the cluster points
            icd = get_intra_cluster_distances(clusters[c])
            ## The first numpy.min obtains the minimum distance for that cluster
            micd = np.max(icd)
            l_micd.append(micd)
        
        ## Obtain the minimum distance across all clusters
        max_intra_cluster_dist = np.max(l_micd)
                    
   
        # Calculate the minimum inter cluster distance

        distances = []
        for i in range(len(clusters)-1):
            for j in range(i+1,len(clusters)):
                distances.append(get_inter_cluster_distances(i, j, clusters))
        min_inter_cluster_dist = np.min(np.min(distances))

        return min_inter_cluster_dist/max_intra_cluster_dist


    proc_metrics={}
    proc_metrics['name'] =  name
    proc_metrics['time'] = time
    proc_metrics['inertia'] = estimator.inertia_
    proc_metrics['calinski_harabaz_score'] = metrics.calinski_harabaz_score(data, estimator.labels_)
    proc_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    proc_metrics['dunn_index'] = dunn_index(estimator,data)
    print(proc_metrics)

def rules_metrics(clusters,rules):

    ### <<< WIP: not working!!
    ## Create contingency Table for each Rule,Cluster pair
    #c examples covered by rule r
    #c examples not covered by rule r
    #not c examples covered by rule r
    #not c examples not covered by rule r

    d_cont_table=[[]]

    for ruleid in rules:
        for example in data:
	    if rule apply to example
	        retrieve cluster id
	        r[ruleid][clusterid] +=1 

    ### VERIFY THIS VV
    for clusterid in clusters:
        d_cont_table=[ruleid][classid]['ncr'] = r[ruleid][clusterid]
        d_cont_table=[ruleid][classid]['nc!r'] = len(clusters[clusterid]) - r[ruleid][clusterid]
        d_cont_table=[ruleid][classid]['n!cr'] = sum(r[ruleid]) - r[ruleid][clusterid]
        d_cont_table=[ruleid][classid]['n!c!r'] = (len(clusters)-len([clusterid])) - (sum(r[ruleid]) - r[ruleid][clusterid])

