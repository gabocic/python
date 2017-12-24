#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics
import numpy as np
from common import get_intra_cluster_distances
from common import split_data_in_clusters
from numpy.linalg import norm

def clustering_metrics(estimator, name, data, time, sample_size,clusters):


    def dunn_index(estimator,data):

        def get_inter_cluster_distances(i, j, clusters):
            distances = []
            for cluster_i_element in clusters[i]:
                for cluster_j_element in clusters[j]:
                    distances.append(norm(cluster_i_element-cluster_j_element))
            return distances

        # Split data into the different clusters
        #clusters = split_data_in_clusters(estimator,data)

        #clusters={}
        #it = np.nditer(estimator.labels_, flags=['f_index'])
        #while not it.finished:
        #    clusterid = int(it[0])
        #    if clusterid in clusters: 
        #        clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
        #    else:
        #        clusters[clusterid] = np.array([data[it.index,:]])
        #    it.iternext()
        

        # Calculates the maximum internal distance.
        l_micd = []
        for c in clusters:
            ## For each cluster, calculates the distances between the cluster points
            # Ignore single element clusters
            if clusters[c].shape[0] == 1:
                print("<<<< SINGLE ELEMENT CLUSTER WAS GENERATED >>>>")
            else:
                icd = get_intra_cluster_distances(clusters[c])
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

def rules_metrics(clusters,rules,n_samples):

    ## Create contingency table for each Rule,Cluster pair:
    # -Cluster 'c' examples covered by rule 'r'
    # -Cluster 'c' examples Not covered by rule 'r'
    # -Non Cluster 'c'  covered by rule 'r'
    # -Non Cluster 'c' examples Not covered by rule r
    
    d_cont_table={}

    for ruleid in rules:
        print("Rule: "+ruleid.__str__())
        print("************************")
        if ruleid not in d_cont_table:
            d_cont_table[ruleid] = {}

        # In this case clusterid is the position of the value in the list since the clusters are also numbered by position
        for clusterid,clustercnt in enumerate(rules[ruleid]['classes_matched'][0]):
            print("Cluster",clusterid)
            print(clustercnt)
            if clusterid not in d_cont_table[ruleid]:
                d_cont_table[ruleid][clusterid] = {}
            d_cont_table[ruleid][clusterid]['ncr'] = clustercnt
            d_cont_table[ruleid][clusterid]['n!cr'] = sum(rules[ruleid]['classes_matched'][0]) - clustercnt
            d_cont_table[ruleid][clusterid]['nc!r'] = len(clusters[clusterid]) - clustercnt
            d_cont_table[ruleid][clusterid]['n!c!r'] = (n_samples - sum(rules[ruleid]['classes_matched'][0])) - (len(clusters[clusterid])+clustercnt)

    ## Weighted Sum of consistency and coverage (Michalsky, 1990)
    # Qws = w1 x cons(R) + w2 x cover(R), with
    #
    #   cons(R) = ncr / nr
    #   cover(R) = ncr / nc
    #   w1 = 0.5 + 1/4 x cons(R)
    #   w2 = 0.5 - 1/4 x cons(R)

    for rule in d_cont_table:
        print('Rule',rule)
        sum_Qws = 0
        for cluster in d_cont_table[rule]:
            cons = d_cont_table[rule][cluster]['ncr'] / (d_cont_table[rule][cluster]['ncr']+d_cont_table[rule][cluster]['n!cr'])
            cover = d_cont_table[rule][cluster]['ncr'] / (d_cont_table[rule][cluster]['ncr']+d_cont_table[rule][cluster]['nc!r'])
            w1 = 0.5 + (1/4 * cons)
            w2 = 0.5 - (1/4 * cons)
            Qws = w1 * cons + w2 * cover
            sum_Qws = sum_Qws + Qws
        avg_Qws = sum_Qws / len(clusters)
        print(avg_Qws)




