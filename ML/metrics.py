#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics
import numpy as np

def clustering_metrics(estimator, name, data, time, sample_size):


    def get_intra_cluster_distances(c,clusters):
        print("in")
        print(clusters[c])

    def dunn_index(estimator,data):

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

        # Calculates the minimum internal distance.
        ## The below line is basically:
        ## For each cluster, calculates the distances between the cluster points(?)
        ## The first numpy.min obtains the minimum distance for that cluster
        ## The outer numpy.min obtains the minimum distance across all clusters
        [get_intra_cluster_distances(c,clusters) for c in clusters]
        #numpy.min([numpy.min(get_intra_cluster_distances(c)) for c in clustering.clusters])
                    
    
    #def max_intercluster_distance(cls, clustering, matrix):



    #    return None

        



    proc_metrics={}
    proc_metrics['name'] =  name
    proc_metrics['time'] = time
    proc_metrics['inertia'] = estimator.inertia_
    proc_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    proc_metrics['dunn_index'] = dunn_index(estimator,data)
    print(proc_metrics) 
