#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics

def clustering_metrics(estimator, name, data, time, sample_size):


    def dunn_index(estimator,data):
        print(data)
        print(estimator.labels_)

        # Calculates the minimum internal distance.
        ## The below line is basically:
        ## For each cluster, calculates the distances between the cluster points(?)
        ## The first numpy.min obtains the minimum distance for that cluster
        ## The outer numpy.min obtains the minimum distance across all clusters
        numpy.min([numpy.min(get_intra_cluster_distances(c, matrix)) for c in clustering.clusters])
                    
    
    @classmethod
    def max_intercluster_distance(cls, clustering, matrix):



        return None

        



    proc_metrics={}
    proc_metrics['name'] =  name
    proc_metrics['time'] = time
    proc_metrics['inertia'] = estimator.inertia_
    proc_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    proc_metrics['dunn_index'] = dunn_index(estimator,data)
    print(proc_metrics) 
