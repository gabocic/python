#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from numpy.linalg import norm
from scipy.special import comb

def split_data_in_clusters(estimator,data):


    sec_idx=[]
    clusters={}
    l_outliers = []
    ignored_samples = 0
    # Estimator will be null if we were not able to find a minimum amount of clusters (see dbscan.py)
    if estimator != None:

        unique, counts = np.unique(estimator.labels_, return_counts=True)

        ## Check that at least two cluster were found
        if unique[-1] > 0:

            ## If cluster algorithm is DBSCAN we need to remove any samples considered "Outliers" (label = -1)
            if -1 in unique:
                it = np.nditer(estimator.labels_, flags=['f_index'])
                while not it.finished:
                    if it[0] == -1:
                        # Save positions to remove
                        l_outliers.append(it.index)
                    it.iternext()
            
            ## Save the number of outliers, ie samples not included in any cluster
            ignored_samples = len(l_outliers)

            
            # Look for any clusters with only one element
            sec_idx = [ idx for idx,cnt in enumerate(counts) if cnt==1 ]
            #print('Single element clusters to be removed:',sec_idx)

            ## Now that I have the Single-element-cluster id I need to search for labels and samples that correspond to that cluster to delete them
            for idx in sec_idx:

                # Search for the positions where the label to be removed is
                ptd = np.where(estimator.labels_==idx)
                l_outliers.append(ptd[0][0])

            ## Delete corresponding data and labels
            data=np.delete(data,l_outliers,0)
            estimator.labels_=np.delete(estimator.labels_,l_outliers,0)

            ## Rearrange tags
            for sec in sec_idx:

                # Check if the removed clusters were among the highest (no arrange is required)
                if sec > max(estimator.labels_):
                    pass
                else:
                    # Get the max cluster id and replace it by the cluster removed
                    estimator.labels_[estimator.labels_ == max(estimator.labels_)] = sec


            # Split data into the different clusters
            it = np.nditer(estimator.labels_, flags=['f_index'])
            while not it.finished:
                clusterid = int(it[0])
                if clusterid in clusters: 
                    clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
                else:
                    clusters[clusterid] = np.array([data[it.index,:]])
                it.iternext()
            cluster_cnt = len(clusters)
            #print('Samples to be considered for clustering metrics:',data.shape[0],estimator.labels_.shape[0])

        # If single cluster
        elif unique[-1] == 0:
            cluster_cnt = 1

        # If no clusters
        else:
            cluster_cnt = 0
        labels_ = estimator.labels_
    else:
        cluster_cnt = 0
        labels_ = None


    return clusters,len(sec_idx),data,labels_,l_outliers,cluster_cnt,ignored_samples

