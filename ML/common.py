#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from numpy.linalg import norm
from scipy.special import comb

def split_data_in_clusters(estimator,data):

    # Check that at least one cluster was found and do not consider "-1" labels
    numclusters = len(np.unique([ label for label in estimator.labels_ if label > -1]))

    ## TODO:
        #1) Remove labels -1 only if they exist
        #2) Exit if 0 or 1 clusters were found, but return the right return variables


    if estimator != None and numclusters > 1:
        # Remove outliers
        l_outliers = []
        it = np.nditer(estimator.labels_, flags=['f_index'])
        while not it.finished:
            if it[0] == -1:
                l_outliers.append(it.index)
            it.iternext()
        estimator.labels_ = np.delete(estimator.labels_,l_outliers,0)
        data = np.delete(scaleddata,l_outliers,0)
        print('Outliers #',len(l_outliers))
    else:
        print("One or zero clusters found")
        return {},{}


    # Look for any unique labels
    unique, counts = np.unique(estimator.labels_, return_counts=True)

    sec_idx = [ idx for idx,cnt in enumerate(counts) if cnt==1 ]
    print('Single element clusters to be removed:',sec_idx)

    ## Now that I have the Single-element-cluster id I need to search for labels and samples that correspond to that cluster to delete them
    for idx in sec_idx:

        # Search for the positions where the label to be removed is
        ptd = np.where(estimator.labels_==idx)
        data=np.delete(data,ptd[0][0],0)
        estimator.labels_=np.delete(estimator.labels_,ptd[0][0],0)

    ## Rearrange tags
    for sec in sec_idx:

        # Check if the removed cluster was among the highest (no arrange is required) - we use "unique" because we have original number of clusters there already
        if sec > max(estimator.labels_):
            pass
        else:
            # Get the max cluster id and replace it by the cluster removed
            estimator.labels_[estimator.labels_ == max(estimator.labels_)] = sec


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


    print('Samples to be considered for clustering metrics:',data.shape[0],estimator.labels_.shape[0])

    return clusters,len(sec_idx),data,estimator.labels_

