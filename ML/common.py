#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from numpy.linalg import norm
from scipy.special import comb

def get_intra_cluster_distances(data):
    print("Calculating distances...","Total values to compute: ",comb(data.shape[0], 2))
    l_pp_dist=[]
    i=1
    for o_row in data:
        for i_row in data[i:data.shape[0],]:
            pp_dist = norm(o_row-i_row)
            l_pp_dist.append(pp_dist)
        i+=1
    return l_pp_dist


def split_data_in_clusters(estimator,data):

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
        #if sec == max(unique):
        if sec > max(estimator.labels_):
            pass
        else:
            # Get the max cluster id and replace it by the cluster removed
            estimator.labels_[estimator.labels_ == max(estimator.labels_)] = sec


    #unique, counts = np.unique(estimator.labels_, return_counts=True)
    
    # Split data into the different clusters
    #samples_to_del=[]
    clusters={}
    it = np.nditer(estimator.labels_, flags=['f_index'])
    while not it.finished:
        #if counts[it[0]] == 1:
        #    samples_to_del.append(it.index)
        #else:
        clusterid = int(it[0])
        if clusterid in clusters: 
            clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
        else:
            clusters[clusterid] = np.array([data[it.index,:]])
        it.iternext()


    print('Samples to be considered for clustering metrics:',data.shape[0],estimator.labels_.shape[0])

    # Remove single-element clusters
#    single_ele_clus=0
#    clus_to_remove=[]
#    for c in clusters:
#        # Saving key to remove it after the loop is done (to avoid "dictionary changed size during iteration")
#        if clusters[c].shape[0] == 1:
#            single_ele_clus+=1
#            clus_to_remove.append(c)

    # Removing single-element cluster data
#    for sec in clus_to_remove:
#        clusters.pop(sec,None)

    return clusters,len(sec_idx),data,estimator.labels_

