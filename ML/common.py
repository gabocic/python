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
    print('sec_idx',sec_idx)
    #print('sec',np.where(estimator.labels_==idx))
    print('sec',unique[sec_idx])

    ## Now that I have the Single-element-cluster id I need to search for labels and samples that correspond to that cluster to delete them
    pos_to_delete = []
    for idx in sec_idx:
        ptd = np.where(estimator.labels_==idx)
        #pos_to_delete.append(ptd[0][0])
        data=np.delete(data,ptd[0][0],0)
        estimator.labels_=np.delete(estimator.labels_,ptd[0][0],0)

    #print('pos to delete',pos_to_delete)

    ## Rearrange tags
    for sec in sec_idx:

        # Check if the removed cluster had the highest id (no arrange is required) - we use "unique" because we have original number of clusters there already
        if sec_idx == max(unique)
            pass
        else:
            # Get the max cluster id and replace it by the cluster removed
            data[data == max(estimator.labels_)] = sec_idx



    
    # Split data into the different clusters
    samples_to_del=[]
    clusters={}
    it = np.nditer(estimator.labels_, flags=['f_index'])
    while not it.finished:
        if counts[it[0]] == 1:
            samples_to_del.append(it.index)
        else:
            clusterid = int(it[0])
            if clusterid in clusters: 
                clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
            else:
                clusters[clusterid] = np.array([data[it.index,:]])
        it.iternext()

    print('Data position to delete',samples_to_del)
    print('Cluster Id to remove:',np.take(estimator.labels_,samples_to_del))
    cleandata=np.delete(data,samples_to_del,0)
    cleanlabels=np.delete(estimator.labels_,samples_to_del,0)

    print('Samples to be considered for clustering metrics:',cleandata.shape[0],cleanlabels.shape[0])

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

    return clusters,len(samples_to_del),cleandata,cleanlabels

