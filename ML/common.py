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

    # Remove single-element clusters
    single_ele_clus=0
    clus_to_remove=[]
    for c in clusters:
        # Saving key to remove it after the loop is done (to avoid "dictionary changed size during iteration")
        if clusters[c].shape[0] == 1:
            single_ele_clus+=1
            clus_to_remove.append(c)

    # Removing single-element cluster data
    for sec in clus_to_remove:
        clusters.pop(sec,None)

    return clusters,single_ele_clus
