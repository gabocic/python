#!/home/gabriel/pythonenvs/v3.5/bin/python

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


