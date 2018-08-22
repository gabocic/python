from sklearn.metrics.pairwise import pairwise_distances
from dataset import create_dataset
from scipy.spatial.distance import pdist
import numpy as np

#dataset,unifo_feat,standa_feat = create_dataset(n_samples=100, n_features=3,perc_lin=0, perc_repeated=80, n_groups=4,perc_outliers=0,debug=0,plot=0,save_to_file=0)

dataset = np.array([[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5],[6,6,6]])
print(np.around(pairwise_distances(dataset),2))
condmat = pdist(dataset)
condmat = np.around(condmat,2)

groups = [[]]
print(condmat)

k=0
j=0
for i in range(0,dataset.shape[0]-1):
    j=i+1
    for k in range(k,k+dataset.shape[0]-i-1):
        print(condmat[k],'i:',i,'j:',j)

        # if the dist is smaller than the threshold
        if condmat[k] > 1:

            # Search for both points on the groups array
            i_ele_pos = np.argwhere(groups==i)
            j_ele_pos = np.argwhere(groups==j)

            # If the point is not on groups, create new group
            if i_ele_pos.size == 0 and j_ele_pos == 0:
                groups.shape[0]+1


        k+=1
        j+=1
    print('----')
