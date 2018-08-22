import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataset import create_dataset
from sklearn.neighbors import NearestNeighbors
from bisect import bisect_right



def checkGroups(dataset):
    nbrs = NearestNeighbors(n_neighbors=round(dataset.shape[0]*0.05), algorithm='auto',n_jobs=4).fit(dataset)
    distances, indices = nbrs.kneighbors(dataset)

    #avgdist = np.mean(distances,axis=None)
    #print('avgdist',avgdist)

    sorteddist = np.sort(distances,axis=None)
    min_non_zero = sorteddist[bisect_right(sorteddist,0)]

    neigh = NearestNeighbors(radius=min_non_zero*1.10)
    neigh.fit(dataset) 

    i=0
    elemtaken = np.array([])
    for ejemplo in dataset:
        rng = neigh.radius_neighbors([ejemplo])
        if len(rng[1][0]) > 5:
            if i == 0:
                elemtaken = np.append(elemtaken,rng[1][0])
                i+=1
            else:
                mask = np.in1d(elemtaken, rng[1][0])
                if True in mask:
                    pass
                else:
                    elemtaken = np.append(elemtaken,rng[1][0])
                    i+=1

    #print(np.asarray(rng)) 
    print('grupos',i)

    repeatedperc=(len(elemtaken)/dataset.shape[0])*100

    print('repeatedperc',repeatedperc)


    #print(np.take(dataset,elemtaken.astype(int),axis=0))
    #dataset=np.delete(dataset,elemtaken.astype(int),axis=0)
    #print(dataset)

    return elemtaken.astype(int),repeatedperc,i

#dataset,unifo_feat,standa_feat = create_dataset(n_samples=1000, n_features=16,
#                            perc_lin=0, perc_repeated=60, n_groups=5,perc_outliers=0,
#                            debug=0,plot=0,save_to_file=0)

#checkGroups(dataset)
