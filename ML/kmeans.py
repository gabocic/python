#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import KMeans
from plot_2d_3d import plot_2d_3d


def k_means_clustering(data):

    ## Creating KMeans object to process the dataset
    ## **********************************************
    # k-means++ initialization scheme is specified (use the init='kmeans++' parameter), which has been implemented in scikit-learn. This initializes the centroids to be (generally) 
    # distant from each other, leading to probably better results than random initialization (init=random and PCA-based are two other possible choices). The parameter n_job specify 
    # the amount of processors to be used (default: 1). A value of -1 uses all available processors, with -2 using one less, and so on.
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10, n_jobs=2)

    ## Compute k-means clustering against the original data set
    kmeans.fit(data)

    ## Save centroids for plotting
    centroids=kmeans.cluster_centers_

    print(centroids)

    ## PLOTTING RESULTS
    ## ******************************

#    element={'type':'dot','value':p1,'color':'g','marker':'x','size':90}
#    element_list.append(element)
    
#    element={'type':'dot','value':p1,'color':'g','marker':'x','size':90}
#    element_list.append(element)


#    ax.plot(data[:, 0], data[:, 1], data[:, 2],'k.', markersize=2) # <-- 3D

#    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
#                marker='x', s=169, linewidths=3,
#                color='g', zorder=10)

#    ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], # <-- 3D
#                marker='x', s=169, linewidths=3,
#                color='g', zorder=10)
