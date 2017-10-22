#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from plot_2d_3d import plot_2d_3d


def k_means_clustering(data,plot=0,p_init='k-means++',p_n_clusters=3,p_n_init=10,p_n_jobs=2):

    ## Creating KMeans object to process the dataset
    ## **********************************************
    # k-means++ initialization scheme is specified (use the init='kmeans++' parameter), which has been implemented in scikit-learn. This initializes the centroids to be (generally) 
    # distant from each other, leading to probably better results than random initialization (init=random and PCA-based are two other possible choices). The parameter n_job specify 
    # the amount of processors to be used (default: 1). A value of -1 uses all available processors, with -2 using one less, and so on.

    if p_init == 'PCA-based':
        n_digits = len(np.unique(digits.target)) ## << Understand this! "digits" is making reference to the sklearn digits dataset
        pca = PCA(n_components=n_digits).fit(data)
        p_init=pca.components_

    kmeans = KMeans(init=p_init, n_clusters=p_n_clusters, n_init=p_n_init, n_jobs=p_n_jobs)

    ## Compute k-means clustering against the original data set
    kmeans.fit(data)

    ## Save centroids for plotting
    centroids=kmeans.cluster_centers_

    if plot == 1:    
        element_list=[]
        element={'type':'dot','value':centroids.T,'color':'g','marker':'x','size':90}
        element_list.append(element)

    #    element={'type':'dot','value':p1,'color':'g','marker':'x','size':90}
    #    element_list.append(element)

        plot_2d_3d(element_list,3)

    return kmeans.labels_
