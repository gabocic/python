#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
from dataset import create_dataset
from analyze import analyze_dataset
from preprocessing import sklearn_scale
from metrics import clustering_metrics
from metrics import rules_metrics
from kmeans import k_means_clustering
from dbscan import dbscan_clustering
from birch import birch_clustering
from meanshift import meanshift_clustering
from CART import CART_classifier
from CN2 import CN2_classifier
from common import split_data_in_clusters
import numpy as np

class bcolors:
    BANNER = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def fatal_error():
    sys.exit()

def dataset_generation_and_validation(p_n_features,p_n_samples,p_perc_lin,p_perc_repeated,p_n_groups,p_perc_outliers):

    

    print("")
    print("")
    print("Dataset generation")
    print("*"*70)
    print("")

    # Generate dataset
    dataset = create_dataset(n_samples=p_n_samples, n_features=p_n_features,
                        perc_lin=p_perc_lin, perc_repeated=p_perc_repeated, n_groups=p_n_groups,perc_outliers=p_perc_outliers,
                        debug=1,plot=1,save_to_file=0)
    
    if dataset.shape == (1,0):
        fatal_error()


    print("")
    print("")
    print("#"*70)
    print("")
    print("")
    print("Dataset validation")
    print("*"*70)
    print("")

    # Validate dataset is within the specifications
    analyze_dataset(data=dataset,debug=0,plot=0,load_from_file=None)


    return dataset

def process_and_analyze():

    l_clustering_alg = [
            'kmeans_++',
            'kmeans_random',
            'kmeans_pca',
            'dbscan',
            'birch',
            'meanshift',
            ]
    l_ruleind_alg = [
            'cart',
            'cn2'
            ]

    n_clusters = 3 # only for the algorithms that support this
    clustering_alg = 'meanshift'
    rulesind_alg = 'cn2'

    # Scale data
    scaleddata = sklearn_scale(dataset) 

    # Clustering phase
    
    if clustering_alg == 'kmeans_++':
        estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='k-means++',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'kmeans_random':
        estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='random',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'kmeans_pca':
        estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='PCA-based',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'dbscan':
        estimator,elap_time = dbscan_clustering(data=scaleddata,plot=0,p_n_jobs=4)
        
        # Remove outliers
        l_outliers = []
        it = np.nditer(estimator.labels_, flags=['f_index'])
        while not it.finished:
            if it[0] == -1:
                l_outliers.append(it.index)
            it.iternext()
        estimator.labels_ = np.delete(estimator.labels_,l_outliers,0)
        scaleddata = np.delete(scaleddata,l_outliers,0)
        dataset = np.delete(dataset,l_outliers,0)
        print('Outliers #',len(l_outliers))

    elif clustering_alg == 'birch':
        estimator,elap_time = birch_clustering(data=scaleddata,plot=0,p_n_clusters=n_clusters,p_n_jobs=4)
    elif clustering_alg == 'meanshift':
        estimator,elap_time = meanshift_clustering(data=scaleddata,plot=0,p_n_jobs=4)

    else:
        print('Clustering algorithm not found')
        sys.exit()


    # Split data in clusters
    print('Split data in clusters')
    clusters = split_data_in_clusters(estimator,scaleddata)
    for singleclus in clusters:
        print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))
    
    #sys.exit()
    # Compute clustering metrics
    sample_size = 50
    clustering_metrics(estimator, clustering_alg, scaleddata, elap_time, sample_size, clusters)

    # Induct group membership rules

    if rulesind_alg == 'cart':
        rules = CART_classifier(dataset,estimator)
    elif rulesind_alg == 'cn2':
        rules = CN2_classifier(dataset,estimator)
    else:
        print('Rules induction algorithm not found')
        sys.exit()

    # Compute rules metrics
    rules_metrics(clusters,rules,n_samples)

if __name__ == '__main__':
    dataset_generation_and_validation(3,100,40,0,2,20)
