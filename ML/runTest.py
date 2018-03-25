#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
from dataset import create_dataset
from analyze import analyze_dataset
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
from parameters import *

from sklearn.preprocessing import StandardScaler

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
    dscount=0
    while dscount < dataset_gen_retry:
        dataset = create_dataset(n_samples=p_n_samples, n_features=p_n_features,
                            perc_lin=p_perc_lin, perc_repeated=p_perc_repeated, n_groups=p_n_groups,perc_outliers=p_perc_outliers,
                            debug=1,plot=0,save_to_file=0)
        
        if dataset.shape == (1,0):
            fatal_error()

        print("")
        print("")
        print("")
        print("Dataset validation")
        print("*"*70)
        print("")

        # Validate dataset is within the specifications
        analysis_results = analyze_dataset(data=dataset,debug=1,plot=0,load_from_file=None)
       
        # Linear points ranges
        if 0 <= p_perc_lin < 20:
            lin_lowlimit = 0
            lin_highlimit = 20
        elif 20 <= p_perc_lin < 60:
            lin_lowlimit = 20
            lin_highlimit = 60
        elif 60 <= p_perc_lin < 100:
            lin_lowlimit = 60
            lin_highlimit = 100


        if p_perc_repeated == 0:
            rep_lowlimit = 0
            rep_highlimit = 5
        else:
            rep_lowlimit = p_perc_repeated*(1-dataset_parameters_error_margin)
            rep_highlimit = p_perc_repeated*(1+dataset_parameters_error_margin)


        if p_perc_outliers == 0:
            ol_lowlimit = 0
            ol_highlimit = 5
        else:
            ol_lowlimit = p_perc_outliers*(1-dataset_parameters_error_margin)
            ol_highlimit = p_perc_outliers*(1+dataset_parameters_error_margin)
        

        print(analysis_results)

        if analysis_results['samples'] == p_n_samples and \
                ol_lowlimit <= analysis_results['outliersperc'] < ol_highlimit and \
                lin_lowlimit <= analysis_results['linpointsperc'] < lin_highlimit and \
                rep_lowlimit <= analysis_results['repeatedperc'] < rep_highlimit and \
                analysis_results['features'] == p_n_features and \
                analysis_results['repeatedgrps'] == p_n_groups:
            print('DATASET IS OK!!')
            break
        else:
            print('INVALID DATASET')
        dscount+=1    
    return dataset

def process_and_analyze(dataset,clustering_alg,rulesind_alg):

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

    # Scale data
    scaleddata = StandardScaler().fit_transform(dataset)

    # Clustering phase

    print("")
    print("")
    print("Clusters discovery")
    print("*"*70)
    print("")
    
    if clustering_alg == 'kmeans_++':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='k-means++',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'kmeans_random':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='random',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'kmeans_pca':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='PCA-based',p_n_clusters=n_clusters,p_n_init=10,p_n_jobs=4)
    elif clustering_alg == 'dbscan':
        estimator,c_elap_time = dbscan_clustering(data=scaleddata,plot=0,p_n_jobs=4)
    

        # Check that at least one cluster was found

        if estimator != None and len(np.unique([ label for label in estimator.labels_ if label > -1])) >=1:
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
        else:
            print('No clusters were found')
            return -1

    elif clustering_alg == 'birch':
        estimator,c_elap_time = birch_clustering(data=scaleddata,plot=0,p_n_clusters=n_clusters,p_n_jobs=4)
    elif clustering_alg == 'meanshift':
        estimator,c_elap_time = meanshift_clustering(data=scaleddata,plot=0,p_n_jobs=4)

    else:
        print('Clustering algorithm not found')
        return -1


    # Split data in clusters
    clusters,sin_ele_clus,cleandata,cleanlabels = split_data_in_clusters(estimator,scaleddata)

    # Check that more than 1 cluster was found
    if len(clusters) <= 1:
        print('Single cluster found. Exiting..')
        return -1

    for singleclus in clusters:
        print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))
     
    # Compute clustering metrics
    print("")
    print("")
    print("Calculate cluster metrics")
    print("*"*70)
    print("")
    sample_size = None
    clus_metrics = clustering_metrics(cleanlabels, clustering_alg,cleandata, c_elap_time, sample_size, clusters,sin_ele_clus)
    print(clus_metrics)

    # Induct group membership rules

    print("")
    print("")
    print("Membership rules induction")
    print("*"*70)
    print("")

    if rulesind_alg == 'cart':
        rules,r_elap_time = CART_classifier(dataset,estimator.labels_)
    elif rulesind_alg == 'cn2':
        rules,r_elap_time = CN2_classifier(dataset,estimator.labels_)
    else:
        print('Rules induction algorithm not found')
        return -1

    #for ruleid in rules:
    #    print(ruleid,rules[ruleid]['classes_matched'])
    print('Rules generated:',len(rules))
    
    # Compute rules metrics
    print("")
    print("")
    print("Calculate rules metrics")
    print("*"*70)
    print("")
    metrics = rules_metrics(clusters,rules,cleandata.shape[0],r_elap_time)
    print(metrics)

if __name__ == '__main__':

    dataset = dataset_generation_and_validation(8,1000,0,0,0,0)
    process_and_analyze(dataset,'kmeans_++','cn2')

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

    lowest_time=0
#    for clustering_alg in l_clustering_alg:
#        for ruleind_alg in l_ruleind_alg:
#            process_and_analyze(dataset,clustering_alg,ruleind_alg)

