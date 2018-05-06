#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
from dataset import create_dataset
from analyze import analyze_dataset
from procmetrics import rules_metrics
from kmeans import k_means_clustering
from dbscan import dbscan_clustering
from birch import birch_clustering
from meanshift import meanshift_clustering
from CART import CART_classifier
from CN2 import CN2_classifier
from common import split_data_in_clusters
import numpy as np
from parameters import *
from sklearn.metrics import calinski_harabaz_score,silhouette_score
from procmetrics import dunn_index,wb_index
from collections import Counter


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

    #print("")
    #print("")
    #print("Dataset generation")
    #print("*"*70)
    #print("")

    # Generate dataset
    dscount=0
    while dscount < dataset_gen_retry:
        dataset = create_dataset(n_samples=p_n_samples, n_features=p_n_features,
                            perc_lin=p_perc_lin, perc_repeated=p_perc_repeated, n_groups=p_n_groups,perc_outliers=p_perc_outliers,
                            debug=0,plot=0,save_to_file=0)
        
        if dataset.shape == (1,0):
            fatal_error()

        #print("")
        #print("")
        #print("")
        #print("Dataset validation")
        #print("*"*70)
        #print("")

        # Validate dataset is within the specifications
        analysis_results = analyze_dataset(data=dataset,debug=0,plot=0,load_from_file=None)
       
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
        

        #print(analysis_results)

        if analysis_results['samples'] == p_n_samples and \
                ol_lowlimit <= analysis_results['outliersperc'] < ol_highlimit and \
                lin_lowlimit <= analysis_results['linpointsperc'] < lin_highlimit and \
                rep_lowlimit <= analysis_results['repeatedperc'] < rep_highlimit and \
                analysis_results['features'] == p_n_features and \
                analysis_results['repeatedgrps'] == p_n_groups:
            #print('DATASET IS OK!!')
            break
        else:
            #print('INVALID DATASET')
            pass
        dscount+=1   
    if dscount == dataset_gen_retry:
        print('Unable to generate a valid dataset after '+dataset_gen_retry.__str__()+' attempts')
        dataset=np.zeros([0])

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

    # Scale data
    scaleddata = StandardScaler().fit_transform(dataset)

    # Clustering phase

    #print("")
    #print("")
    #print("Clusters discovery")
    #print("*"*70)
    #print("")
    
    if clustering_alg == 'kmeans_++':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='k-means++',p_n_init=10,p_n_jobs=parallelism)
    elif clustering_alg == 'kmeans_random':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='random',p_n_init=10,p_n_jobs=parallelism)
    elif clustering_alg == 'kmeans_pca':
        estimator,c_elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='PCA-based',p_n_init=10,p_n_jobs=parallelism)
    elif clustering_alg == 'dbscan':
        estimator,c_elap_time = dbscan_clustering(data=scaleddata,plot=0,p_n_jobs=parallelism)
    elif clustering_alg == 'birch':
        estimator,c_elap_time = birch_clustering(data=scaleddata,plot=0,p_n_jobs=parallelism)
    elif clustering_alg == 'meanshift':
        estimator,c_elap_time = meanshift_clustering(data=scaleddata,plot=0,p_n_jobs=parallelism)

    else:
        print('Clustering algorithm not found')
        return {},{}


    # Split data in clusters
    clusters,sin_ele_clus,cleanscaleddata,cleanlabels,samples_to_delete,cluster_cnt,ignored_samples= split_data_in_clusters(estimator,scaleddata)


    for singleclus in clusters:
        print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))
     
    #print("")
    #print("")
    #print("Calculate cluster metrics")
    #print("*"*70)
    #print("")

    # Compute clustering metrics
    clus_metrics={}

    clus_metrics['name'] = clustering_alg
    clus_metrics['sin_ele_clus'] = sin_ele_clus
    clus_metrics['cluster_cnt'] = cluster_cnt
    clus_metrics['ignored_samples'] = ignored_samples

    # Check that more than 1 cluster was found
    if cluster_cnt <= 1:
        print('Less than ',min_clusters,' clusters found. Skipping metrics calculation')
        clus_metrics['dunn_index'] = None
        clus_metrics['calinski_harabaz_score'] = None
        clus_metrics['silhouette_score'] = None
        clus_metrics['time'] = 0
        clus_metrics['wb_index'] = None
        return clus_metrics,{}
    else:
        clus_metrics['wb_index'] = round(wb_index(clusters,cleanscaleddata),metric_decimals)
        clus_metrics['time'] = round(c_elap_time,metric_decimals)
        clus_metrics['dunn_index'] = round(dunn_index(clusters),metric_decimals)
        clus_metrics['calinski_harabaz_score'] = round(calinski_harabaz_score(cleanscaleddata, cleanlabels),metric_decimals)
        clus_metrics['silhouette_score'] = round(silhouette_score(cleanscaleddata, cleanlabels,metric='euclidean',sample_size=None),metric_decimals)

    return clus_metrics,{}

    # Induct group membership rules

    # Remove samples that were discarded for the clustering phase
    cleandata = np.delete(dataset,samples_to_delete,0)

    #print("")
    #print("")
    #print("Membership rules induction")
    #print("*"*70)
    #print("")

    if rulesind_alg == 'cart':
        rules,r_elap_time,classes = CART_classifier(cleandata,cleanlabels)
    elif rulesind_alg == 'cn2':
        rules,r_elap_time = CN2_classifier(cleandata,cleanlabels)
    else:
        #print('Rules induction algorithm not found')
        return clus_metrics,{}

    print('Rules generated:',len(rules))
    
    # Compute rules metrics
    #print("")
    #print("")
    #print("Calculate rules metrics")
    #print("*"*70)
    #print("")


    rulind_metrics = rules_metrics(clusters,rules,cleandata.shape[0],round(r_elap_time,metric_decimals))
    
    return clus_metrics,rulind_metrics

if __name__ == '__main__':

    paramlist = []
#    paramlist.append([8,1000,0,0,0,0])
#    paramlist.append([8,1000,10,0,0,0])
#    paramlist.append([16,1000,10,0,0,0])
#    paramlist.append([24,1000,10,0,0,0])
#    paramlist.append([8,1000,40,0,0,0])
#    paramlist.append([8,1000,80,0,0,0])
#    paramlist.append([8,1000,0,10,2,0])
#    paramlist.append([8,1000,0,40,2,0])
#    paramlist.append([8,1000,0,80,2,0])
#    paramlist.append([8,1000,0,40,3,0])
#    paramlist.append([8,1000,0,80,3,0])
#    paramlist.append([8,1000,0,40,4,0])
#    paramlist.append([8,1000,0,80,4,0])
#    paramlist.append([8,1000,0,0,0,6])
#    paramlist.append([8,1000,0,0,0,12])
    paramlist.append([8,1000,0,0,0,18])
#    paramlist.append([8,1000,10,10,2,6])
#    paramlist.append([8,1000,10,10,2,12])
#    paramlist.append([8,1000,10,10,2,18])
#    paramlist.append([8,1000,10,40,2,6])
#    paramlist.append([8,1000,10,40,2,12])
#    paramlist.append([8,1000,10,40,2,18])
#    paramlist.append([8,1000,40,10,2,6])
#    paramlist.append([8,1000,40,10,2,12])
#    paramlist.append([8,1000,40,10,2,18])
#    paramlist.append([8,1000,40,40,2,6])

    for params in paramlist:
        print('')
        print('')
        print('================================================================================================================')
        print('================================================================================================================')
        print('********************** ',params,'*********************************')
        print('================================================================================================================')
        print('================================================================================================================')
        print('')
        print('')
        dataset = dataset_generation_and_validation(*params)
        if dataset.shape[0] == 0:
            sys.exit()

        else:
            pass
#            process_and_analyze(dataset,'dbscan','cn2')
#            sys.exit()

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
                    #'cn2'
                    ]
            all_metrics = {}
            metrics_winners=[0,0,0,0]
            metrics_win_val=[0,0,0,0]
            for caidx,clustering_alg in enumerate(l_clustering_alg):
                for riaidx,ruleind_alg in enumerate(l_ruleind_alg):
                    print('')
                    print('###############################################')
                    print(clustering_alg.upper(),'&',ruleind_alg.upper())
                    print('###############################################')
                    print('')
                    clus_metrics,rulind_metrics = process_and_analyze(dataset,clustering_alg,ruleind_alg)
                    print(clus_metrics)
                    print('')
                    print(rulind_metrics)

                    if clus_metrics['cluster_cnt'] > 1:
                    
                        if metrics_winners[0] == 0:
                            metrics_winners[0] = clus_metrics['name']
                            metrics_win_val[0] = clus_metrics['silhouette_score']
                            metrics_winners[1] = clus_metrics['name']
                            metrics_win_val[1] = clus_metrics['calinski_harabaz_score']
                            metrics_winners[2] = clus_metrics['name']
                            metrics_win_val[2] = clus_metrics['dunn_index']
                            metrics_winners[3] = clus_metrics['name']
                            metrics_win_val[3] = clus_metrics['wb_index']
                        else:
                            if clus_metrics['silhouette_score'] > metrics_win_val[0]:
                                metrics_winners[0] = clus_metrics['name']
                                metrics_win_val[0] = clus_metrics['silhouette_score']
                            
                            if clus_metrics['calinski_harabaz_score'] > metrics_win_val[1]:
                                metrics_winners[1] = clus_metrics['name']
                                metrics_win_val[1] = clus_metrics['calinski_harabaz_score']
                            
                            if clus_metrics['dunn_index'] > metrics_win_val[2]:
                                metrics_winners[2] = clus_metrics['name']
                                metrics_win_val[2] = clus_metrics['dunn_index']
                            
                            if clus_metrics['wb_index'] < metrics_win_val[3]:
                                metrics_winners[3] = clus_metrics['name']
                                metrics_win_val[3] = clus_metrics['wb_index']
                            
                    # Save metrics 
                    all_metrics[clus_metrics['name']] = clus_metrics
                    
                    print(metrics_winners)
                    print(metrics_win_val)

            ocurrences = Counter(metrics_winners)
            print(ocurrences)
            winners_cnt = max(ocurrences.values())
            winners_idx = [i for i, j in enumerate(ocurrences.values()) if j == winners_cnt]

            ocurrkeys = list(ocurrences.keys())
            if len(winners_idx) == 1:
                print('The winner is ',ocurrkeys[winners_idx[0]])
            else:
                print('We have a tie')
                flag_sel=0 #flag to detect single element clusters
                flag_is=0 #flag to detect ignored samples
                metrics_winners=[0,0,0]
                metrics_win_val=[0,0,0]
                for winner_idx in winners_idx:
                    algname = ocurrkeys[winner_idx]
                    print(ocurrkeys[winner_idx])
                    if metrics_winners[0] == 0:
                        metrics_winners[0] = all_metrics[algname]['name']
                        metrics_win_val[0] = all_metrics[algname]['time']
                        metrics_winners[1] = all_metrics[algname]['name']
                        metrics_win_val[1] = all_metrics[algname]['sin_ele_clus']
                        metrics_winners[2] = all_metrics[algname]['name']
                        metrics_win_val[2] = all_metrics[algname]['ignored_samples']
                    else:
                        if all_metrics[algname]['time'] < metrics_win_val[0]:
                            metrics_winners[0] = all_metrics[algname]['name']
                            metrics_win_val[0] = all_metrics[algname]['time']
                        
                        if all_metrics[algname]['sin_ele_clus'] < metrics_win_val[1]:
                            metrics_winners[1] = all_metrics[algname]['name']
                            metrics_win_val[1] = all_metrics[algname]['sin_ele_clus']
                        if all_metrics[algname]['sin_ele_clus'] > 0:
                            flag_sel=1
                
                        if all_metrics[algname]['ignored_samples'] < metrics_win_val[2]:
                            metrics_winners[2] = all_metrics[algname]['name']
                            metrics_win_val[2] = all_metrics[algname]['ignored_samples']
                        if all_metrics[algname]['ignored_samples'] > 0:
                            flag_is=1
                    print(metrics_winners)
                    print(metrics_win_val)

                # If no iteration generated single element clusters, do not consider this metric 
                if flag_sel == 0:
                    del metrics_winners[1]
                    del metrics_win_val[1]

                # If no iteration has ingnore samples, do not consider this metric
                if flag_is == 0:
                    del metrics_winners[-1]
                    del metrics_win_val[-1]

                ocurrences = Counter(metrics_winners)
                print(ocurrences)
                winners_cnt = max(ocurrences.values())
                winners_idx = [i for i, j in enumerate(ocurrences.values()) if j == winners_cnt]

                ocurrkeys = list(ocurrences.keys())
                if len(winners_idx) == 1:
                    print('The winner is ',ocurrkeys[winners_idx[0]])
                else:
                    print('We have a tie')






