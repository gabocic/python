#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
import warnings
from dataset import create_dataset
from analyze import analyze_dataset
from procmetrics import rules_metrics
from procmetrics import rule_induction_process_metric
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
from davies_bouldin import davies_bouldin_score
from sklearn.preprocessing import StandardScaler
import mysql
from sklearn.model_selection import train_test_split
from numpy import genfromtxt


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
        dataset,unifo_feat,standa_feat = create_dataset(n_samples=p_n_samples, n_features=p_n_features,
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

    return dataset,unifo_feat,standa_feat,analysis_results

def clustering_and_metrics(dataset,clustering_alg):

    samples_to_delete=np.array([])
    cleanlabels=np.array([])
    clusters={}

    l_clustering_alg = [
            'kmeans_++',
            'kmeans_random',
            'kmeans_pca',
            'dbscan',
            'birch',
            'meanshift',
            ]

    # Scale data
    scaleddata = StandardScaler().fit_transform(dataset)

    # Clustering phase

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
        return {},samples_to_delete,cleanlabels,{}


    # Split data in clusters
    clusters,sin_ele_clus,cleanscaleddata,cleanlabels,samples_to_delete,cluster_cnt,ignored_samples= split_data_in_clusters(estimator,scaleddata)


    for singleclus in clusters:
        print('Cluster '+singleclus.__str__()+':',len(clusters[singleclus]))
    

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
        clus_metrics['davies_bouldin_score'] = None
    else:
        clus_metrics['time'] = round(c_elap_time,metric_decimals)
        clus_metrics['wb_index'] = float(round(wb_index(clusters,cleanscaleddata),metric_decimals))
        clus_metrics['dunn_index'] = float(round(dunn_index(clusters),metric_decimals))
        clus_metrics['calinski_harabaz_score'] = float(round(calinski_harabaz_score(cleanscaleddata, cleanlabels),metric_decimals))
        clus_metrics['silhouette_score'] = float(round(silhouette_score(cleanscaleddata, cleanlabels,metric='euclidean',sample_size=None),metric_decimals)) # forcing data type due to insert error

        # Supress expected runtime "divide by zero" warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clus_metrics['davies_bouldin_score'] = float(round(davies_bouldin_score(cleanscaleddata,cleanlabels),metric_decimals))

    return clus_metrics,samples_to_delete,cleanlabels,clusters

def rule_induction_and_metrics(dataset,rulesind_alg,samples_to_delete,cleanlabels,clusters):

    l_ruleind_alg = [
            'cart',
            'cn2'
            ]

    # Induct group membership rules

    # Remove samples that were discarded for the clustering phase
    cleandata = np.delete(dataset,samples_to_delete,0)

    ## Split the dataset into a training and and a testing set
    X_train, X_test, y_train, y_test = train_test_split(cleandata,cleanlabels,test_size=0.2,random_state=0)


    if rulesind_alg == 'cart':
        #rules,r_elap_time,predicted_labels,y_test,predicted_labels_prob = CART_classifier(cleandata,cleanlabels)
        rules,r_elap_time,predicted_labels,predicted_labels_prob = CART_classifier(X_train, X_test, y_train)
        # Obtain unique labels
        uniquelabels = np.unique(y_train)
    elif rulesind_alg == 'cn2':
        #rules,r_elap_time,predicted_labels,y_test,predicted_labels_prob = CN2_classifier(cleandata,cleanlabels)
        rules,r_elap_time,predicted_labels,predicted_labels_prob = CN2_classifier(X_train, X_test, y_train, y_test)
        # Obtain unique labels
        uniquelabels = np.unique(cleanlabels)
    else:
        print('Rules induction algorithm not found')
        return {}

    print('ANTES - predicted_labels_prob.shape',predicted_labels_prob.shape)


    print('cleanlabels',np.unique(cleanlabels).shape)
    print('y_train',np.unique(y_train).shape)
    print('y_test',np.unique(y_test).shape)
    print('predicted_labels_prob.shape',predicted_labels_prob.shape)

    # Calculate rule induction process metrics
    rulind_metrics = rule_induction_process_metric(y_test,predicted_labels,predicted_labels_prob,uniquelabels)

    # Calculate rule metrics
    #rulind_metrics = rules_metrics(clusters,rules,cleandata.shape[0],round(r_elap_time,metric_decimals))
   
    # Append time to the metrics dict
    rulind_metrics['time'] = round(r_elap_time,metric_decimals)

    # Append number of rules too
    rulind_metrics['n_rules'] = len(rules)

    # Append algorithm name
    rulind_metrics['name'] = rulesind_alg

    return rulind_metrics

if __name__ == '__main__':

    # Create DB connection
    db = mysql.createDbConn()
    print(mysql.getVersion(db)) 

    # Insert run row
    runid = mysql.insertRun(db)

    ## DELETE
    params=[1,1,1,1,1,1]
    ## DELETE 

    #dataset = genfromtxt('/home/gabriel/Desktop/segmentation.csv', delimiter=',')
    #dataset = genfromtxt('/home/gabriel/Desktop/page-blocks.data', delimiter=',')
    dataset = genfromtxt('/home/gabriel/Downloads/datasets/house16H.dat', delimiter=',',max_rows=5000)

    if dataset.shape[0] == 0:
        mysql.updateRun(db,runid)
        db.close()
        sys.exit()

    else:
        # Insert dataset row
        datasetid = mysql.insertDataset(db,runid,*params,0,0,'0')
        #datasetvalidationid = mysql.insertDatasetValidation(db,runid,datasetid,
        #                                                    analysis_results['features'],
        #                                                    analysis_results['samples'],
        #                                                    analysis_results['linpointsperc'],
        #                                                    analysis_results['repeatedperc'],
        #                                                    analysis_results['repeatedgrps'],
        #                                                    analysis_results['outliersperc'],
        #                                                    analysis_results['outliersbyperpenperc'])

        # Clustering algorithm list
        l_clustering_alg = [
                'kmeans_++',
                'kmeans_random',
                'kmeans_pca',
                'dbscan',
                'birch',
                'meanshift',
                ]
        all_metrics = {}
        all_samples_to_delete = {}
        all_labels = {}
        all_clusters = {}
        metrics_winners=[0,0,0,0,0]
        metrics_win_val=[0,0,0,0,0]
        for caidx,clustering_alg in enumerate(l_clustering_alg):
            print('')
            print('####################################')
            print(clustering_alg.upper())
            print('####################################')
            print('')
            clus_metrics,samples_to_delete,cleanlabels,clusters = clustering_and_metrics(dataset,clustering_alg)

            # Saving Clustering metrics into the database
            print(clus_metrics)
            print('')
            clusmetricsid = mysql.insertClusMetrics(db,datasetid,clus_metrics['name'],
                                                    clus_metrics['cluster_cnt'],
                                                    clus_metrics['sin_ele_clus'],
                                                    clus_metrics['ignored_samples'],
                                                    clus_metrics['time'],
                                                    clus_metrics['silhouette_score'],
                                                    clus_metrics['calinski_harabaz_score'],
                                                    clus_metrics['wb_index'],
                                                    clus_metrics['dunn_index'],
                                                    clus_metrics['davies_bouldin_score'])


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
                    metrics_winners[4] = clus_metrics['name']
                    metrics_win_val[4] = clus_metrics['davies_bouldin_score']
                else:
                    if clus_metrics['silhouette_score'] > metrics_win_val[0]*(1+0.05):
                        metrics_winners[0] = clus_metrics['name']
                        metrics_win_val[0] = clus_metrics['silhouette_score']
                    
                    if clus_metrics['calinski_harabaz_score'] > metrics_win_val[1]*(1+0.05):
                        metrics_winners[1] = clus_metrics['name']
                        metrics_win_val[1] = clus_metrics['calinski_harabaz_score']
                    
                    if clus_metrics['dunn_index'] > metrics_win_val[2]*(1+0.05):
                        metrics_winners[2] = clus_metrics['name']
                        metrics_win_val[2] = clus_metrics['dunn_index']
                    
                    if clus_metrics['wb_index'] < metrics_win_val[3]*(1-0.05):
                        metrics_winners[3] = clus_metrics['name']
                        metrics_win_val[3] = clus_metrics['wb_index']
                    
                    if clus_metrics['davies_bouldin_score'] < metrics_win_val[4]*(1-0.05):
                        metrics_winners[4] = clus_metrics['name']
                        metrics_win_val[4] = clus_metrics['davies_bouldin_score']
                    
            # Save metrics, labels and samples_to_remove, and data splitted in clusters
            all_metrics[clus_metrics['name']] = clus_metrics
            all_labels[clus_metrics['name']] = cleanlabels
            all_samples_to_delete[clus_metrics['name']] = samples_to_delete
            all_clusters[clus_metrics['name']] = clusters
            
            print(metrics_winners)
            print(metrics_win_val)

    ocurrences = Counter(metrics_winners)
    print(ocurrences)
    winners_cnt = max(ocurrences.values())
    winners_idx = [i for i, j in enumerate(ocurrences.values()) if j == winners_cnt]

    ocurrkeys = list(ocurrences.keys())
    winners = [algo for algo in ocurrkeys if ocurrkeys.index(algo) in winners_idx]

    # Determine which metrics each winner won
    for winalg in winners:
        metricsmask={}
        metricsmask['silhouette_score']=False
        metricsmask['calinski_harabaz']=False
        metricsmask['dunn']=False
        metricsmask['wb']=False
        metricsmask['davies_bouldin']=False
        for metricidx,algoname in enumerate(metrics_winners):
            if winalg == algoname:
                if metricidx == 0:
                    metricsmask['silhouette_score']=True
                elif metricidx == 1:
                    metricsmask['calinski_harabaz']=True
                elif metricidx == 2:
                     metricsmask['dunn']=True
                elif metricidx == 3:
                     metricsmask['wb']=True
                elif metricidx == 4:
                     metricsmask['davies_bouldin']=True
        print('Winner algo ',winalg,metricsmask)
        if len(winners_idx) == 1:
            iswinner='YES'
        else:
            iswinner='NO'
        mysql.insertDatasetClusFinalistsR1(db,datasetid,winalg,metricsmask,iswinner)

    if len(winners_idx) == 1:
        print('The winner is ',winners[0])
        

        # Update dataset row to include the winning algorithm
        mysql.updateDatasetClusAlg(db,datasetid,winners[0])

    else:
        print('We have a tie')
        flag_sel=0 #flag to detect single element clusters
        flag_is=0 #flag to detect ignored samples
        metrics_winners=[0,0,0]
        metrics_win_val=[0,0,0]
        for winner_idx in winners_idx:
            algname = ocurrkeys[winner_idx]
            print(ocurrkeys[winner_idx])
            if all_metrics[algname]['ignored_samples'] > 0:
                flag_is=1
            if all_metrics[algname]['sin_ele_clus'] > 0:
                flag_sel=1
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
        
                if all_metrics[algname]['ignored_samples'] < metrics_win_val[2]:
                    metrics_winners[2] = all_metrics[algname]['name']
                    metrics_win_val[2] = all_metrics[algname]['ignored_samples']
            print(metrics_winners)
            print(metrics_win_val)

        # If no iteration generated single element clusters, do not consider this metric 
        if flag_sel == 0:
            print('not considering single-element-cluster')
            del metrics_winners[1]
            del metrics_win_val[1]

        # If no iteration has ingnore samples, do not consider this metric
        if flag_is == 0:
            print('not considering ignored-samples')
            del metrics_winners[-1]
            del metrics_win_val[-1]

        ocurrences = Counter(metrics_winners)
        print(ocurrences)
        winners_cnt = max(ocurrences.values())
        winners_idx = [i for i, j in enumerate(ocurrences.values()) if j == winners_cnt]

        ocurrkeys = list(ocurrences.keys())
        winners = [algo for algo in ocurrkeys if ocurrkeys.index(algo) in winners_idx]

        # Retrieve on which secondary metrics the algorith won
        for winalg in winners:
            metricsmask={}
            metricsmask['time']=False
            metricsmask['sin_ele_clus']=False
            metricsmask['ignored_samples']=False
            for metricidx,algoname in enumerate(metrics_winners):
                if winalg == algoname:
                    if metricidx == 0:
                        metricsmask['time']=True
                    elif metricidx == 1:
                        metricsmask['sin_ele_clus']=True
                    elif metricidx == 2:
                         metricsmask['ignored_samples']=True
            print('Winner algo ',winalg,metricsmask)
            if len(winners_idx) == 1:
                iswinner='YES'
            else:
                iswinner='TIE'
            mysql.updateDatasetClusFinalistsR2(db,datasetid,winalg,metricsmask,iswinner)

        if len(winners_idx) == 1:
            print('The winner is ',winners[0])

            # Update dataset row to include the winning algorithm
            mysql.updateDatasetClusAlg(db,datasetid,winners[0])
        else:
            print('We have a tie')
            mysql.updateDatasetClusAlg(db,datasetid,'TIE')
            print(winners)
    # Induct rules for the winner clustering
    l_ruleind_alg = [
            'cn2',
            'cart'
            ]
   
    # Always use first element of "winners": if we have a tie, I'll just pick the first algorithm in the list
    clusalg=winners[0]
    ri_metrics_winners=[0]
    ri_metrics_win_val=[0]
    for riaidx,ruleind_alg in enumerate(l_ruleind_alg):
        rimetrics = rule_induction_and_metrics(dataset,ruleind_alg,all_samples_to_delete[clusalg],all_labels[clusalg],all_clusters[clusalg])

        # Save rule induction metrics in mysql
        mysql.insertRIMetrics(db,datasetid,clusmetricsid,rimetrics['name'],rimetrics['n_rules'],rimetrics['time'],rimetrics['auc'])

        print('ruleind_alg:',ruleind_alg,'auc:',rimetrics['auc'])
        if ri_metrics_winners[0] == 0:
            ri_metrics_winners[0] = rimetrics['name']
            ri_metrics_win_val[0] = rimetrics['auc']
            all_metrics[rimetrics['name']] = rimetrics
        else:

            # The auc value should be at least 5% higher or 5% lower than the existing winner or % lo
            wonby='metr'
            if rimetrics['auc'] > ri_metrics_win_val[0]*(1+.05):
                winner = rimetrics['name']
            elif rimetrics['auc'] < ri_metrics_win_val[0]*(1-.05):
                winner = ri_metrics_winners[0] 
            else:
                print('We have a tie')
                wonby='time'
                if rimetrics['time'] < all_metrics[ri_metrics_winners[0]]['time']:
                    winner = rimetrics['name']
                elif rimetrics['time'] > all_metrics[ri_metrics_winners[0]]['time']:
                    winner = ri_metrics_winners[0]
                else:
                    print('we have a tie in RI alg')
                    winner='tie'
                    wonby='tie'

            print('winner',winner)

            # Updating mysql with the RI algorithm
            mysql.updateDatasetRIAlg(db,datasetid,winner,wonby)

    #dstypeidx+=1
# Update run table with end date
mysql.updateRun(db,runid)
db.close()
