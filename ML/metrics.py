#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics
import numpy as np
from common import split_data_in_clusters
from numpy.linalg import norm
from math import e
from math import log
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_distances

def clustering_metrics(estimator, name, data, time, sample_size,clusters,sin_ele_clus):


    def dunn_index(estimator,data):
    # The higher the value, the “better” the clustering will be

        #def get_inter_cluster_distances(i, j, clusters):
        #    distances = []
        #    for cluster_i_element in clusters[i]:
        #        for cluster_j_element in clusters[j]:
        #            distances.append(norm(cluster_i_element-cluster_j_element))
        #    return distances


        # Calculates the maximum internal distance.
        #single_ele_clus = 0
        l_micd = []
        #clus_to_remove = []
        for c in clusters:
            ## Ignore single element clusters
            #if clusters[c].shape[0] == 1:
            #    #<<<< SINGLE ELEMENT CLUSTER WAS GENERATED >>>>
            #    single_ele_clus+=1
            #    # Saving key to remove it after the loop is done (to avoid "dictionary changed size during iteration")
            #    clus_to_remove.append(c)
            #else:
            ## Condensed distance matrix
            icd = pdist(clusters[c])
            micd = np.max(icd)
            l_micd.append(micd)
       

        # Removing single-element cluster data
        #for sec in clus_to_remove:
        #    clusters.pop(sec,None)
        
        ## Obtain the minimum distance across all clusters
        max_intra_cluster_dist = np.max(l_micd)
                    
   
        # Calculate the minimum inter cluster distance
        distances = []
        for i in clusters.keys():
            for j in clusters.keys():
                if j > i:
                    # Pairwise distance between cluster i and j
                    print('Pairwise distance between cluster ',i,' and ',j)
                    pd = pairwise_distances(clusters[i],clusters[j],n_jobs=1) # n_jobs > 1 was slowing down the process when several small clusters were formed

                    # Save the minimum distance from the pd matrix
                    distances.append(np.min(pd))

        # Obtain the minimum of the minimum distances
        min_inter_cluster_dist = np.min(distances)

        return min_inter_cluster_dist/max_intra_cluster_dist


    clus_metrics={}
    clus_metrics['name'] =  name
    clus_metrics['time'] = time
    clus_metrics['calinski_harabaz_score'] = metrics.calinski_harabaz_score(data, estimator.labels_)
    clus_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    clus_metrics['dunn_index'] = dunn_index(estimator,data)
    clus_metrics['sin_ele_clus'] = sin_ele_clus

    return clus_metrics

def rules_metrics(clusters,rules,n_samples,elap_time):

    ## Create contingency table for each Rule,Cluster pair:
    # -Cluster 'c' examples covered by rule 'r'
    # -Cluster 'c' examples Not covered by rule 'r'
    # -Non Cluster 'c'  covered by rule 'r'
    # -Non Cluster 'c' examples Not covered by rule r
    
    d_cont_table={}

    for ruleid in rules:

        # In this case clusterid is the position of the value in the list since the clusters are also numbered by position
        for clusterid,clustercnt in enumerate(rules[ruleid]['classes_matched'][0]):

            # Filter out any rules not covering at least 30% of the cluster samples
            if clustercnt/len(clusters[clusterid]) > 0.3:
                if ruleid not in d_cont_table:
                    d_cont_table[ruleid] = {}
                if clusterid not in d_cont_table[ruleid]:
                    d_cont_table[ruleid][clusterid] = {}
                d_cont_table[ruleid][clusterid]['ncr'] = clustercnt
                d_cont_table[ruleid][clusterid]['n!cr'] = sum(rules[ruleid]['classes_matched'][0]) - clustercnt
                d_cont_table[ruleid][clusterid]['nc!r'] = len(clusters[clusterid]) - clustercnt
                d_cont_table[ruleid][clusterid]['n!c!r'] = (n_samples - sum(rules[ruleid]['classes_matched'][0])) - len(clusters[clusterid]) + clustercnt

    ## Weighted Sum of consistency and coverage (Michalsky, 1990)
    # Qws = w1 x cons(R) + w2 x cover(R), with
    #
    #   cons(R) = ncr / nr
    #   cover(R) = ncr / nc
    #   w1 = 0.5 + 1/4 x cons(R)
    #   w2 = 0.5 - 1/4 x cons(R)

    ## Product of consistency and coverage (Brazdil & Torgo, 1990)
    #
    # Qprod = cons(R) x e^(cover(R) - 1)

    ## Pearson X2 statistic [DISABLED] - as per the paper, a different formula was used for the experiments
    #
    # x2 = N(ncr x n!c!r - n!cr x nc!r)^2 / (nc x n!c x nr x n!r) 

    ## Cohen's formula (Cohen, 1960)
    #
    #   Qcohen = frc + f!r!c - (fr x fc + f!r x f!c) / (1 - (fr x fc + f!r x f!c))
    # 
    # It is directly interpretable as the proportion of joint judgments in which there is agreement, after chance
    # agreement is excluded. Its upper limit is +1.00, and its lower limit
    # falls between zero and -1.00, depending on the distribution of
    # judgments by the two judges.

    ## Coleman's Formula (Bishop, Fienbehg and Holland, 1991; Bruha and Kockova, 1993)
    #
    # Qcoleman = (fcr - fr * fc)/(fr - fr * fc)


    ## Information Score (Kononenko and Bratko, 1991)
    #
    # Qis = -log2(nc / N) + log2(nrc / nr)
    #
    # This score could oscilate between 0 and +oo. The higher the score, the higher the information provided, which is better
    # Information score calculation will fail for every Rule-Cluster combination for which the rule doesn't cover any cluster members (ncr = 0)
    # All Rule-Cluster combinations where ncr = 0  won't be considered


    ## Measure of logical sufficiency (Duda, Gaschnig and Hart, 1979; Ali, Pazzani, 1993)
    # 
    # Qls = (ncr/nc) / (nrIc/nIc)

    ## Measure of discrimination (An and Cercone, 1998)
    #
    # The higher the better
    # Qmd = log((ncr/ncIr)/(nIcr/nIcIr))

    rules_metrics=[]

    for rule in d_cont_table:
        e_rules_metrics={}
        e_rules_metrics['ruleid'] = rule

        for cluster in d_cont_table[rule]:
            e_rules_metrics['cluster'] = cluster

            ncr = d_cont_table[rule][cluster]['ncr']
            nIcr = d_cont_table[rule][cluster]['n!cr']
            ncIr = d_cont_table[rule][cluster]['nc!r']
            nIcIr = d_cont_table[rule][cluster]['n!c!r']
            nc = d_cont_table[rule][cluster]['nc!r'] + d_cont_table[rule][cluster]['ncr']
            nr = d_cont_table[rule][cluster]['n!cr'] + d_cont_table[rule][cluster]['ncr']
            nIc = d_cont_table[rule][cluster]['n!c!r'] + d_cont_table[rule][cluster]['n!cr']
            nIr = d_cont_table[rule][cluster]['n!c!r'] + d_cont_table[rule][cluster]['nc!r']
            cons = ncr / (ncr + nIcr)
            cover = ncr / (ncr + ncIr)
            w1 = 0.5 + (1/4 * cons)
            w2 = 0.5 - (1/4 * cons)

            # Qws
            Qws = round(w1 * cons + w2 * cover,3)
            e_rules_metrics['Qws'] = Qws

            # Qprod
            Qprod = round(cons * (e**(cover-1)),3)
            e_rules_metrics['Qprod'] = Qprod

            # Qcohen
            fr = nr / n_samples 
            fc = nc / n_samples
            fIr = nIr / n_samples
            fIc = nIc / n_samples
            fcr = ncr / n_samples
            fIcIr = nIcIr / n_samples
            Qcohen = round((fcr + fIcIr - (fr * fc + fIr * fIc)) / (1 - (fr * fc + fIr * fIc)),3)
            e_rules_metrics['Qcohen'] = Qcohen

            # Qcoleman
            Qcoleman = round((fcr - fr * fc)/(fr - fr * fc),3)
            e_rules_metrics['Qcoleman'] = Qcoleman

            # Qis
            if ncr/nr == 0:
                Qis = None
            else:
                Qis = round(-log((nc/n_samples),2) + log((ncr/nr),2),3)
            e_rules_metrics['Qis'] = Qis

            # Qls
            if nc == 0 or nIc == 0 or nIcr/nIc == 0:
                Qls = None 
            else:
                Qls = round((ncr/nc) / (nIcr/nIc),3)
            e_rules_metrics['Qls'] = Qls

            # Qmd
            if ncIr == 0 or nIcIr == 0 or nIcr/nIcIr == 0 or (ncr/ncIr)/(nIcr/nIcIr) == 0:
                Qmd = None
            else:
                Qmd = round(log((ncr/ncIr)/(nIcr/nIcIr)),3)
            e_rules_metrics['Qmd'] = Qmd

        rules_metrics.append(e_rules_metrics)          
    dict_rules_metrics={"time":elap_time,"rules_metrics":rules_metrics}
    return dict_rules_metrics
