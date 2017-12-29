#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics
import numpy as np
from common import get_intra_cluster_distances
from common import split_data_in_clusters
from numpy.linalg import norm
from math import e
from math import log

def clustering_metrics(estimator, name, data, time, sample_size,clusters):


    def dunn_index(estimator,data):

        def get_inter_cluster_distances(i, j, clusters):
            distances = []
            for cluster_i_element in clusters[i]:
                for cluster_j_element in clusters[j]:
                    distances.append(norm(cluster_i_element-cluster_j_element))
            return distances

        # Split data into the different clusters
        #clusters = split_data_in_clusters(estimator,data)

        #clusters={}
        #it = np.nditer(estimator.labels_, flags=['f_index'])
        #while not it.finished:
        #    clusterid = int(it[0])
        #    if clusterid in clusters: 
        #        clusters[clusterid] = np.append(clusters[clusterid],[data[it.index,:]],axis=0)
        #    else:
        #        clusters[clusterid] = np.array([data[it.index,:]])
        #    it.iternext()
        

        # Calculates the maximum internal distance.
        l_micd = []
        for c in clusters:
            ## For each cluster, calculates the distances between the cluster points
            # Ignore single element clusters
            if clusters[c].shape[0] == 1:
                print("<<<< SINGLE ELEMENT CLUSTER WAS GENERATED >>>>")
            else:
                icd = get_intra_cluster_distances(clusters[c])
                micd = np.max(icd)
                l_micd.append(micd)
        
        ## Obtain the minimum distance across all clusters
        max_intra_cluster_dist = np.max(l_micd)
                    
   
        # Calculate the minimum inter cluster distance

        distances = []
        for i in range(len(clusters)-1):
            for j in range(i+1,len(clusters)):
                distances.append(get_inter_cluster_distances(i, j, clusters))
        min_inter_cluster_dist = np.min(np.min(distances))

        return min_inter_cluster_dist/max_intra_cluster_dist


    proc_metrics={}
    proc_metrics['name'] =  name
    proc_metrics['time'] = time
    #proc_metrics['inertia'] = estimator.inertia_
    proc_metrics['calinski_harabaz_score'] = metrics.calinski_harabaz_score(data, estimator.labels_)
    proc_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    proc_metrics['dunn_index'] = dunn_index(estimator,data)
    print(proc_metrics)

def rules_metrics(clusters,rules,n_samples):

    ## Create contingency table for each Rule,Cluster pair:
    # -Cluster 'c' examples covered by rule 'r'
    # -Cluster 'c' examples Not covered by rule 'r'
    # -Non Cluster 'c'  covered by rule 'r'
    # -Non Cluster 'c' examples Not covered by rule r
    
    d_cont_table={}

    for ruleid in rules:
        print("Rule: "+ruleid.__str__())
        print("************************")
        if ruleid not in d_cont_table:
            d_cont_table[ruleid] = {}

        # In this case clusterid is the position of the value in the list since the clusters are also numbered by position
        for clusterid,clustercnt in enumerate(rules[ruleid]['classes_matched'][0]):
            print("Cluster",clusterid)
            print(clustercnt)
            if clusterid not in d_cont_table[ruleid]:
                d_cont_table[ruleid][clusterid] = {}
            d_cont_table[ruleid][clusterid]['ncr'] = clustercnt
            d_cont_table[ruleid][clusterid]['n!cr'] = sum(rules[ruleid]['classes_matched'][0]) - clustercnt
            d_cont_table[ruleid][clusterid]['nc!r'] = len(clusters[clusterid]) - clustercnt
            d_cont_table[ruleid][clusterid]['n!c!r'] = (n_samples - sum(rules[ruleid]['classes_matched'][0])) - (len(clusters[clusterid])+clustercnt)

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
    # Qis = -log(nc / N) + log(nrc / nr)
    #
    # Information score calculation will fail for every Rule-Cluster combination for which the rule doesn't cover any cluster members (ncr = 0)
    # All Rule-Cluster combinations where ncr = 0  won't be considered in the average (avg_Qis)


    ## Measure of logical sufficiency (Duda, Gaschnig and Hart, 1979; Ali, Pazzani, 1993)
    # 
    # Qls = (ncr/nc) / (nrIc/nIc)

    ## Measure of discrimination (An and Cercone, 1998)
    #
    # Qmd = log((ncr/ncIr)/(nIcr/nIcIr))

    for rule in d_cont_table:
        print('Rule',rule)
        sum_Qws = 0
        sum_Qprod = 0
        #sum_X2 = 0
        sum_Qcohen = 0
        sum_Qcoleman = 0
        sum_Qis = 0
        sum_Qls = 0
        sum_Qmd = 0

        for cluster in d_cont_table[rule]:
            ncr = d_cont_table[rule][cluster]['ncr']
            nIcr = d_cont_table[rule][cluster]['n!cr']
            ncIr = d_cont_table[rule][cluster]['nc!r']
            cons = ncr / (ncr + nIcr)
            cover = ncr / (ncr + ncIr)
            w1 = 0.5 + (1/4 * cons)
            w2 = 0.5 - (1/4 * cons)

            # Qws
            Qws = w1 * cons + w2 * cover
            sum_Qws = sum_Qws + Qws

            # Qprod
            Qprod = cons * (e**(cover-1))
            sum_Qprod = sum_Qprod + Qprod

            # X2
            nIcIr = d_cont_table[rule][cluster]['n!c!r']
            nc = d_cont_table[rule][cluster]['nc!r'] + d_cont_table[rule][cluster]['ncr']
            nr = d_cont_table[rule][cluster]['n!cr'] + d_cont_table[rule][cluster]['ncr']
            nIc = d_cont_table[rule][cluster]['n!c!r'] + d_cont_table[rule][cluster]['n!cr']
            nIr = d_cont_table[rule][cluster]['n!c!r'] + d_cont_table[rule][cluster]['nc!r']
            #X2 = n_samples * ((ncr*nIcIr - nIcr*ncIr)**2) / (nc*nIc*nr*nIr)
            #sum_X2 = sum_X2 + X2

            # Qcohen
            fr = nr / n_samples 
            fc = nc / n_samples
            fIr = nIr / n_samples
            fIc = nIc / n_samples
            fcr = ncr / n_samples
            fIcIr = nIcIr / n_samples
            Qcohen = (fcr + fIcIr - (fr * fc + fIr * fIc)) / (1 - (fr * fc + fIr * fIc))
            sum_Qcohen = sum_Qcohen + Qcohen

            # Qcoleman
            Qcoleman = (fcr - fr * fc)/(fr - fr * fc)
            sum_Qcoleman = sum_Qcoleman + Qcoleman

            # Qis
            if ncr/nr == 0:
                pass
            else:
                Qis = -log(nc / n_samples) + log(ncr / nr)
                sum_Qis = sum_Qis + Qis

            # Qls
            if nc == 0 or nIc == 0 or nIcr/nIc == 0:
                pass
            else:
                Qls = (ncr/nc) / (nIcr/nIc)
                sum_Qls = sum_Qls + Qls

            # Qmd
            if ncIr == 0 or nIcIr == 0 or nIcr/nIcIr == 0 or (ncr/ncIr)/(nIcr/nIcIr) == 0:
                pass
            else:
                Qmd = log((ncr/ncIr)/(nIcr/nIcIr))
                sum_Qmd = sum_Qmd + Qmd

        avg_Qws = round(sum_Qws / len(clusters),2)
        avg_Qprod = round(sum_Qprod / len(clusters),2)
        #avg_X2 = round(sum_X2 / len(clusters),2)
        avg_Qcohen = round(sum_Qcohen / len(clusters),2)
        avg_Qcoleman = round(sum_Qcoleman / len(clusters),2)
        avg_Qis = round(sum_Qis / len(clusters),2)
        avg_Qls = round(sum_Qls / len(clusters),2)
        avg_Qmd = round(sum_Qmd / len(clusters),2)
        print('avg_Qws: ',avg_Qws)
        print('avg_Qprod: ',avg_Qprod)
        #print('avg_X2: ',avg_X2)
        print('avg_Qcohen: ',avg_Qcohen)
        print('avg_Qcoleman: ',avg_Qcoleman)
        print('avg_Qis ',avg_Qis)
        print('avg_Qls ',avg_Qls)
        print('avg_Qmd ',avg_Qmd)
