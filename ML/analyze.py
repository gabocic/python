#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
import numpy as np
from numpy.linalg import lstsq
from sklearn import datasets
from numpy.linalg import norm
from sympy.solvers import solve
from sympy import Symbol
from scipy.spatial import distance_matrix
from common import get_intra_cluster_distances
from plot_2d_3d import plot_2d_3d


class DatafileNotFound(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def analyze_dataset(data=None,debug=0,plot=0,load_from_file='dataset.svl'):

    def logger(var=None,message=None,dbg_level=2):
        if dbg_level <= debug:
            if var is None:
                var = ''
            if message is None:
                message = ''
            print(message,var)

    if data is None and load_from_file is not None:
        try:
            dataspmat,tags = datasets.load_svmlight_file(load_from_file)
            ## Converting from SciPy Sparse matrix to numpy ndarray
            data = dataspmat.toarray()
        except:
            raise DatafileNotFound("Failed to load dataset from "+load_from_file)

    logger(message="\n Loaded data: \n *******************",var=None,dbg_level=2)
    logger(message=None,var=data,dbg_level=2)
    n_features = data.shape[1]

    # Average for each set of coordinates
    datamean = data.mean(axis=0)

    # Singular Value Descomposition
    ## Any given matrix can be factorize as the product of three matrixes: A = U E V*
    ## The matrix V[n_features x n_features] is unitary and its first row is a vector which corresponds to the direction of the line that fit the data points
    U,E,V = np.linalg.svd(data - datamean)

   
    # What I'm trying to do here is to calculate the norm of the fitting line that intersects the "box" where the points are contained.
    # The challenge is to find the two parameters that would become the two ends of the line
   
    maxcords=np.amax(data,axis=0)
    mincords=np.amin(data,axis=0)

    logger(message="maxcords",var=maxcords,dbg_level=2)
    logger(message="mincords",var=mincords,dbg_level=2)
    logger(message="Direction vector",var=V[0],dbg_level=2)
    logger(message="Mean point",var=datamean,dbg_level=2)


    # ToDo: Checking if for some reason the max or min points are part of the line
    #test=(maxcords-datamean)/V[0]
    #logger(message="Lambdas for all components are equal? ->",var=test,dbg_level=0)
    
    #test=(mincords-datamean)/V[0]
    #logger(message="Lambdas for all components are equal? ->",var=test,dbg_level=0)

    # Calculating parameters as MaxCor_x / V_x,  MaxCor_y / V_y, etc
    
    l_lambdas=[]
    l_hlpoints=[]
    
    # For every V and maxcords components, find which plane the line crosses
    for idx in range(0,n_features):
        # Calculate lambda for this component to be equal the max component in the dataset
        v_lambda = (maxcords[idx]-datamean[idx]) / V[0][idx]
        r_lambda = datamean + v_lambda * V[0]
        logger(message="r_lambda",var=r_lambda,dbg_level=2)

        # Check if the for the above lambda, the remaining components are within the plane 
        p=0
        ok_count=0
        for r_lambda_comp in r_lambda:
            if p == idx:
                pass
            else:
                if r_lambda_comp >= mincords[p] and r_lambda_comp <= maxcords[p]:
                    ok_count +=1
                    
            p+=1
        if ok_count == n_features -1:
            l_lambdas.append(v_lambda)
            l_hlpoints.append(r_lambda)
            logger(message="Candidate lambda",var=v_lambda,dbg_level=2)
            logger(message="Candidate point",var=r_lambda,dbg_level=2)
            

    # Calculating parameters as MinCor_x / V_x,  MinCor_y / V_y, etc

    # For every V and mincords components, find which plane the line crosses
    for idx in range(0,n_features):
        # Calculate lambda for this component to be equal the min component in the dataset
        v_lambda = (mincords[idx]-datamean[idx]) / V[0][idx]
        r_lambda = datamean + v_lambda * V[0]
        logger(message="r_lambda",var=r_lambda,dbg_level=2)

        # Check if the for the above lambda, the remaining components are within the plane 
        p=0
        ok_count=0
        for r_lambda_comp in r_lambda:
            if p == idx:
                pass
            else:
                if r_lambda_comp >= mincords[p] and r_lambda_comp <= maxcords[p]:
                    ok_count +=1
                    
            p+=1
        if ok_count == n_features -1:
            l_lambdas.append(v_lambda)
            l_hlpoints.append(r_lambda)
            logger("Candidate lambda",v_lambda,dbg_level=2)
            logger("Candidate point",r_lambda,dbg_level=2)

    # Define radius for which points will be considered as "linear"
    dthres = (norm(l_hlpoints[0]-l_hlpoints[1])) * 0.05

    ## Parametric line: r-> = ro + kv->
    linepts = V[0] * np.mgrid[l_lambdas[0]:l_lambdas[1]:2j][:, np.newaxis]
    
    # adding the datamean point to the points generated previously to obtain the final fitting line
    linepts += datamean
    
    #Calculate the distance of each point to the line
    pdist=[]
    for row in data:
        A = datamean
        B = datamean + V[0]
        P = row
        pa = P - A
        ba = B - A
        t = np.dot(pa,ba)/np.dot(ba,ba)
        d = norm(pa - t*ba)
        pdist.append(d)
    pdist = np.asarray(pdist)


    # Separate "linear" points from "non linear"
    linp=[]
    nlinp=[]
    u = 0
    for d in pdist:
        if d < dthres:
            linp.append(data[u,:])
        else:
            nlinp.append(data[u,:])
        u+=1

    l_linp = len(linp)
    l_nlinp = len(nlinp)

    linpointsperc = round(100*(l_linp/(l_linp+l_nlinp)),2)
    
    logger(message="Percentage of linear points:",var=linpointsperc,dbg_level=0)
    #logger(message="Percentage of Non linear points:",var=round(100*(l_nlinp/(l_linp+l_nlinp)),2),dbg_level=0)
    #sys.exit()

    ## Calculate distance between points
    l_pp_dist = get_intra_cluster_distances(data)
    avgdist = np.mean(l_pp_dist)
    print("Average distance",avgdist,len(l_pp_dist))

    ## Density coeficient = Avg point-to-point distance / norm(max90th - min90th)
    ## Max and Min will only consider component percentile 90th 
    #np.percentile(data, 90, axis=0)

    boxdiag = norm(maxcords-mincords)
    print("Box diagonal",boxdiag)

    density_coef = boxdiag/avgdist
    print("Density coeficient",density_coef)


 
    ## Distance distribution analysis

    # Generate distance to the mean matrix
    dist2mean = distance_matrix(data,[datamean])

    # Sort the distances array to reduce time complexity
    dist2mean = np.sort(dist2mean,axis=0)
    
    # Get the 20% furthest points
    last20 = -(int(dist2mean.shape[0]*.2)) 
    l_20percfur = dist2mean[last20:]
    #print(l_20percfur)

    # Determine outlier threslhold as 1.5 x distance from the mean to the furthest point not included in the top 20%
    olthres = 1.5 * dist2mean[last20-1:last20]
    #olthres = int(l_20percfur[-8:-7])
    print('olthres',olthres)
    position = np.searchsorted(l_20percfur.T[0],olthres)
    print('position',position)
    numol = l_20percfur.shape[0] - position
    print('numol',numol)
    outliersperc = round(100*(numol[0,0]/data.shape[0]),2)


    #### Repeated analysis ##########

    uniq,arrcount = np.unique(data,axis=0,return_counts=True)
    groups = [ count for count in arrcount if count >= 0.05 * data.shape[0]] 
    #print('arrcount:',arrcount)
    
    n_groups = len(groups)
    n_repeated = sum(groups) - n_groups
    #print('groups:',n_groups)
    #print('repeated:',n_repeated)
    repeatedperc = round(100*(n_repeated/data.shape[0]),2)



    #l_bins=[0,distmax2mean[0][0]*0.3,distmax2mean[0][0]*0.6,distmax2mean[0][0]*0.9]
    #print(l_bins)

    # Check how many points are higher than the 90th percentile
    #print('Distance histogram')
    #print(np.histogram(dist2mean.T[0],bins=l_bins))

    # Prepare output
    outdict = {}
    outdict['features'] = n_features
    outdict['samples'] = data.shape[0]
    outdict['linpointsperc'] = linpointsperc
    outdict['outliersperc'] = outliersperc
    outdict['repeatedperc'] = repeatedperc
    outdict['repeatedgrps'] = n_groups

    if plot == 1:
        if n_features < 4:
            # Plot samples
            element_list=[]
            for point in linp:
                element={'type':'dot','value':point,'color':'y','marker':'o','size':10}
                element_list.append(element)
            for point in nlinp:
                element={'type':'dot','value':point,'color':'r','marker':'o','size':10}
                element_list.append(element)
            element={'type':'dot','value':datamean,'color':'g','marker':'o','size':10}
            element_list.append(element)
            element={'type':'line','value':linepts.T,'color':'b'}
            element_list.append(element)
            plot_2d_3d(element_list,n_features)

    return outdict
