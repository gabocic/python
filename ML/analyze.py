#!/home/gabriel/pythonenvs/v3.5/bin/python

import sys
import numpy as np
from sklearn import datasets
from numpy.linalg import norm
from scipy.spatial import distance_matrix
from plot_2d_3d import plot_2d_3d
from parameters import *


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

    ### Main ##
    
    # Attempting to load data from a file if None was passed
    if data is None and load_from_file is not None:
        try:
            dataspmat,tags = datasets.load_svmlight_file(load_from_file)
            ## Converting from SciPy Sparse matrix to numpy ndarray
            data = dataspmat.toarray()
        except:
            raise DatafileNotFound("Failed to load dataset from "+load_from_file)
        else:
            logger(message="\n Loaded data: \n *******************",var=None,dbg_level=2)
            logger(message=None,var=data,dbg_level=2)
    
    # Features number
    n_features = data.shape[1]

    # Average for each set of coordinates
    datamean = data.mean(axis=0)

    # Sample number
    samplenum = data.shape[0]


    ########################
    ## Outliers detection ##
    ########################

    ## Obtain distance-to-mean matrix
    dist2mean = distance_matrix(data,[datamean])

    # Obtain the sorted array of dist2mean indexes
    sortd2midx = np.argsort(dist2mean,axis=0)

    # Get the 20% furthest points
    last20 = int(sortd2midx.shape[0]*.2)
    l_20percfur = np.take(dist2mean,sortd2midx[-last20:])

    # Outlier threshold
    olthres = analysis_outlier_factor * np.take(dist2mean,sortd2midx[-last20-1:-last20])

    # Look for the first sample further than 'threshold' (ie. first outlier, if exists)
    firstolpos = np.searchsorted(l_20percfur.T[0],olthres)[0,0]

    # Determine number of outlier 
    numol = l_20percfur.shape[0] - firstolpos
    outliersperc = round(100*(numol/samplenum),2)
    print(numol,outliersperc)

    # Get the indexes for the outliers so we can skip them from the analysis
    if numol > 0:
        olidxs = sortd2midx[-last20:][-numol:]
        # Get outliers
        olsamples = np.take(data,olidxs,axis=0)

        # Remove outliers from dataset
        data=np.delete(data,olidxs,0)





    ########################
    ## Repeated analysis  ##
    ########################

    uniq,dataidx,arrcount = np.unique(data,axis=0,return_counts=True,return_index=True)

    # Only consider a group when it has more than n% of samples
    groups = [ count for count in arrcount if count >= analysis_group_min_members_perc * samplenum] 

    # Remove repeated samples
    data = uniq

    n_groups = len(groups)
    n_repeated = sum(groups) - n_groups
    repeatedperc = round(100*(n_repeated/samplenum),2)

    print('n_groups',n_groups)
    print('repeatedperc',repeatedperc)

    
    ##############################
    ## Linear points detection  ##
    ##############################

    # Average for each set of coordinates
    datamean = data.mean(axis=0)

    # Singular Value Descomposition
    ## Any given matrix can be factorize as the product of three matrixes: A = U E V*
    ## The matrix V[n_features x n_features] is unitary and its first row is a vector which corresponds to the direction of the line that fit the data points
    U,E,V = np.linalg.svd(data - datamean)

   
    # What I'm doing here is calculating the norm of the fitting line where it intersects the "box" (or "hyperbox")  in which the data points are contained.
    
    # Remove outliers from dataset before getting the max/min coordenates
    #if numol > 0:
    #    data_no_ol = np.take(data,(sortd2midx[:-numol].T)[0],axis=0)
    #else:
    #    data_no_ol = data

    maxcords=np.amax(data,axis=0)
    mincords=np.amin(data,axis=0)

    logger(message="maxcords",var=maxcords,dbg_level=2)
    logger(message="mincords",var=mincords,dbg_level=2)
    logger(message="Direction vector",var=V[0],dbg_level=2)
    logger(message="Mean point",var=datamean,dbg_level=2)


    # Checking if for some reason the max or min points are part of the line
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
    dthres = (norm(l_hlpoints[0]-l_hlpoints[1])) * analysis_fit_line_fraction

    ## Parametric line: r-> = ro + kv->
    linepts = V[0] * np.mgrid[l_lambdas[0]:l_lambdas[1]:2j][:, np.newaxis]
    
    # adding the datamean point to the points generated previously to obtain the final fitting line
    linepts += datamean
    
    #Calculate the distance of each point to the line. Then separate "linear" points from "non linear"
    linp=[]
    nlinp=[]
    nlinpd=[]
    for row in data:
        
        # Point-Line Distance 
        A = datamean
        B = datamean + V[0]
        P = row
        pa = P - A
        ba = B - A
        t = np.dot(pa,ba)/np.dot(ba,ba)
        d = norm(pa - t*ba)

        # Separate linear from non-linear samples
        if d < dthres:
            linp.append(row)
        else:
            nlinp.append(row)
            nlinpd.append(d)

    n_linp = len(linp)
    n_nlinp = len(nlinp)

    linpointsperc = round(100*(n_linp/samplenum),2)

    # If the dataset is fairly linear, check for outliers by perpendicularity
    if linpointsperc > 60:
        perpolthres = (norm(l_hlpoints[0]-l_hlpoints[1])) * analysis_fit_line_fraction_outliers
        perpol = [ dist2line for dist2line in nlinpd if dist2line >= perpolthres ] 
        outliersbyperpenperc = round(100*(len(perpol)/samplenum),2)
    else:
        outliersbyperpenperc = 0

    # Prepare output
    outdict = {}
    outdict['features'] = n_features
    outdict['samples'] = samplenum
    outdict['linpointsperc'] = linpointsperc
    outdict['outliersperc'] = float(outliersperc) # forcing conversion due to database insert error
    outdict['repeatedperc'] = float(repeatedperc) # forcing conversion due to database insert error
    outdict['repeatedgrps'] = n_groups
    outdict['outliersbyperpenperc'] = outliersbyperpenperc


    # Plot if requested
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
            element={'type':'dot','value':l_hlpoints[0],'color':'c','marker':'x','size':20}
            element_list.append(element)
            element={'type':'dot','value':l_hlpoints[1],'color':'c','marker':'x','size':20}
            element_list.append(element)
            if numol > 0:
                element={'type':'blob','value':olsamples[:,0],'color':'k','marker':'^','size':19}
                element_list.append(element)
            plot_2d_3d(element_list,n_features)

    return outdict
