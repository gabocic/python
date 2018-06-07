#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from sklearn import datasets
from numpy.linalg import norm
from scipy.spatial import distance_matrix
from plot_2d_3d import plot_2d_3d
from parameters import *


class TooFewPoints(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def create_dataset(n_samples=20, n_features=3,
                        perc_lin=20, perc_repeated=10, n_groups=6,perc_outliers=10,
                        debug=1,plot=0,save_to_file=0):

    def logger(message,dbg_level):
        if dbg_level <= debug:
            print(message)


    ### Main ####

    ## A few safety checks
    
    # Sample types percentage should not exceed 100%
    if (perc_lin + perc_repeated + perc_outliers) > max_special_points_perc:
        logger('The sum of special points percentages cannot exceed '+max_special_points_perc.__str__(),0)
        return np.array([[]])
    
    # Check that the amount of samples is above the minimum
    if n_samples < min_dataset_size:
        logger('The minimum number of samples is '+min_dataset_size.__str__(),0)
        return np.array([[]])

    # Check that outliers percentage does not exceeds 20%
    if perc_outliers > 20:
        logger('The minimum number of samples is '+min_dataset_size.__str__(),0)


    # Calculate the number of samples for each type
    lin_samples = int(0.01 * perc_lin * n_samples)
    rep_samples = int(0.01 * perc_repeated * n_samples)
    out_samples = int(0.01 * perc_outliers * n_samples)
    usef_samples = n_samples - out_samples - rep_samples - lin_samples
    logger("Random samples: "+usef_samples.__str__(),1)
    logger("Linear samples: "+lin_samples.__str__(),1)
    logger("Repeated samples: "+rep_samples.__str__()+' - Groups: '+n_groups.__str__(),1)
    logger("Outliers: "+out_samples.__str__(),1)
    logger("Total samples: "+n_samples.__str__(),1)

    # Features distributions
    ## Interleave standard normal and uniform values
    unifor_feat = int(n_features/2)
    standa_feat = n_features - unifor_feat
    logger("uniform features: "+unifor_feat.__str__(),1)
    logger("standard features: "+standa_feat.__str__(),1)

    # Initialize dataset. Including out_samples as they will be moved away from the mean later
    X = np.zeros((usef_samples+out_samples, n_features))
    Xs = np.zeros((usef_samples+out_samples, standa_feat))
    Xu = np.zeros((usef_samples+out_samples, unifor_feat))

    # Generate standard attributes. Including out_samples as they will be moved away from the mean later
    for i in range(0,standa_feat):
        generator = np.random
        # Create a random number for mean and stdev
        mean = (generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(1)))[0]

        # Generate stdev as a percentage of mean (betwee 10% and 50%)
        generator = np.random
        stdev = (generator.random_integers(low=1, high=5, size=(1)))[0]*0.1*abs(mean)
        logger("mean: "+mean.__str__(),2)
        logger("stdev: "+stdev.__str__(),2)
        generator = np.random
        m = stdev * generator.randn(usef_samples+out_samples,1) + mean
        Xs[:usef_samples+out_samples, i:i+1] = m

    # Round Xs values to 3 decimals
    Xs = np.around(Xs,feat_values_decimals)

    logger("Standard attributes:",2)
    logger(Xs,2)

    # Generate uniform attributes. Including out_samples as they will be moved away from the mean later
    for i in range(0,unifor_feat):
        # Create a random number for mean and stdev
        generator = np.random
        m = generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(usef_samples+out_samples,1))
        
        Xu[:usef_samples+out_samples, i:i+1] = m

    logger("Uniform attributes:",2)
    logger(Xu,2)

    # Append attributes to X - interleave standard and uniform attributes
    r = 0
    for p in range(0,unifor_feat):
        X[:usef_samples+out_samples,r:r+1] = Xs[:,p:p+1]
        X[:usef_samples+out_samples,r+1:r+2] = Xu[:,p:p+1]
        r += 2
    if standa_feat > unifor_feat:
        X[:usef_samples+out_samples,r:r+1] = Xs[:,p+1:p+2]

    logger("Standard and uniform attributes combined",2)
    logger(X,2)

    ## Generate samples with linear relation to a random pair of samples
    #  ******************************************************************
    # Choose two points, generate the parametric equation of the line that crosses those two points, and use the parameter to generate samples
    # Add some noise to make it more realistic

    if lin_samples > 0:

        # Calculate the mean for the values already generated
        Xmean = X.mean(axis=0)

        #Single Value Decomposition
        U,E,V = np.linalg.svd(X - Xmean)

        ## Choose a ramdom sample
        generator = np.random
        sampleidx = (generator.random_integers(low=0, high=usef_samples-1, size=(1)))[0]
        p0 = Xmean
        
        lin_points = np.zeros((lin_samples,n_features))
       
        ## Make half of the points to go in p1-p0 direction and the other half in the opposite (p0-p1)
        lin_samples_d0 = int(lin_samples/2)
        lin_samples_d1 = int(lin_samples - lin_samples_d0)

        d0 = V[0]
        for a in range(0,lin_samples_d0+1):
            # Adjusting point "lambda" to prevent too many outliers
            lins = p0+linear_points_lambda_adj_factor*a*d0
            lin_points[a-1:a,:] = lins
        maxlins = lins 

        d1 = -1*V[0]
        for b in range(0,lin_samples_d1+1):
            # Adjusting point "lambda" to prevent too many outliers
            lins = p0+linear_points_lambda_adj_factor*b*d1
            a+=1
            lin_points[a-1:a,:] = lins
        minlins = lins 

        # Add some noise 
        boxnorm = norm(maxlins - minlins)
        lin_points += np.random.normal(size=lin_points.shape) * boxnorm * linear_noise_factor

        logger("Linear points:",2)
        logger(lin_points,2)

    # Repeated samples generation
    if rep_samples > 0:
        if (rep_samples / n_groups) < (repeated_min_samples_per_group_perc*n_samples):
            logger('The number of samples per group should not be lower than 5% of total samples. Also, there cannot be more groups than completely random samples',0) 
            return np.array([[]])

        repeated = np.zeros((rep_samples,n_features))
        
        # Calculate samples per group
        samp_per_group = int(rep_samples / n_groups)
        
        # For each group, use i-element of the random values as seed
        for i in range(0,n_groups):
            repeated[i*samp_per_group:(i+1)*samp_per_group] = X[i]
        
        # Use the last sample to cover any repeated left
        repeated[(i+1)*samp_per_group:] = X[i]



    Xf = X
    #print('Random')
    #print('*********************************************************')
    #print(Xf)
    #print('*********************************************************')
    #print('Linear')
    #print('*********************************************************')
    #print(np.around(lin_points,3))
    #print('*********************************************************')
    #print('Repeated')
    #print('*********************************************************')
    #print(np.around(repeated,3))
    #print('*********************************************************')
    # Stack useful,linear and repeated samples
    if lin_samples > 0:
        Xf = np.vstack((X,np.around(lin_points,feat_values_decimals)))
    if rep_samples > 0:
        Xf = np.vstack((Xf,np.around(repeated,feat_values_decimals)))


    logger("\n Final Dataset:\n *****************",2)
    logger(Xf,2)
    #print('****************************Xf*************************')
    #print(Xf) 
    # Outliers generation

    if out_samples > 0:
        ## Dataset mean
        datamean = Xf.mean(axis=0)

        ## Obtain distance-to-mean matrix
        dist2mean = distance_matrix(Xf,[datamean])

        # Obtain the sorted array of dist2mean indexes
        sortd2midx = np.argsort(dist2mean,axis=0)

        #dist2mean = np.sort(dist2mean,axis=0)

        # Get the 20% furthest points
        last20 = int(sortd2midx.shape[0]*.2)
        l_20percfur = np.take(dist2mean,sortd2midx[-last20:])

        # Outlier threshold
        olthres = analysis_outlier_factor * np.take(dist2mean,sortd2midx[-last20-1:-last20])

        # Look for the first sample further than 'threshold' (ie. first outlier, if exists)
        firstolpos = np.searchsorted(l_20percfur.T[0],olthres)[0,0]
        
        # Determine number of outlier 
        numol = l_20percfur.shape[0] - firstolpos
        points_to_fix = numol - out_samples

        # If there are points to fix..
        if points_to_fix != 0:
            # Absolute position of the first outlier: Total samples - further20% + first oultlier position but within l_20percfur -1 (To obtain the index)
            abs_firstolpos = sortd2midx.shape[0] - last20 + firstolpos -1
            
            ## Distance factor: distance for first outlier divided by threshold distance
            #v_lambda = 1.5 * olthres / dist2mean[abs_firstolpos]
            v_lambda = olthres / dist2mean[abs_firstolpos]

            if points_to_fix > 0:
                # Get the necessary points closer to the mean
                print('Exceso de outliers')

                for dot in sortd2midx[abs_firstolpos:]:
                   Xf[dot,:] = datamean + (1/v_lambda)*(Xf[dot,:]-datamean)

            elif points_to_fix < 0:
                # Get the necessary points further from the mean
                print('Falta de outliers')

                #for dot in sortd2midx[sortd2midx.shape[0]-1-last20:abs_firstolpos]:
                for dot in sortd2midx[abs_firstolpos+points_to_fix:abs_firstolpos]:
                   Xf[dot,:] = datamean + v_lambda*(Xf[dot,:]-datamean)

        #print('Outliers')
        #print('***************')
        #print(np.take(Xf,(sortd2midx[abs_firstolpos+points_to_fix:abs_firstolpos].T)[0],axis=0))

    if plot == 1:
        if n_features < 4: 
            # Plot samples
            element_list=[]
            element={'type':'blob','value':X,'color':'r','marker':'o'}
            element_list.append(element)
            if lin_samples > 0:
                element={'type':'blob','value':lin_points,'color':'b','marker':'o'}
                element_list.append(element)
                element={'type':'dot','value':p0,'color':'g','marker':'x','size':190}
                element_list.append(element)
                #element={'type':'dot','value':p1,'color':'g','marker':'x','size':90}
                #element_list.append(element)
            if out_samples > 0:
                element={'type':'blob','value':np.take(Xf,sortd2midx[sortd2midx.shape[0]-1-last20:abs_firstolpos].T[0],axis=0),'color':'c','marker':'o'}
                element_list.append(element)
            plot_2d_3d(element_list,n_features)



    # Save to file
    if save_to_file == 1:
        datasets.dump_svmlight_file(Xf,np.zeros(n_samples),'dataset.svl')

    return Xf,unifor_feat,standa_feat
