#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np
from numpy import ones,vstack
from numpy.linalg import lstsq
from sklearn import datasets
from numpy.linalg import norm

from plot_2d_3d import plot_2d_3d

class TooFewPoints(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def create_dataset(n_samples=20, n_features=3,
                        perc_lin=20, perc_repeated=10, n_groups=2,perc_outliers=10,
                        debug=1,plot=0,save_to_file=0):

    def logger(message,dbg_level):
        if dbg_level <= debug:
            print(message)


    ### Main ####

    max_special_points_perc = 90
    min_dataset_size = 100

    if (perc_lin + perc_repeated + perc_outliers) > max_special_points_perc:
        logger('The sum of special points percentages cannot exceed '+max_special_points_perc.__str__(),0)
        return np.array([[]])

    if n_samples < min_dataset_size:
        logger('The minimum number of samples is '+min_dataset_size.__str__(),0)
        return np.array([[]])

    # Calculate the number of samples for each type
    lin_samples = int(0.01 * perc_lin * n_samples)
    rep_samples = int(0.01 * perc_repeated * n_samples)
    out_samples = int(0.01 * perc_outliers * n_samples)
    usef_samples = n_samples - out_samples - rep_samples - lin_samples
    logger("Random samples: "+usef_samples.__str__(),1)
    logger("Linear samples: "+lin_samples.__str__(),1)
    logger("Repeated samples: "+rep_samples.__str__(),1)
    logger("Outliers: "+out_samples.__str__(),1)

    # Features distributions
    ## Interleave standard normal and uniform values
    unifor_feat = int(n_features/2)
    standa_feat = n_features - unifor_feat
    logger("uniform features: "+unifor_feat.__str__(),1)
    logger("standard features: "+standa_feat.__str__(),1)

    # Harcoded value range: all sample values will be between value_limit and -value_limit
    value_limit = 100000

    # Initialize dataset 
    X = np.zeros((usef_samples, n_features))
    Xs = np.zeros((usef_samples, standa_feat))
    Xu = np.zeros((usef_samples, unifor_feat))

    # Generate standard attributes
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
        m = stdev * generator.randn(usef_samples,1) + mean
        Xs[:usef_samples, i:i+1] = m

    # Round Xs values to 3 decimals
    Xs = np.around(Xs,3)

    logger("Standard attributes:",2)
    logger(Xs,2)

    # Generate uniform attributes
    for i in range(0,unifor_feat):
        # Create a random number for mean and stdev
        generator = np.random
        m = generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(usef_samples,1))
        Xu[:usef_samples, i:i+1] = m

    logger("Uniform attributes:",2)
    logger(Xu,2)

    # Append attributes to X - interleave standard and uniform attributes
    r = 0
    for p in range(0,unifor_feat):
        X[:usef_samples,r:r+1] = Xs[:,p:p+1]
        X[:usef_samples,r+1:r+2] = Xu[:,p:p+1]
        r += 2
    if standa_feat > unifor_feat:
        X[:usef_samples,r:r+1] = Xs[:,p+1:p+2]

    logger("Standard and uniform attributes combined",2)
    logger(X,2)

    ## Generate samples with linear relation to a random pair of samples
    #  ******************************************************************
    # Choose two points, generate the parametric equation of the line that crosses those two points, and use the parameter to generate samples
    # Add some noise to make it more realistic

    if lin_samples > 0:
        ## Choose a ramdom sample
        generator = np.random
        sampleidx = (generator.random_integers(low=0, high=usef_samples-1, size=(1)))[0]
        logger("Winning samples:",2)
        p0 = X[sampleidx]
        logger("p0:",2)
        logger(p0,2)

        ## Choose another point based on p0 
        if sampleidx+1 <= X.shape[0]-1:
            p1 = X[sampleidx+1]
        elif sampleidx-1 >= 0:
            p1 = X[sampleidx-1]
        else:
            raise TooFewPoints('Not able to find two points to generate pseudo linear samples')
        logger("p1:",2)
        logger(p1,2)
        
        lin_points = np.zeros((lin_samples,n_features))
       
        ## Make half of the points to go in p1-p0 direction and the other half in the opposite (p0-p1)
        lin_samples_d0 = int(lin_samples/2)
        lin_samples_d1 = int(lin_samples - lin_samples_d0)

        d0 = np.array(p1 - p0)
        for a in range(0,lin_samples_d0+1):
            # Making constants smaller to prevent too many outliers
            lins = p0+a*(0.1)*d0
            lin_points[a-1:a,:] = lins

        d1 = np.array(p0 - p1)
        for b in range(0,lin_samples_d1+1):
            # Making constants smaller to prevent too many outliers
            lins = p0+b*(0.1)*d1
            a+=1
            lin_points[a-1:a,:] = lins

        # Add some noise 
        boxnorm = norm(np.amax(lin_points,axis=0) - np.amin(lin_points,axis=0))
        lin_points += np.random.normal(size=lin_points.shape) * boxnorm * 0.01

        logger("Linear points:",2)
        logger(lin_points,2)

    # Repeated samples generation
    repeated = np.zeros((rep_samples,n_features))
    ## << CONTINUTE HERE!!!
    # 1) Choose samples from the Random samples based on n_groups
    # 2) Generate the repeated submatrix based on the above samples
    # 3) Stack

    # Outliers generation
    # 1) Choose the N furthest points
    # 2) Move that point away from 'mean' in point-mean direction by 'dist'
    # 3) Stack


    Xf = X
    # Stack useful,linear and repeated samples
    if lin_samples > 0:
        Xf = np.vstack((X,np.around(lin_points,3)))
    if rep_samples > 0:
        Xf = np.vstack((Xf,np.around(repeated,3)))


    logger("\n Final Dataset:\n *****************",2)
    logger(Xf,2)

    if plot == 1:
        if n_features < 4: 
            # Plot samples
            element_list=[]
            element={'type':'blob','value':X,'color':'r','marker':'o'}
            element_list.append(element)
            element={'type':'blob','value':lin_points,'color':'b','marker':'o'}
            element_list.append(element)
            element={'type':'dot','value':p0,'color':'g','marker':'x','size':90}
            element_list.append(element)
            element={'type':'dot','value':p1,'color':'g','marker':'x','size':90}
            element_list.append(element)
            plot_2d_3d(element_list,n_features)



    # Save to file
    if save_to_file == 1:
        datasets.dump_svmlight_file(Xf,np.zeros(n_samples),'dataset.svl')

    return Xf
