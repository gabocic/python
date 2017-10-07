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
                        perc_lin=20, perc_repeated=10, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=25,
                        shuffle=True,feat_dist=0,debug=0,plot=0):

    def logger(message,dbg_level):
        if dbg_level <= debug:
            print(message)


    ### Main ####
    # Calculate the percentage of useful values we need to generate

    # << ToDo: check that percentage parameters do not exceed 80% and that they end in zero (10, 20, 30.. etc)
    perc_usef_samples = 100 - perc_lin - perc_repeated
    usef_samples = int(0.01 * perc_usef_samples * n_samples)
    lin_samples = int(0.01 * perc_lin * n_samples)
    rep_samples = int(0.01 * perc_repeated * n_samples)
    logger("Useful samples: "+usef_samples.__str__(),1)
    logger("Linear samples: "+lin_samples.__str__(),1)
    logger("Repeated samples: "+rep_samples.__str__(),1)

    # feat_dist =  Feature distribution
    ## 0: interleave standard normal and uniform values
    ## [ x, y]: provide amount of attributes for each type (x and y >= 0)

    if feat_dist == 0:
        unifor_feat = int(n_features/2)
        standa_feat = n_features - unifor_feat
    logger("uniform features: "+unifor_feat.__str__(),1)
    logger("standard features: "+standa_feat.__str__(),1)

    # Harcoded value range
    value_limit = 100
    #value_limit = 10

    # Random numbers generator
    generator = np.random

    # Initialize dataset 
    X = np.zeros((usef_samples, n_features))
    Xs = np.zeros((usef_samples, standa_feat))
    Xu = np.zeros((usef_samples, unifor_feat))

    # Generate standard columns
    for i in range(0,standa_feat):
        # Create a random number for mean and stdev
        mean = (generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(1)))[0]

        # Generate stdev as a percentage of mean (betwee 10% and 50%)
        stdev = (generator.random_integers(low=1, high=5, size=(1)))[0]*0.1*abs(mean)
        logger("mean: "+mean.__str__(),2)
        logger("stdev: "+stdev.__str__(),2)
        m = stdev * generator.randn(usef_samples,1) + mean
        Xs[:usef_samples, i:i+1] = m

    Xs = np.around(Xs,3)

    logger("Standard columns:",2)
    logger(Xs,2)

    # Generate uniform columns
    for i in range(0,unifor_feat):
        # Create a random number for mean and stdev
        m = generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(usef_samples,1))
        Xu[:usef_samples, i:i+1] = m

    logger("Uniform columns:",2)
    logger(Xu,2)

    # Append columns to X
    #X[:usef_samples,0:standa_feat] = Xs
    X[:usef_samples,0:unifor_feat] = Xu
    X[:usef_samples,unifor_feat:standa_feat+unifor_feat] = Xs

    logger("Standard and uniform columns combined",2)
    logger(X,2)

    ## Generate samples with linear relation to a ramdom sample
    #  **********************************************************
    # Choose two points, generate the parametric equation of the line that passes through those two points, and use the parameter to generate samples
    # Add some noise to make it more realistic

    if lin_samples > 0:
        ## Choose a ramdom sample
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
            print("Cannot find another point to generate linear samples")
            raise TooFewPoints('Not able to find two points to generate pseudo linear samples')
        logger("p1:",2)
        logger(p1,2)

        d0 = np.array(p1 - p0)
        lin_points = np.zeros((lin_samples,n_features))
        for a in range(0,lin_samples+1):
            # Making constants smaller to prevent too many outliers
            lins = p0+a*(0.1)*d0
            lin_points[a-1:a,:] = lins

        # Add some noise 
        boxnorm = norm(np.amax(lin_points,axis=0) - np.amin(lin_points,axis=0))
        lin_points += np.random.normal(size=lin_points.shape) * boxnorm * 0.01

        logger("Linear points:",2)
        logger(lin_points,2)

    # Dummy samples generation
    repeated = np.zeros((rep_samples,n_features))        


    Xf = X
    # Stack useful,linear and repeated samples
    if lin_samples > 0:
        Xf = np.vstack((X,lin_points))
    if rep_samples > 0:
        Xf = np.vstack((Xf,repeated))

    ## Shrink the dataset by shrink factor
    # Average for each set of coordinates
    datamean = Xf.mean(axis=0)
    
    # Scale values
    u = 0
    for point in Xf:
        #print(point)
        dv = point - datamean



    logger("\n Final Dataset:\n *****************",2)
    logger(Xf,2)

    if plot == 1:
        if n_features < 4: 
            # Plot samples
            plot_2d_3d(p0,p1,Xf)
            plot_2d_3d(p0*10,p1*10,Xf*10)


    # Randomly permute features
    #indices = np.arange(n_features)
    #generator.shuffle(indices)
    #X[:, :] = X[:, indices]

    # Save to file
    datasets.dump_svmlight_file(Xf,np.zeros(n_samples),'dataset.svl')


create_dataset(n_samples=100, n_features=6,
                        perc_lin=0, perc_repeated=0, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=0,
                        shuffle=True,feat_dist=0,debug=2,plot=0)

