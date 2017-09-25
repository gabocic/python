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
                        shuffle=True,feat_dist=0):

    # Calculate the percentage of useful values we need to generate

    # << ToDo: check that percentage parameters do not exceed 80% and that they end in zero (10, 20, 30.. etc)
    perc_usef_samples = 100 - perc_lin - perc_repeated
    usef_samples = int(0.01 * perc_usef_samples * n_samples)
    lin_samples = int(0.01 * perc_lin * n_samples)
    rep_samples = int(0.01 * perc_repeated * n_samples)
    print("Useful samples: "+usef_samples.__str__())
    print("Linear samples: "+lin_samples.__str__())
    print("Repeated samples: "+rep_samples.__str__())

    # feat_dist =  Feature distribution
    ## 0: interleave standard normal and uniform values
    ## [ x, y]: provide amount of attributes for each type (x and y >= 0)

    if feat_dist == 0:
        unifor_feat = int(n_features/2)
        standa_feat = n_features - unifor_feat
    print("uniform features: "+unifor_feat.__str__())
    print("standard features: "+standa_feat.__str__())

    # Harcoded value range
    value_limit = 10000

    # Random numbers generator
    #generator = np.random.RandomState(seed)
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
        print("mean: "+mean.__str__())
        print("stdev: "+stdev.__str__())
        m = stdev * generator.randn(usef_samples,1) + mean
        Xs[:usef_samples, i:i+1] = m

    Xs = np.around(Xs,3)
    #print(Xs)

    # Generate uniform columns
    for i in range(0,unifor_feat):
        # Create a random number for mean and stdev
        m = generator.random_integers(low=(-1)*value_limit, high=value_limit, size=(usef_samples,1))
        Xu[:usef_samples, i:i+1] = m

    print(Xu)

    # Append columns to X
    X[:usef_samples,0:standa_feat] = Xs
    X[:usef_samples,standa_feat:standa_feat+unifor_feat] = Xu
    print(X)

    ## Generate samples with linear relation to a ramdom sample
    #  **********************************************************
    # Choose two points, generate the parametric equation of the line that passes through those two points, and use the parameter to generate samples
    # Add some noise to make it more realistic

    ## Choose a ramdom sample
    sampleidx = (generator.random_integers(low=0, high=usef_samples-1, size=(1)))[0]
    print("Index: "+sampleidx.__str__())
    print("Winning samples:")
    p0 = X[sampleidx]
    print(p0)

    ## Choose another point based on p0 
    if sampleidx+1 <= X.shape[0]-1:
        p1 = X[sampleidx+1]
    elif sampleidx-1 >= 0:
        p1 = X[sampleidx-1]
    else:
        print("Cannot find another point to generate linear samples")
        raise TooFewPoints('Not able to find two points to generate pseudo linear samples')
    print(p1)

    print("Linear samples: "+lin_samples.__str__())
   

    d0 = np.array(p1 - p0)
    points = np.zeros((lin_samples,n_features))
    for a in range(0,lin_samples+1):
        # Making constants smaller to prevent too many outliers
        lins = p0+a*(0.1)*d0
        points[a-1:a,:] = lins

    # Add some noise 
    boxnorm = norm(np.amax(points,axis=0) - np.amin(points,axis=0))
    points += np.random.normal(size=points.shape) * boxnorm * 0.01

    if n_features < 4: 
        # Plot samples
        plot_2d_3d(points,p0,p1,X)


    # Dummy samples generation
    repeated = np.zeros((rep_samples,n_features))        

    # Stack useful and linear samples
    Xf = np.vstack((X,points))
    Xf = np.vstack((Xf,repeated))

    # Randomly permute features
    indices = np.arange(n_features)
    generator.shuffle(indices)
    X[:, :] = X[:, indices]

    # Save to file
    datasets.dump_svmlight_file(Xf,np.zeros(n_samples),'dataset.svl')


create_dataset(n_samples=100, n_features=7,
                        perc_lin=80, perc_repeated=0, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=10,
                        shuffle=True,feat_dist=0)

