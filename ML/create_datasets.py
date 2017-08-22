#!/home/gabriel/pythonenvs/v3.5/bin/python

import numpy as np

def create_dataset(n_samples=10, n_features=3,
                        perc_lin=20, perc_repeated=20, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=25,
                        shuffle=True,feat_dist=0):

    # Calculate the percentage of useful values we need to generate

    # << ToDo: check that percentage parameters do not exceed 80% and that they end in zero (10, 20, 30.. etc)
    perc_usef_samples = 100 - perc_lin - perc_repeated
    usef_samples = int(0.01 * perc_usef_samples * n_samples)
    lin_samples = int(0.01 * perc_lin * n_samples)
    print("Useful samples: "+usef_samples.__str__())

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
    X = np.zeros((n_samples, n_features))
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

    # Generate samples with linear relation to a ramdom sample
    ## Choose a ramdom sample
    sampleidx = (generator.random_integers(low=0, high=usef_samples, size=(1)))[0]
    print("Index: "+sampleidx.__str__())
    print("Winning samples:")
    p0 = X[sampleidx]
    print(p0)
    if X[sampleidx+1][0]:
        p1 = X[sampleidx+1]
    elif X[sampleidx-1][0]:
        p1 = X[sampleidx-1]
    else:
        print("Cannot find another point to generate linear samples")
        raise 
    print(p1)
    print("Linear samples: "+lin_samples.__str__())
    i = 0
    d_vector = p1 - p0
    print(d_vector)
    #for i in range(lin_samples): # 
    #### << PARA COMPUTAR ESTA METRICA: Usar ajuste / regression lineal y medir la distancia promedio entre los valores reales y los de la hiper recta para las mismas coordenadas
    #### << PARA GENERAR LOS VALORES: tomar dos samples y encontrar la recta que pasa por ese punto (ej Po + d->). Luego pasar valores de xo, x1...,xn-1 y calcular xn

    #from numpy import ones,vstack
    #from numpy.linalg import lstsq

    #points = [(1,5,3),(3,4,2),(5,3,2),(9,0,4)]
    #x_coords, y_coords,z_coords = zip(*points)
    #A = np.vstack([x_coords, y_coords,np.ones(len(x_coords))]).T
    #mx,my,c = np.linalg.lstsq(A, z_coords)[0]

    # Randomly permute features
    indices = np.arange(n_features)
    generator.shuffle(indices)
    X[:, :] = X[:, indices]

    #print(X)



create_dataset(n_samples=10, n_features=5,
                        perc_lin=20, perc_repeated=20, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=25,
                        shuffle=True,feat_dist=0)

