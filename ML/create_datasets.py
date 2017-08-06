import numpy as np



def create_dataset(n_samples=10, n_features=3,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):

    # Harcoded seed
    seed = 563456

    # Random numbers generator
    generator = np.random.RandomState(seed)

    # Initialize dataset 
    X = np.zeros((n_samples, n_features))


    n_columns = 2

    # Generate a matrix of n_samples X n_columns with random samples from the “standard normal” distribution
    m1 = generator.randn(n_samples,n_columns)

    # Generate a matrix of n_samples X n_columns with random samples from a uniform distribution over [0, 1)
    m2 = generator.rand(n_samples,n_columns)
    ## Round floats to two decimals
    m2.round(decimals=2)


    # Generate a matrix of n_samples X n_columns with random samples from a uniform distribution over [0, 1)
    m3 = generator.random_integers(low=-10, high=10, size=(n_samples,n_columns))



    # Populating only the first "n_columns" columns of the matrix
    X[:, :n_columns] = m1



    ## Generating redundant atributes 
    # Creating matrix B which are basically columns of random numbers multplied by 2 and substracted 1
    B = 2 * generator.rand(n_informative, n_redundant) - 1

    # Redundant values are generated by "dot / escalar" multiplying B (random numbers) by the informative values (linear combination?)
    redundant_column = np.dot(X[:, :n_informative], B)

    # Adding redundant column to X
    X[:, n_informative:n_informative + n_redundant] = redundant_column



def create_dataset(n_samples=10, n_features=3, n_informative=0,
                        n_redundant=2, n_repeated=0, n_classes=2,
                        n_clusters_per_class=2, weights=None, flip_y=0.01,
                        class_sep=1.0, shift=0.0, scale=1.0,
                        shuffle=True, random_state=None):
