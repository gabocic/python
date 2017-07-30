import numpy as np



def create_dataset(n_samples=10, n_features=3, n_informative=0,
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

	# Populating only the first "n_columns" columns of the matrix
	n_columns = 2
	X[:, :n_columns] = generator.randn(n_samples,n_columns)
