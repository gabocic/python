#!/home/gabriel/pythonenvs/v3.5/bin/python

from dataset import create_dataset
from analyze import analyze_dataset
from preprocessing import sklearn_scale
from kmeans import k_means_clustering

def main():

    # Generate dataset
    dataset = create_dataset(n_samples=100, n_features=3,
                        perc_lin=20, perc_repeated=0, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=0,
                        shuffle=True,feat_dist=0,debug=0,plot=1,save_to_file=0)

    # Validate dataset is within the specifications
    analyze_dataset(data=dataset,debug=0,plot=1,load_from_file=None)

    # Scale data
    scaleddata = sklearn_scale(dataset) 

    # Find clusters using K-means algorithm
    k_means_clustering(scaleddata)

if __name__ == '__main__':
    main()
