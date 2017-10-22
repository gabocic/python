#!/home/gabriel/pythonenvs/v3.5/bin/python

from dataset import create_dataset
from analyze import analyze_dataset
from preprocessing import sklearn_scale
from metrics import clustering_metrics
from kmeans import k_means_clustering

def main():

    # Generate dataset
    dataset = create_dataset(n_samples=2000, n_features=3,
                        perc_lin=20, perc_repeated=0, n_groups=2,
                        avg_sample_dist=1.0, shift=0.0, scale=1.0, perc_feat_lin_dep=0,
                        shuffle=True,feat_dist=0,debug=0,plot=0,save_to_file=0)

    # Validate dataset is within the specifications
    analyze_dataset(data=dataset,debug=0,plot=0,load_from_file=None)

    # Scale data
    scaleddata = sklearn_scale(dataset) 

    # Find clusters using K-means algorithm
    estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='k-means++',p_n_clusters=3,p_n_init=10,p_n_jobs=4)
    #estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='random',p_n_clusters=3,p_n_init=10,p_n_jobs=4)
    #estimator,elap_time = k_means_clustering(data=scaleddata,plot=0,p_init='PCA-based',p_n_clusters=3,p_n_init=10,p_n_jobs=4)
    
    # Compute clustering metrics
    sample_size = 50
    clustering_metrics(estimator, 'k-means-plusplus', scaleddata, elap_time, sample_size)

if __name__ == '__main__':
    main()
