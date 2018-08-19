#### DATASET ###############

# Max amount of manipulated points
max_special_points_perc = 90

# Minimum dataset samples
min_dataset_size = 100

# Features values range: all features values will be between value_limit and -value_limit
value_limit = 10000

# Numbers of attempts to generate a dataset that fulfill the specified characteristics 
dataset_gen_retry = 100

# Number of decimals for features values
feat_values_decimals = 3

# When generating linear points, we use the below factor to prevent too many outliers. It is basically affecting the point's "lambda" (r-> = p0 + Lambda*d->)
linear_points_lambda_adj_factor = 300

# Noise factor to prevent all linear points from belonging to straight line
linear_noise_factor = 0.01

# Minimum number of samples per group as percentage of the dataset size
repeated_min_samples_per_group_perc = 0.05 # 5%

#### METRICS ###############

# Precision
metric_decimals = 4




#### ANALYSIS ###############

# Dataset parameters tolerance when validating it
dataset_parameters_error_margin = 0.05 # 5%

# Degree of parallelism for any processes that support jobs
parallelism=4

# Minimum number of samples for a group to be considered (Repeated analysis)
analysis_group_min_members_perc = 0.08 # 8%

# This parameter determines how many times the distance of the furthest point a point should be to be considered an Outlier
analysis_outlier_factor = 1.5

# Fraction of the Fit line that deternines the max radius for a point to be considered Linear
analysis_fit_line_fraction = 0.025

# On datasets with more than 60% of linear points, we check for another kind of outliers: those that are further from the fit line by "fraction x" of the fit line. This parameter determines that fraction
analysis_fit_line_fraction_outliers = 0.3


#### MEANSHIFT ###############

# Percentage of dataset samples used to estimate bandwidth
#ms_estimate_bandwidth_samples_perc = 1 # 100%



#### DBSCAN ###############

# Minimum percentage of the total dataset to be considered a cluster
dbs_min_samples_per_cluster_perc = 0.1 # 10%

# Minimum amount of clusters
min_clusters = 2

