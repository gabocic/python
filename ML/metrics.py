#!/home/gabriel/pythonenvs/v3.5/bin/python

from sklearn import metrics

def clustering_metrics(estimator, name, data, time, sample_size):
    proc_metrics={}
    proc_metrics['name'] =  name
    proc_metrics['time'] = time
    proc_metrics['inertia'] = estimator.inertia_
    proc_metrics['silhouette_score'] = metrics.silhouette_score(data, estimator.labels_,metric='euclidean',sample_size=sample_size)
    print(proc_metrics) 
