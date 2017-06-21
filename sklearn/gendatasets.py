#!/usr/bin/python
from sklearn import datasets

# Genero un set det de datos
## make_blobs = nube de datos isotropica, gaussiana
atributos = 2
centros = 3
data,tags = datasets.make_blobs(n_samples=100, n_features=atributos, centers=centros, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

datasets.dump_svmlight_file(data,tags,'dataset.svl')
