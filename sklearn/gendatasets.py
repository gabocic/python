#!/usr/bin/python
from sklearn import datasets
import argparse

# Genero un set det de datos
## make_blobs = nube de datos isotropica, gaussiana


parser = argparse.ArgumentParser(description='Create a ramdom data set')
parser.add_argument('-a','--attributes', type=int, nargs='+',required=True,
                           help='Atributes')
parser.add_argument('-c','--centers', type=int, nargs='+',required=True,
                           help='Centers')
args = parser.parse_args()

data,tags = datasets.make_blobs(n_samples=200, n_features=args.attributes[0], centers=args.centers[0], cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

datasets.dump_svmlight_file(data,tags,'dataset.svl')
