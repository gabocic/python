#!/usr/bin/python

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn import datasets
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.datasets import load_svmlight_file

# Genero un set det de datos
## make_blobs = nube de datos isotropica, gaussiana
#atributos = 2
#centros = 3
#data,tags = datasets.make_blobs(n_samples=100, n_features=atributos, centers=centros, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None)

data,tags = datasets.load_svmlight_file('dataset.svl')

print(data)
print(tags)

#Preprocessing values
#scaleddata = scale(data)


## Extraigo la cantidad de clases del array de etiquetas
n_digits = len(np.unique(tags))


kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)

print(tags)

kmeans.fit(data)
print(kmeans.labels_)

#kmeans.fit(scaleddata)
print(kmeans.labels_)


# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
#plt.figure(1)
#plt.clf()
#plt.imshow(Z, interpolation='nearest',
#           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
#           cmap=plt.cm.Paired,
#           aspect='auto', origin='lower')

plt.plot(data[:, 0], data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
#centroids = kmeans.cluster_centers_
#plt.scatter(centroids[:, 0], centroids[:, 1],
#            marker='x', s=169, linewidths=3,
#            color='w', zorder=10)
#plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
#          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

