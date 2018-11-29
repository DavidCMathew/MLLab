import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


data = pd.read_csv('data/ex.csv')

f1 = data['V1'].values
f2 = data['V2'].values
X = np.array(list(zip(f1, f2)))
print("x: ", X)

print('Graph for whole dataset')
plt.scatter(f1, f2, c='black')  # size can be set by adding s=size as param
plt.show()

kmeans = KMeans(2)
labels = kmeans.fit(X).predict(X)
print("labels for kmeans:", labels)

print('Graph using Kmeans Algorithm')
plt.scatter(f1, f2, c=labels)

centroids = kmeans.cluster_centers_
print("centroids:", centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', c='red')

plt.show()

gmm = GaussianMixture(2)
labels = gmm.fit(X).predict(X)
print("Labels for GMM: ", labels)

print('Graph using EM Algorithm')
plt.scatter(f1, f2, c=labels)
plt.show()
