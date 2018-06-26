from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def myKMeans(x, classesCount):
    xCopy = x.copy()
    xCopy = xCopy.T
    featureCount, sampleSize = xCopy.shape

    centroids = np.random.rand(featureCount, classesCount) * x.max()
    oldDistances = np.zeros((classesCount, sampleSize))
    changes = True
    while changes:
        distances = np.empty((classesCount, sampleSize))
        for i in range(classesCount):
            result = np.sqrt(
                np.sum(
                    np.power(
                        xCopy - centroids[:, i].reshape(featureCount, 1), 2), axis=0))
            distances[i, :] = result

        difference = distances - oldDistances
        if np.sum(difference) == 0:
            changes = False

        labels = distances.argmin(axis=0)
        centroids = np.empty((featureCount, classesCount))
        for i in range(classesCount):
            classIndices = np.where(labels == i)[0]
            classValues = xCopy[:, classIndices]
            newCentroid = np.mean(classValues, axis=1)
            if np.sum(np.isnan(newCentroid)) > 0:
                newCentroid = np.random.rand(featureCount) * x.max()
            centroids[:, i] = newCentroid
        oldDistances = distances
    return labels, centroids


def main():
    plt.close('all')

    classesCount = 3
    sampleCount = 300
    x, y = make_blobs(n_samples=sampleCount, centers=classesCount, n_features=2, cluster_std=1, random_state=0)

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
    plt.show()

    kmeans = KMeans(n_clusters=classesCount, random_state=0).fit(x)

    sklearnLabels = kmeans.predict(x)
    sklearnCentroids = kmeans.cluster_centers_

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=sklearnLabels, s=50)
    plt.scatter(sklearnCentroids[:, 0], sklearnCentroids[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    myLabels, myCentroids = myKMeans(x, classesCount)
    myLabels = myLabels.T
    myCentroids = myCentroids.T

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=myLabels, s=50)
    plt.scatter(myCentroids[:, 0], myCentroids[:, 1], c='red', s=200, alpha=0.5)
    plt.show()

    print("Sklearn k-means precision: %f" % adjusted_rand_score(y, sklearnLabels))
    print("My k-mean precision: %f" % adjusted_rand_score(y, myLabels))


if __name__ == '__main__':
    main()
