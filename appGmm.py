from __future__ import division

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture


def myGmm(x, classesCount):
    kmeans = KMeans(n_clusters=classesCount).fit(x)
    meansArray = kmeans.cluster_centers_
    labels = kmeans.labels_

    sampleCount, featureCount = x.shape

    # init data
    covarianceMatrices = []
    weights = []
    for i in range(classesCount):
        classIndices = np.where(labels == i)[0]
        classValues = x[classIndices, :]

        covarianceMatrices.append(np.cov(classValues.T))
        weights.append(np.sum(labels == i) / sampleCount)

    # EM  Iteration
    convergence = False
    t = 0
    logLikelihood = []
    while not convergence:
        # Step 1: Expectation - Compute posteriors of each cluster
        posteriors = np.zeros((sampleCount, classesCount))
        for n in range(sampleCount):
            for k in range(classesCount):
                posteriors[n, k] = weights[k] * multivariate_normal.pdf(x[n, :],
                    mean=meansArray[k, :], cov=covarianceMatrices[k], allow_singular=False)
            posteriors[n, :] = posteriors[n, :] / np.sum(posteriors[n, :])

        # Step 2: Maximization - Update model parameters(means, covariance matrices and weights)
        sp = np.zeros(classesCount)
        for k in range(classesCount):
            # Update means
            sp[k] = np.sum(posteriors[:, k])
            posteriorIncreased = np.reshape(posteriors[:, k], (-1, 1))
            meansArray[k, :] = np.sum(posteriorIncreased * x, axis=0) / sp[k]

            # Update covariance matrices
            covarianceMatrices[k] = np.zeros((featureCount, featureCount))
            for n in range(sampleCount):
                zeroMeanXvector = (x[n, :] - meansArray[k, :])
                zeroMeanXmatrix = np.reshape(zeroMeanXvector, (-1, 1))
                covarianceMatrices[k] = covarianceMatrices[k] + np.dot(
                    posteriors[n, k],
                    np.dot(zeroMeanXmatrix, zeroMeanXmatrix.T))
            covarianceMatrices[k] = covarianceMatrices[k] / sp[k]

        # Update weights
        weights = sp / np.sum(sp)

        # Step 3: Evaluation - Compute log likelihood
        logLikelihood.append(0)
        for i in range(sampleCount):
            innerterm = 0
            for k in range(classesCount):
                innerterm = innerterm + weights[k] * multivariate_normal.pdf(x[i, :],
                    mean=meansArray[k, :], cov=covarianceMatrices[k], allow_singular=False)
            logLikelihood[t] = logLikelihood[t] + np.log(innerterm)

        logLikelihood[t] = logLikelihood[t] / sampleCount

        if t > 0:
            convergence = (logLikelihood[t] - logLikelihood[t - 1]) < 0.000001
            logLikelihood[t] - logLikelihood[t - 1]
        print("Step: %d, logLikelihood: %f" % (t, logLikelihood[t]))
        t = t + 1
    return meansArray, covarianceMatrices, weights


# def plot_results(X, Y_, means, covariances, index, title):
#     splot = plt.subplot(2, 1, 1 + index)
#     for i, (mean, covar) in enumerate(zip(
#             means, covariances)):
#         v, w = linalg.eigh(covar)
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         u = w[0] / linalg.norm(w[0])
#         # as the DP will not use every component it has access to
#         # unless it needs it, we shouldn't plot the redundant
#         # components.
#         if not np.any(Y_ == i):
#             continue
#         plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8)
#
#         # Plot an ellipse to show the Gaussian component
#         angle = np.arctan(u[1] / u[0])
#         angle = 180. * angle / np.pi  # convert to degrees
#         ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle)
#         ell.set_clip_box(splot.bbox)
#         ell.set_alpha(0.5)
#         splot.add_artist(ell)
#
#     plt.xlim(-9., 5.)
#     plt.ylim(-3., 6.)
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(title)


def main():
    plt.close('all')

    classesCount = 3
    sampleCount = 300
    x, y = make_blobs(n_samples=sampleCount, centers=classesCount, n_features=2, cluster_std=1, random_state=0)
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=y, s=50)
    plt.show()

    gmm = GaussianMixture(n_components=classesCount, random_state=0).fit(x)

    sklearnLabels = gmm.predict(x)
    sklearMeans = gmm.means_
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=sklearnLabels, s=50)
    plt.scatter(sklearMeans[:, 0], sklearMeans[:, 1], c='black', s=200, alpha=0.5)
    plt.show()

    meansArray, covarianceMatrices, weights = myGmm(x, classesCount)

    posteriors = np.zeros((sampleCount, classesCount))
    for n in range(sampleCount):
        for k in range(classesCount):
            posteriors[n, k] = np.dot(weights[k],
                multivariate_normal.pdf(x[n, :], mean=meansArray[k, :],
                    cov=covarianceMatrices[k], allow_singular=False))
    myLabels = posteriors.argmax(axis=1)

    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], c=myLabels, s=50)
    plt.scatter(meansArray[:, 0], meansArray[:, 1], c='red', s=200, alpha=0.5)
    plt.show()

    print("Sklearn gmm precision: %f" % adjusted_rand_score(y, sklearnLabels))
    print("My gmm precision: %f" % adjusted_rand_score(y, myLabels))

    # plot_results(x, gmm.predict(x), gmm.means_, gmm.covariances_, 0,
    #              'Gaussian Mixture')


if __name__ == '__main__':
    main()
