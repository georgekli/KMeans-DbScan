import scipy.io
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import zscore

########################################
# 2.2 Difference of non-normalized and normalized data on DBSCAN clustering of Iris
########################################
if __name__ == '__main__':
    # Load data from mydata
    meas = load_iris().data
    X = meas[:, [2, 3]]
    # Define DBSCAN parameters
    epsilon = 0.1
    MinPts = 5
    # Execute clustering without normalization
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Plot data
    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()
    # Plot clusters and data
    plt.figure(2)
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()

    # Now we should test if normalizing input data changes results
    # Normalize data
    xV1 = zscore(X[:,0])
    xV2 = zscore(X[:,1])
    X = np.vstack([xV1, xV2]).T
    # Execute clustering with normalization
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Show results without clusters
    plt.figure(3)
    plt.scatter(xV1, xV2)
    plt.show()
    # Plot clusters and data
    plt.figure(4)
    plt.scatter(xV1, xV2, c=IDX)
    plt.show()
