import scipy.io
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn import metrics

########################################
# 2.3 Difference of epsilon and MinPts parameters on DBSCAN clustering of xV.mat
########################################
if __name__ == '__main__':
    # Load data from mydata
    mat_file = scipy.io.loadmat('xV.mat')
    xV = np.array(mat_file['xV'])
    X = xV[:, [0, 1]]
    # Define DBSCAN parameters
    epsilon = 0.3
    MinPts = 50
    # Execute clustering for the first 2 features
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Plot clusters and data
    plt.figure(1)
    plt.title("ε=0.3 MinPts=50")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()

    # Define the new DBSCAN parameters for the first 2 features
    epsilon = 0.6
    MinPts = 43
    # Execute clustering again
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Plot clusters and data
    plt.figure(2)
    plt.title("ε=0.6 MinPts=43")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()
    # Try more for epsilon and minPoints
    epsilon = 0.4
    MinPts = 30
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    plt.figure(2)
    plt.title("ε=0.4 MinPts=30")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()
    epsilon = 0.2
    MinPts = 40
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    plt.figure(2)
    plt.title("ε=0.2 MinPts=40")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()

    # Define the new DBSCAN parameters for the last 2 features
    X = xV[:, [467, 468]]
    epsilon = 0.005
    MinPts = 10
    # Execute clustering again
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Plot clusters and data
    plt.figure(3)
    plt.title("ε=0.005 MinPts=10")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()
    # Try more for epsilon and minPoints
    epsilon = 0.02
    MinPts = 30
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    plt.figure(2)
    plt.title("ε=0.02 MinPts=30")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()
    epsilon = 0.04
    MinPts = 2
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    plt.figure(3)
    plt.title("ε=0.04 MinPts=2")
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()