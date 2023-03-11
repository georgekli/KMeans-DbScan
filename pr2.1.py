import scipy.io
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

########################################
# 2.1 Do DBSCAN clustering on data of mydata.mat
########################################
if __name__ == '__main__':
    # Load data from mydata
    mat_file = scipy.io.loadmat('mydata.mat')
    X = np.array(mat_file['X'])
    # Define DBSCAN parameters
    epsilon = 0.5
    MinPts = 15
    # Execute clustering
    dbscan = DBSCAN(eps=epsilon, min_samples=MinPts).fit(X)
    IDX = dbscan.labels_
    # Show results
    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

    plt.figure(1)
    plt.scatter(X[:, 0], X[:, 1], c=IDX)
    plt.show()
