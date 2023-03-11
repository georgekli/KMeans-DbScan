import sklearn
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
########################################
# 1.1 Do k-means clustering on some dimensions of iris dataset
########################################
if __name__ == '__main__':
    meas = load_iris().data
    # We initially use the last 2 data columns
    X = meas[:, [2, 3]]
    k = 3  # We need to do 3 class clustering
    # init = "random" # Test random initial centroids
    kmeans = KMeans(n_clusters=k).fit(X)  # Do k-means algorithm
    IDX = kmeans.labels_ # Get the labels of the clusters calculated
    C = kmeans.cluster_centers_ # Get the cluster centers
    plt.figure(1)
    # Plot the data indexes as clusters
    plt.plot(IDX[:], 'o')
    plt.show()
    # Plot the centroids and data in scatter
    plt.plot(X[IDX == 0][:, 0], X[IDX == 0][:, 1], 'limegreen', marker='o', linewidth=0, label='C1')
    plt.plot(X[IDX == 1][:, 0], X[IDX == 1][:, 1], 'yellow', marker='o', linewidth=0, label='C2')
    plt.plot(X[IDX == 2][:, 0], X[IDX == 2][:, 1], 'c.', marker='o', label='C3')
    plt.scatter(C[:, 0], C[:, 1], marker='x', color='black', s=150, linewidth=3, label="Centroids", zorder=10)
    plt.legend()
    plt.show()
    # Calculate Sum of Squered Errors
    SSE = kmeans.inertia_
    print("SSE is:", SSE)
    # Calculate Silouhette score
    sil = sklearn.metrics.silhouette_score(X, IDX)
    print("Silhouette score is:", sil)

    # Try to examine the k and SSE as well as k and SilScore function
    SSEs = [0] * 20
    SILs = [0] * 20
    for i in range(2, 20):
        # Do the same clustering for variable ks 2 to 10
        X = meas[:, [1, 3]]
        k = i
        kmeans = KMeans(n_clusters=k).fit(X)
        IDX = kmeans.labels_  # Get the labels of the clusters calculated
        # Calculate Metrics
        SSE = kmeans.inertia_
        sil = sklearn.metrics.silhouette_score(X, IDX)
        SSEs[i] = SSE
        SILs[i] = sil
    # Show results of SSE and k as well as SilScore and k
    plt.figure(2)
    plt.plot(range(2, 20), SSEs[2:20], 'go-', label='line 1', linewidth=2)
    plt.title("SSE(k)")
    plt.xlabel("k-clusters used")
    plt.ylabel("SSE")
    plt.show()
    plt.figure(3)
    plt.plot(range(2, 20), SILs[2:20], 'go-', label='line 1', linewidth=2)
    plt.title("Silhouette_Score(k)")
    plt.xlabel("k-clusters used")
    plt.ylabel("Silhouette score")
    plt.show()
