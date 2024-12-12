import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
 # Install jqmcvi for Dunn Index calculation: pip install jqmcvi

# Sample Data
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=500, centers=5, random_state=42)

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# 1. Calculate Inertia (Within-Cluster Sum of Squares)
inertia = kmeans.inertia_
print(f"Inertia: {inertia}")

# 2. Calculate Dunn Index
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances

def dunn_index(X, labels):
    """
    Calculates the Dunn index for a given clustering.

    Parameters:
        X (numpy.ndarray): Data matrix.
        labels (numpy.ndarray): Cluster labels for each data point.

    Returns:
        float: Dunn index value.
    """

    n_clusters = len(np.unique(labels))

    # Calculate inter-cluster distances
    min_inter_cluster_dist = np.inf
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            cluster_i = X[labels == i]
            cluster_j = X[labels == j]
            dist = pairwise_distances(cluster_i, cluster_j).min()
            if dist < min_inter_cluster_dist:
                min_inter_cluster_dist = dist

    # Calculate intra-cluster distances
    max_intra_cluster_dist = 0
    for i in range(n_clusters):
        cluster_i = X[labels == i]
        dist = pairwise_distances(cluster_i).max()
        if dist > max_intra_cluster_dist:
            max_intra_cluster_dist = dist

    return min_inter_cluster_dist / max_intra_cluster_dist

dunn_index = dunn_index(X, labels)
print(f"Dunn Index: {dunn_index}")

# 3. Calculate Davies-Bouldin Index
db_index = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_index}")
