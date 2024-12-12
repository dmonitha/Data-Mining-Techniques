import numpy as np
from sklearn.cluster import KMeans

# Function to calculate centroid
def calculate_centroid(points):
    """
    Calculate the centroid of a set of points.
    :param points: List or array of points (each point is a tuple or array of coordinates).
    :return: Centroid as a numpy array.
    """
    return np.mean(points, axis=0)

# Function to calculate within-cluster variation (WCSS)
def calculate_within_cluster_variation(cluster_points, centroid):
    """
    Calculate the within-cluster variation (sum of squared distances from centroid).
    :param cluster_points: Points in the cluster.
    :param centroid: Centroid of the cluster.
    :return: Within-cluster variation.
    """
    return np.sum((cluster_points - centroid) ** 2)

# Function to calculate Sum of Squared Error (SSE) for all clusters
def calculate_sse(data, labels, centroids):
    """
    Calculate SSE for all clusters.
    :param data: Original dataset.
    :param labels: Cluster labels for data points.
    :param centroids: Centroids of the clusters.
    :return: SSE value.
    """
    sse = 0
    for i, centroid in enumerate(centroids):
        cluster_points = data[labels == i]
        sse += np.sum((cluster_points - centroid) ** 2)
    return sse

# Example usage
if __name__ == "__main__":
    # Example dataset
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [10, 2], [10, 4], [10, 0]])

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42).fit(data)
    
    # Extract centroids and labels
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Calculate centroid for each cluster
    print("Centroids:", centroids)

    # Calculate within-cluster variation for each cluster
    for i in range(len(centroids)):
        cluster_points = data[labels == i]
        wcv = calculate_within_cluster_variation(cluster_points, centroids[i])
        print(f"Within-cluster variation for Cluster {i}: {wcv}")

    # Calculate SSE for all clusters
    sse = calculate_sse(data, labels, centroids)
    print("Sum of Squared Error (SSE):", sse)
