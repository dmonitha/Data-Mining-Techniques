import numpy as np
from sklearn.cluster import KMeans

def kmeans_outlier_detection(data, k, threshold=2.0):
    """
    Detects outliers using k-means clustering.

    Parameters:
        data (list of tuples or numpy array): Coordinates of data points (e.g., [(x1, y1), (x2, y2), ...]).
        k (int): Number of clusters for k-means.
        threshold (float): Multiplier for the mean distance to identify outliers (default is 2.0).

    Returns:
        clusters (dict): A dictionary with cluster labels as keys and lists of points as values.
        outliers (list): A list of points identified as outliers.
    """
    # Convert data to a numpy array
    data = np.array(data)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    
    # Get cluster labels and centroids
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Calculate distances from each point to its cluster centroid
    distances = np.linalg.norm(data - centroids[labels], axis=1)

    # Compute the mean and standard deviation of distances for each cluster
    cluster_distances = {i: [] for i in range(k)}
    for i, label in enumerate(labels):
        cluster_distances[label].append(distances[i])

    cluster_mean_std = {
        i: (np.mean(dist), np.std(dist)) for i, dist in cluster_distances.items()
    }

    # Identify outliers
    outliers = []
    for i, (point, label) in enumerate(zip(data, labels)):
        mean, std_dev = cluster_mean_std[label]
        if distances[i] > mean + threshold * std_dev:
            outliers.append(tuple(point))

    # Group points by cluster
    clusters = {i: [] for i in range(k)}
    for point, label in zip(data, labels):
        clusters[label].append(tuple(point))

    return clusters, outliers,centroids

# Example usage
if __name__ == "__main__":
    # Example dataset: 2D points
    data_points = [
        (2,10),(2,6),(11,11),(6,9),(6,4),(1,2),(5,10),(4,9),(10,12),(7,5),(9,11),(4,6),(3,10),(3,8),(6,11)
    ]

    # Number of clusters
    k = 3

    # Detect clusters and outliers
    clusters, outliers, centroids = kmeans_outlier_detection(data_points, k)

    print("Clusters:")
    for cluster, points in clusters.items():
        print(f"Cluster {cluster}: {points}")

    print("\nCentroids:")
    for i, centroid in enumerate(centroids):
        print(f"Centroid {i}: {centroid}")

    print("\nOutliers:")
    print(outliers)