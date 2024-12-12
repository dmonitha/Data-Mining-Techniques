import numpy as np

def z_score_normalization(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    z_scores = (data - mean) / std_dev
    return z_scores

# Example usage
data = np.array([70, 80, 90, 100, 110, 130, 150])
normalized_data = z_score_normalization(data)

print("Original data:", data)
print("Z-score normalized data:", normalized_data)
