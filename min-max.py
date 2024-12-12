import numpy as np

def min_max_normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

# Example usage
data = np.array([14, 9, 24, 39, 60])
normalized_data = min_max_normalization(data)

print("Original data:", data)
print("Normalized data:", normalized_data)
