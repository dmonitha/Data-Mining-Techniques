import numpy as np

def equal_width_binning(data, num_bins):
    # Convert data to numpy array
    data = np.array(data)
    
    # Calculate min and max values
    min_val, max_val = np.min(data), np.max(data)
    
    # Calculate bin width
    bin_width = (max_val - min_val) / num_bins
    
    # Create bin edges
    bin_edges = [min_val + i * bin_width for i in range(num_bins + 1)]
    
    # Initialize dictionary to store binned data
    binned_data = {i: [] for i in range(num_bins)}
    
    # Assign data points to bins
    for value in data:
        for i in range(num_bins):
            if bin_edges[i] <= value < bin_edges[i + 1]:
                binned_data[i].append(value)
                break
        else:
            # For the last bin, include the max value
            binned_data[num_bins - 1].append(value)
    
    return binned_data, bin_edges

# Example usage
data = [5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215]
num_bins = 3

binned_result, bin_edges = equal_width_binning(data, num_bins)

print("Bin edges:", bin_edges)
for bin_num, values in binned_result.items():
    print(f"Bin {bin_num}: {values}")
