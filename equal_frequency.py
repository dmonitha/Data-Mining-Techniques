import numpy as np
import pandas as pd

def equal_frequency_binning(data, num_bins):
    # Convert data to a pandas Series
    series = pd.Series(data)
    
    # Use pd.qcut() to perform equal frequency binning
    binned_data = pd.qcut(series, q=num_bins, labels=False)
    
    # Group the original data by the bin labels
    result = series.groupby(binned_data).apply(list).to_dict()
    
    return result

# Example usage
data = [5, 10, 11, 13, 15, 35, 50, 55, 72, 92, 204, 215]
num_bins = 3

binned_result = equal_frequency_binning(data, num_bins)

for bin_num, values in binned_result.items():
    print(f"Bin {bin_num}: {values}")
