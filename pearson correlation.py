from scipy.stats import pearsonr

# Example data
x = [25,30,36,43]
y = [30000,44000,52000,7000]

# Calculate Pearson correlation coefficient
correlation_coefficient, p_value = pearsonr(x, y)

print("Pearson correlation coefficient:", correlation_coefficient)
print("p-value:", p_value)
