def calculate_z_score(x, mean, std_dev):
    return (x - mean) / std_dev

# Example usage:
mean = 1000
std_dev = 150
x = 1150

z_score = calculate_z_score(x, mean, std_dev)
print(f"The z-score is: {z_score}")
