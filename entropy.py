import math

def calculate_entropy(p):
    if p == 0 or p == 1:
        return 0
    return p * math.log2(p)

entropy = calculate_entropy(3/6)
print(f"Entropy: {entropy}")
