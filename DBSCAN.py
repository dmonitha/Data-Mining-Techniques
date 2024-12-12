import math

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def get_neighbors(points, eps):
    neighbors = {}
    for i, p1 in enumerate(points):
        neighbors[chr(65 + i)] = set()
        for j, p2 in enumerate(points):
            if i != j and distance(p1, p2) <= eps:
                neighbors[chr(65 + i)].add(chr(65 + j))
    return neighbors

def find_core_points(neighborhoods, min_pts):
    return [point for point, neighbors in neighborhoods.items() if len(neighbors) >= min_pts - 1]

def find_outliers(neighborhoods, core_points):
    return [point for point in neighborhoods.keys() if point not in core_points]

# Define the points
points = [(3, 7), (4, 6), (5, 5), (6, 4), (7, 3), (6, 2), (7, 2), (8, 4)]

# Parameters
eps = 2.5
min_pts = 3

# Calculate neighborhoods
neighborhoods = get_neighbors(points, eps)

# Find core points and outliers
core_points = find_core_points(neighborhoods, min_pts)
outliers = find_outliers(neighborhoods, core_points)

# Print neighborhoods
print("Neighborhoods:")
for point, neighbors in neighborhoods.items():
    print(f"N({point}) = {{{', '.join(sorted(neighbors))}}}")

# Print core points
print("\nCore Points:")
print(", ".join(sorted(core_points)))

# Print outliers
print("\nOutliers:")
print(", ".join(sorted(outliers)))
