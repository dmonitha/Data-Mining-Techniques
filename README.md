# Data-Mining-Techniques
Python code for various data mining techniques calculation

# Model Evaluation and Selection Techniques- 
TP -> True Positives
TN -> True Negatives
FP -> False Positives 
FN -> False Negatives

Confusion Matrix 
 TP | FP
 ---|---
 FN | TN

* Accuracy -
  - Measures overall correctness of the model's predictions.
  - Formula - $$TP + TN / P+N$$
* Misclassification/ Error Rate -
  - Measures the proportion of incorrect predictions out of all predictions
  - Formula - $$FP+FN/ P+N$$
* Sensitivity/ True Positive Rate/ Recall -
    - Indicates the rate of correctly predicited positives
    - Formula - $$TP/P$$
* Specificity/ True Negative Rate -
    - Proportion of actual negatives correctly identified
    - Formula - $$TN/N$$
* Precision-
  - Proportion of predicted positives that are actual positives.
  - Formula - $$TP/TP+FP$$
* F-1 Score-
    - Harmonic mean of Precision and Recall, balancing the two metrics'
    - Formula - $$2 * Precision * Recall / Precision + Recall$$

# Density Based Clustering -
* DBSCAN - Density-based spatial clustering of applications with noise
  - Epsilon (𝜀): The maximum distance within which points are considered neighbors.
  - MinPts: The minimum number of points required to form a dense region (a core point).
  - Core Point: A point with at least MinPts neighbors within the 𝜀-radius.
  - Border Point: A point that is within 𝜀 of a core point but has fewer than MinPts neighbors itself.
  - Noise Point: A point that does not belong to any cluster (neither a core nor border point).\
 
# Data Discretization - 
The process of transforming continuous data into discrete intervals or bins. 

- Equal Distance (Equal Width Binning): Divides the range of data into intervals of equal size.
- Equal Frequency (Equal Depth Binning): Divides data into intervals so that each bin contains approximately the same number of data points.



