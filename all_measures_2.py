# Calculating Accuracy, Precision, TPR,TNR, F1 Score when input is given as a table entry

import pandas as pd
from tabulate import tabulate
import numpy as np

# Sample data: target and pred columns
data = {
    "target": ['spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam', 'spam',
               'ham', 'spam', 'ham', 'ham', 'ham', 'ham', 'ham', 'spam', 'ham', 'ham'],
    "pred": ['ham', 'ham', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam', 'spam',
             'spam', 'ham', 'ham', 'ham', 'ham', 'ham', 'spam', 'spam', 'ham', 'spam']
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Calculate confusion matrix
confusion_matrix = pd.crosstab(df['target'], df['pred'], rownames=['Actual'], colnames=['Predicted'])

# Convert confusion matrix to numpy array
labels = ['ham', 'spam']  # Ensure the order is consistent
matrix = confusion_matrix.reindex(index=labels, columns=labels, fill_value=0).to_numpy()

# Define metrics calculation function
def confusion_matrix_metrics(confusion_matrix):
    if isinstance(confusion_matrix, list):
        confusion_matrix = np.array(confusion_matrix)
    
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2")
    
    tn, fp, fn, tp = confusion_matrix.ravel()
    p = tp + fn  # Total positives
    n = tn + fp  # Total negatives

    metrics = {}
    
    # Accuracy, recognition rate
    metrics['Accuracy/ Recognition Rate'] = (tp + tn) / (p + n)
    
    # Error rate, misclassification rate
    metrics['Error Rate/Misclassification'] = (fp + fn) / (p + n)
    
    # Sensitivity, true positive rate, recall
    metrics['Sensitivity/TPR/Recall'] = tp / p if p > 0 else 0
    
    # Specificity, true negative rate
    metrics['Specificity/TNR'] = tn / n if n > 0 else 0
    
    # Precision
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # F-score (F1 score)
    precision = metrics['Precision']
    recall = metrics['Sensitivity/TPR/Recall']
    metrics['F1-score'] = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    beta = 1
    beta_squared = beta ** 2
    metrics['F-beta score'] = ((1 + beta_squared) * precision * recall) / ((beta_squared * precision) + recall) if (precision + recall) > 0 else 0

    return metrics

# Calculate metrics
results = confusion_matrix_metrics(matrix)

# Prepare data for tabulate
table_data = [[metric, f"{value:.4f}"] for metric, value in results.items()]

# Print the table
print("Confusion Matrix:")
print(confusion_matrix)
print("\nMetrics:")
print(tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid"))
