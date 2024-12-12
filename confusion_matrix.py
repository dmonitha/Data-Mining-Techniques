import numpy as np

def confusion_matrix_metrics(confusion_matrix):
    """
    Calculate metrics from a confusion matrix.
    
    :param confusion_matrix: 2x2 numpy array or list of lists
    :return: dict of calculated metrics
    """
    if isinstance(confusion_matrix, list):
        confusion_matrix = np.array(confusion_matrix)
    
    if confusion_matrix.shape != (2, 2):
        raise ValueError("Confusion matrix must be 2x2")

    tn, fp, fn, tp = confusion_matrix.ravel()

    metrics = {}
    
    # Basic counts
    metrics['True Positives'] = tp
    metrics['True Negatives'] = tn
    metrics['False Positives'] = fp
    metrics['False Negatives'] = fn
    
    # Accuracy
    metrics['Accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # Precision
    metrics['Precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall (Sensitivity)
    metrics['Recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # Specificity
    metrics['Specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # F1 Score
    metrics['F1 Score'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall']) if (metrics['Precision'] + metrics['Recall']) > 0 else 0
    
    # Negative Predictive Value
    metrics['NPV'] = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    # False Discovery Rate
    metrics['FDR'] = fp / (fp + tp) if (fp + tp) > 0 else 0
    
    # False Omission Rate
    metrics['FOR'] = fn / (fn + tn) if (fn + tn) > 0 else 0
    
    return metrics

# Example usage
confusion_matrix = [
    [2588, 412],  # [TN, FP]
    [46, 6954]   # [FN, TP]
]

results = confusion_matrix_metrics(confusion_matrix)

for metric, value in results.items():
    print(f"{metric}: {value:.4f}")
