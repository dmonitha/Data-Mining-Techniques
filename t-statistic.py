import numpy as np
from scipy import stats

def calculate_t_statistic(data1, data2=None, paired=False, population_mean=None):
    """
    Calculate t-statistic for different types of t-tests.
    
    :param data1: First sample data (numpy array or list)
    :param data2: Second sample data for two-sample tests (numpy array or list)
    :param paired: Boolean indicating if the test is paired (default: False)
    :param population_mean: Known population mean for one-sample test (default: None)
    :return: t-statistic
    """
    if data2 is None and population_mean is None:
        raise ValueError("For one-sample t-test, provide population_mean")
    
    if data2 is not None and population_mean is not None:
        raise ValueError("Cannot perform both one-sample and two-sample test simultaneously")
    
    if population_mean is not None:
        # One-sample t-test
        sample_mean = np.mean(data1)
        sample_std = np.std(data1, ddof=1)
        n = len(data1)
        t_stat = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    elif paired:
        # Paired t-test
        d = np.array(data1) - np.array(data2)
        t_stat, _ = stats.ttest_rel(data1, data2)
    else:
        # Independent two-sample t-test
        t_stat, _ = stats.ttest_ind(data1, data2, equal_var=False)
    
    return t_stat

# Example usage
data1 = [1, 2, 3, 4, 5]
data2 = [2, 4, 6, 8, 10]
population_mean = 3

print("One-sample t-statistic:", calculate_t_statistic(data1, population_mean=population_mean))
print("Independent two-sample t-statistic:", calculate_t_statistic(data1, data2))
print("Paired t-statistic:", calculate_t_statistic(data1, data2, paired=True))
