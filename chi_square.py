from scipy.stats import chi2_contingency, chi2

# defining the table
data = [[88,93,110], [112,107,90]]
stat, p, dof, expected = chi2_contingency(data)

# interpret p-value
# significance level = 1- confidence level
alpha = 0.10
print("degree of freedon",dof)
critical_value = chi2.ppf(1 - alpha, dof)
print("critical value:", critical_value)
print("chi square test statistic",stat)
print("p value is " + str(p))
if p <= alpha:
    print('Dependent (reject null hypothesis H0)')
else:
    print('Independent (H0 holds true), accept null hypothesis')