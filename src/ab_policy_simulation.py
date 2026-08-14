import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

np.random.seed(42)
df = pd.read_csv('data/raw/loan_data.csv', index_col=0)
df['default_flag'] = (df['outcome'] == 'default').astype(int)

# Policy simulation: DTI threshold at the median
dti_median = df['dti'].median()
print(f"DTI median (proposed policy cutoff): {dti_median}")
print()

group_low = df[df['dti'] <= dti_median]
group_high = df[df['dti'] > dti_median]

n_low, n_high = len(group_low), len(group_high)
defaults_low = group_low['default_flag'].sum()
defaults_high = group_high['default_flag'].sum()
rate_low = defaults_low / n_low
rate_high = defaults_high / n_high

print("=== Observational comparison: default rate by DTI group ===")
print(f"Low DTI (<= median):  n={n_low}, defaults={defaults_low}, default rate={rate_low:.4f}")
print(f"High DTI (> median):  n={n_high}, defaults={defaults_high}, default rate={rate_high:.4f}")
print(f"Absolute difference in default rate: {rate_high - rate_low:.4f}")
print()

count = np.array([defaults_high, defaults_low])
nobs = np.array([n_high, n_low])
z_stat, p_value = proportions_ztest(count, nobs, alternative='larger')

print("=== Two-proportion z-test ===")
print("H0: default rate (high DTI) = default rate (low DTI)")
print("H1: default rate (high DTI) > default rate (low DTI)  [one-tailed]")
print(f"z-statistic: {z_stat:.4f}")
print(f"p-value: {p_value:.4e}")
decision = "Reject H0" if p_value < 0.05 else "Fail to reject H0"
print(f"Decision at alpha=0.05: {decision}")
print()

print("=== Illustrative policy impact (backtest only, not causal) ===")
print(f"If loans with DTI > median had all been declined historically:")
print(f"  Defaults avoided (upper bound, assumes no substitution effect): {defaults_high}")
print(f"  Loans foregone (also would have excluded good borrowers): {n_high - defaults_high}")