import pandas as pd
import numpy as np
import statsmodels.api as sm

np.random.seed(42)
df = pd.read_csv('data/raw/loan_data.csv', index_col=0)

# ---------- Poisson Regression: open_acc ~ annual_inc + dti + loan_amnt + borrower_score ----------
features = ['annual_inc', 'dti', 'loan_amnt', 'borrower_score']
X = sm.add_constant(df[features])
y = df['open_acc']

print("=== Overdispersion check ===")
print(f"Mean(open_acc): {y.mean():.4f}")
print(f"Variance(open_acc): {y.var():.4f}")
print(f"Variance/Mean ratio: {y.var()/y.mean():.4f}")
print()

poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()
print("=== Poisson Regression: open_acc ~ annual_inc + dti + loan_amnt + borrower_score ===")
print(poisson_model.summary())
print()

# Pearson chi2 / df as formal overdispersion statistic
pearson_chi2 = poisson_model.pearson_chi2
df_resid = poisson_model.df_resid
dispersion = pearson_chi2 / df_resid
print(f"Pearson chi2: {pearson_chi2:.2f}")
print(f"Degrees of freedom (resid): {df_resid}")
print(f"Dispersion statistic (chi2/df): {dispersion:.4f}")
print("(Should be ~1.0 if Poisson assumption holds; >1 indicates overdispersion)")
print()

# Incidence rate ratios (exponentiated coefficients)
irr = np.exp(poisson_model.params)
print("=== Incidence Rate Ratios (exp(coef)) ===")
print(irr)