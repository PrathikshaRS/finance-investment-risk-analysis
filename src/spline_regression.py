import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)
df = pd.read_csv('data/raw/loan_data.csv', index_col=0)

x = df['annual_inc'].values
y = df['loan_amnt'].values

# ---------- Step 1: Decile-binned means to check for genuine nonlinearity ----------
df['income_decile'] = pd.qcut(df['annual_inc'], 10, labels=False)
decile_means = df.groupby('income_decile').agg(
    mean_income=('annual_inc', 'mean'),
    mean_loan_amt=('loan_amnt', 'mean'),
    n=('loan_amnt', 'size')
).reset_index()
print("=== Decile-binned means (income vs loan amount) ===")
print(decile_means)
print()

# ---------- Step 2: Linear model (baseline, from Phase 6) ----------
X_lin = sm.add_constant(df['annual_inc'])
lin_model = sm.OLS(y, X_lin).fit()
lin_pred = lin_model.predict(X_lin)
lin_r2 = r2_score(y, lin_pred)
lin_rmse = np.sqrt(mean_squared_error(y, lin_pred))

# ---------- Step 3: Log-income transform ----------
df['log_income'] = np.log(df['annual_inc'])
X_log = sm.add_constant(df['log_income'])
log_model = sm.OLS(y, X_log).fit()
log_pred = log_model.predict(X_log)
log_r2 = r2_score(y, log_pred)
log_rmse = np.sqrt(mean_squared_error(y, log_pred))

# ---------- Step 4: Natural cubic spline (4 df) ----------
spline_basis = dmatrix("cr(x, df=4)", {"x": df['annual_inc']}, return_type='dataframe')
spline_model = sm.OLS(y, spline_basis).fit()
spline_pred = spline_model.predict(spline_basis)
spline_r2 = r2_score(y, spline_pred)
spline_rmse = np.sqrt(mean_squared_error(y, spline_pred))

print("=== Model comparison: loan_amnt ~ f(annual_inc) ===")
print(f"{'Model':<20}{'R2':<10}{'RMSE':<12}{'AIC':<12}")
print(f"{'Linear':<20}{lin_r2:<10.4f}{lin_rmse:<12.2f}{lin_model.aic:<12.1f}")
print(f"{'Log-income':<20}{log_r2:<10.4f}{log_rmse:<12.2f}{log_model.aic:<12.1f}")
print(f"{'Natural spline(4df)':<20}{spline_r2:<10.4f}{spline_rmse:<12.2f}{spline_model.aic:<12.1f}")