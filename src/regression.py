import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score

np.random.seed(42)
df = pd.read_csv('data/raw/loan_data.csv', index_col=0)

# Linear Regression: loan_amnt ~ annual_inc
X_lin = sm.add_constant(df['annual_inc'])
y_lin = df['loan_amnt']
lin_model = sm.OLS(y_lin, X_lin).fit()
print(lin_model.summary())

# Logistic Regression: default ~ income + loan_amnt + dti + revol_util + borrower_score
df['default_flag'] = (df['outcome'] == 'default').astype(int)
features = ['annual_inc', 'loan_amnt', 'dti', 'revol_util', 'borrower_score']
X, y = df[features], df['default_flag']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

X_train_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_train_sm).fit(disp=0)
print(logit_model.summary2())
print(f"McFadden pseudo R²: {logit_model.prsquared:.4f}")

X_test_sm = sm.add_constant(X_test)
y_pred_prob = logit_model.predict(X_test_sm)
y_pred = (y_pred_prob >= 0.5).astype(int)

print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_prob):.4f}")