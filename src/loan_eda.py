"""
Phase 3 — Loan Dataset EDA: Income vs. Default Risk
Finance & Investment Risk Analysis Project

Run from the project root as:
    python src/loan_eda.py

Uses loan_data.csv (the larger, 45,342-row dataset) as the primary
source for income-vs-default analysis, since it is the only one of the
two loan files that contains an income column.

NOTE ON loan3000.csv: inspected but NOT used for income analysis in
this phase. It contains only `outcome`, `purpose_`, `dti`,
`borrower_score`, and `payment_inc_ratio` — no `annual_inc` or
`loan_amnt` column at all. It remains useful as a smaller illustrative
subset for dti/borrower_score work in later phases (hypothesis
testing, regression), but cannot answer an income-vs-default question
because it simply does not carry income.

NOTE ON THE OUTCOME VARIABLE: `status` has three raw values (Fully
Paid / Charged Off / Default), collapsed into a binary `outcome`
(paid off / default). The resulting sample is an exact 50/50 split
(22,671 / 22,671) — this is almost certainly a deliberately balanced
(undersampled) dataset for teaching/modelling purposes, NOT the true
population default rate. This is flagged explicitly wherever "default
rate" is discussed, so it isn't mistaken for a real-world base rate.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


DATA_DIR = Path("data/raw")
FIG_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

#Loads the loan data
def load_loan_data():
    """Load both loan files. loan_data is the primary analysis dataset
    (has income); loan3000 is loaded for reference/later phases only.
    Both are already clean on inspection: 0 missing values, 0 exact
    duplicate rows, no impossible negative income/loan amounts."""
    loan_data = pd.read_csv(DATA_DIR / "loan_data.csv", index_col=0)
    loan3000 = pd.read_csv(DATA_DIR / "loan3000.csv", index_col=0)
    return loan_data, loan3000


# Gives an overview of the loan datasets and checks data quality
def summarize_loan_dataset(loan_data: pd.DataFrame, loan3000: pd.DataFrame) -> dict:
    """Print and return a concise overview of both loan datasets."""
    print("=" * 60)
    print("LOAN DATASET OVERVIEW")
    print("=" * 60)
    print(f"loan_data.csv : {loan_data.shape[0]:,} rows x {loan_data.shape[1]} cols")
    print(f"loan3000.csv  : {loan3000.shape[0]:,} rows x {loan3000.shape[1]} cols "
          f"(no income column -> reference only in this phase)")

    print(f"\nMissing values in loan_data : {loan_data.isnull().sum().sum()}")
    print(f"Duplicate rows in loan_data  : {loan_data.duplicated().sum()}")

    outcome_counts = loan_data["outcome"].value_counts()
    print(f"\nOutcome balance in loan_data:\n{outcome_counts}")
    print("NOTE: exact 50/50 split -> this is a balanced (likely undersampled)")
    print("dataset, not the true population default rate. Any 'X% default rate'")
    print("statement about the real world should NOT be based on this split.")

    summary = {
        "loan_data_rows": loan_data.shape[0],
        "loan3000_rows": loan3000.shape[0],
        "missing_values": int(loan_data.isnull().sum().sum()),
        "duplicate_rows": int(loan_data.duplicated().sum()),
        "n_default": int(outcome_counts.get("default", 0)),
        "n_paid_off": int(outcome_counts.get("paid off", 0)),
    }
    pd.Series(summary).to_csv(RESULTS_DIR / "loan_dataset_overview.csv")
    return summary

# Analyzes the distribution of borrowers' annual income
def analyze_income_distribution(loan_data: pd.DataFrame) -> pd.Series:
    """Descriptive stats + distribution plot of annual_inc."""
    income = loan_data["annual_inc"]
    desc = income.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\nAnnual income distribution ($):")
    print(desc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    cap = income.quantile(0.99)
    sns.histplot(income[income <= cap], bins=60, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distribution of Annual Income\n(up to 99th percentile, for readability)")
    axes[0].set_xlabel("Annual Income ($)")

    sns.boxplot(x=income, ax=axes[1], color="#55A868", fliersize=2)
    axes[1].set_title("Boxplot of Annual Income (full data)")
    axes[1].set_xlabel("Annual Income ($)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "income_distribution.png")
    plt.close(fig)

    desc.to_csv(RESULTS_DIR / "income_distribution_stats.csv")
    return desc

# Counts and compares defaulted and paid-off loans
def analyze_default_distribution(loan_data: pd.DataFrame) -> pd.Series:
    """Bar chart of default vs paid-off counts."""
    counts = loan_data["outcome"].value_counts()
    pct = (counts / counts.sum() * 100).round(1)

    print("\nDefault vs non-default counts:")
    for k in counts.index:
        print(f"  {k}: {counts[k]:,} ({pct[k]}%)")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    colors = {"default": "#C44E52", "paid off": "#55A868"}
    sns.barplot(x=counts.index, y=counts.values, ax=ax,
                hue=counts.index, palette=colors, legend=False)
    ax.set_title("Loan Outcome Counts (balanced sample)")
    ax.set_ylabel("Number of Loans")
    ax.set_xlabel("")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v:,}", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "default_outcome_counts.png")
    plt.close(fig)

    counts.to_csv(RESULTS_DIR / "default_outcome_counts.csv")
    return counts

# Compares borrowers' income between defaulters and paid-off borrowers
def analyze_income_vs_default(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Compare income distributions between defaulters and non-defaulters."""
    grp_stats = loan_data.groupby("outcome")["annual_inc"].describe()
    print("\nIncome by outcome:")
    print(grp_stats)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.boxplot(data=loan_data, x="outcome", y="annual_inc", ax=axes[0],
                hue="outcome", showfliers=False,
                palette={"default": "#C44E52", "paid off": "#55A868"}, legend=False)
    axes[0].set_title("Income by Outcome\n(outliers hidden for readability)")
    axes[0].set_ylabel("Annual Income ($)")
    axes[0].set_xlabel("")

    cap = loan_data["annual_inc"].quantile(0.99)
    trimmed = loan_data[loan_data["annual_inc"] <= cap]
    sns.kdeplot(data=trimmed, x="annual_inc", hue="outcome", fill=True,
                alpha=0.4, ax=axes[1], common_norm=False,
                palette={"default": "#C44E52", "paid off": "#55A868"})
    axes[1].set_title("Income Density by Outcome\n(<=99th pct, for readability)")
    axes[1].set_xlabel("Annual Income ($)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "income_vs_default.png")
    plt.close(fig)

    grp_stats.to_csv(RESULTS_DIR / "income_by_outcome_stats.csv")
    return grp_stats

# Detects unusually high or low income values using the IQR method
def detect_income_outliers(loan_data: pd.DataFrame) -> dict:
    """Tukey's-fence IQR outlier detection on annual_inc."""
    income = loan_data["annual_inc"]
    q1, q3 = income.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    is_outlier = (income < lower) | (income > upper)
    n_outliers = int(is_outlier.sum())
    pct_outliers = n_outliers / len(income) * 100

    result = {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": max(lower, 0),  # income can't be negative
        "upper_fence": upper,
        "n_outliers": n_outliers, "pct_outliers": pct_outliers,
        "n_total": len(income),
    }

    print("\nIQR outlier analysis (annual income):")
    for k, v in result.items():
        print(f"  {k}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
    print("\nDECISION: high-income outliers are RETAINED, not removed or capped.")
    print("They are almost certainly genuine high earners, and income is a")
    print("candidate PREDICTOR for later regression/hypothesis testing —")
    print("silently deleting real high-income borrowers would bias that")
    print("analysis rather than clean it. Any single extreme point (e.g. the")
    print("$7.14M row) should be sanity-checked individually before it is")
    print("allowed to drive a regression fit.")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.boxplot(x=income, ax=ax, color="#C44E52", fliersize=2)
    ax.axvline(upper, color="black", linestyle="--", linewidth=1, label=f"Upper fence (${upper:,.0f})")
    ax.set_title("IQR Outlier Fence — Annual Income")
    ax.set_xlabel("Annual Income ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "income_iqr_outliers.png")
    plt.close(fig)

    pd.Series(result).to_csv(RESULTS_DIR / "income_iqr_outliers.csv")
    return result

# Finds relationships between important loan and risk variables
def analyze_loan_correlations(loan_data: pd.DataFrame) -> pd.DataFrame:
    """Correlation matrix of key numeric risk variables."""
    numeric_cols = ["annual_inc", "loan_amnt", "dti", "payment_inc_ratio",
                     "revol_bal", "revol_util", "grade", "borrower_score"]
    corr = loan_data[numeric_cols].corr(method="pearson")

    print("\nCorrelation matrix (key numeric variables):")
    print(corr.round(2))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation Matrix — Loan Risk Variables")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "loan_correlation_heatmap.png")
    plt.close(fig)

    corr.to_csv(RESULTS_DIR / "loan_correlation_matrix.csv")
    return corr

# Creates a plot to visually compare important loan variables
def create_loan_pairplot(loan_data: pd.DataFrame, sample_n: int = 2000) -> None:
    """
    Pair plot of income, loan amount, DTI, and borrower score, coloured
    by outcome. A random sample (default 2,000 of 45,342 rows) is used
    to keep the plot fast to render and readable.
    """
    cols = ["annual_inc", "loan_amnt", "dti", "borrower_score", "outcome"]
    sample = loan_data[cols].sample(n=min(sample_n, len(loan_data)), random_state=RANDOM_SEED)

    cap = loan_data["annual_inc"].quantile(0.99)
    sample = sample[sample["annual_inc"] <= cap]

    print(f"\nPair plot: {len(sample)}-row random sample, variables = "
          f"{[c for c in cols if c != 'outcome']}, coloured by outcome")

    g = sns.pairplot(sample, hue="outcome", diag_kind="hist",
                      palette={"default": "#C44E52", "paid off": "#55A868"},
                      plot_kws={"alpha": 0.4, "s": 12})
    g.fig.suptitle("Pair Plot — Income, Loan Amount, DTI, Borrower Score", y=1.02)
    g.savefig(FIG_DIR / "loan_pairplot.png")
    plt.close(g.fig)


if __name__ == "__main__":
    loan_data, loan3000 = load_loan_data()
    summarize_loan_dataset(loan_data, loan3000)
    analyze_income_distribution(loan_data)
    analyze_default_distribution(loan_data)
    analyze_income_vs_default(loan_data)
    detect_income_outliers(loan_data)
    analyze_loan_correlations(loan_data)
    create_loan_pairplot(loan_data)