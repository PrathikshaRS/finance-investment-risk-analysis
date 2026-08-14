"""
Phase 5 — Hypothesis Testing
Finance & Investment Risk Analysis Project

Run from the project root as:
    python src/hypothesis_tests.py

Three tests, each stated with H0/H1 BEFORE looking at results:

  1. Two-sample t-test: income, defaulters vs. non-defaulters (two-tailed)
  2. One-tailed t-test: defaulters have LOWER income than non-defaulters
  3. Two-sample t-test: mean daily $ price change, two sectors chosen for
     a pre-existing business reason (not cherry-picked for significance)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
FIG_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

ALPHA = 0.05

# Loads the loan dataset
def load_loan_data() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "loan_data.csv", index_col=0)

# Loads S&P 500 stock price-change data and keeps only stocks, not ETFs
def load_sp500_data():
    prices = pd.read_pickle(DATA_PROCESSED / "sp500_price_changes_long.pkl")
    return prices[prices["is_etf"] == 0]  # stocks only, consistent with Phase 2

# Tests whether the average income differs between defaulters and non-defaulters
def two_sample_income_ttest(loan_data: pd.DataFrame) -> dict:
    """
    H0: mu_default = mu_non_default  (no difference in mean income)
    H1: mu_default != mu_non_default (two-tailed)
    """
    default_income = loan_data.loc[loan_data["outcome"] == "default", "annual_inc"]
    paid_income = loan_data.loc[loan_data["outcome"] == "paid off", "annual_inc"]

    levene_stat, levene_p = stats.levene(default_income, paid_income)
    equal_var = levene_p >= ALPHA
    test_name = "Student's t-test (equal variance)" if equal_var else "Welch's t-test (unequal variance)"

    t_stat, p_value = stats.ttest_ind(default_income, paid_income, equal_var=equal_var)

    result = {
        "test": test_name,
        "group1_mean": default_income.mean(), "group1_n": len(default_income),
        "group2_mean": paid_income.mean(), "group2_n": len(paid_income),
        "levene_stat": levene_stat, "levene_p": levene_p, "equal_var_assumed": equal_var,
        "t_statistic": t_stat, "p_value": p_value,
        "decision": "Reject H0" if p_value < ALPHA else "Fail to reject H0",
    }

    print("=" * 60)
    print("TEST 1: TWO-SAMPLE T-TEST -- INCOME BY DEFAULT STATUS")
    print("=" * 60)
    print("H0: mu_default = mu_non_default")
    print("H1: mu_default != mu_non_default (two-tailed)")
    print(f"alpha = {ALPHA}\n")
    print(f"Levene's test for equal variances: stat={levene_stat:.3f}, p={levene_p:.4g}")
    print(f"  -> {'Equal variance assumption holds' if equal_var else 'Variances differ significantly'}, "
          f"using {test_name}\n")
    print(f"Defaulters   : mean=${result['group1_mean']:,.2f}, n={result['group1_n']:,}")
    print(f"Non-defaults : mean=${result['group2_mean']:,.2f}, n={result['group2_n']:,}")
    print(f"t-statistic  : {t_stat:.4f}")
    print(f"p-value      : {p_value:.4g}")
    print(f"Decision     : {result['decision']} at alpha={ALPHA}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(data=loan_data, x="outcome", y="annual_inc", ax=ax,
                hue="outcome", showfliers=False,
                palette={"default": "#C44E52", "paid off": "#55A868"}, legend=False)
    ax.set_title(f"Income by Outcome (Welch t={t_stat:.2f}, p={p_value:.2e})")
    ax.set_ylabel("Annual Income ($)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ttest_income_by_outcome.png")
    plt.close(fig)

    pd.Series(result).to_csv(RESULTS_DIR / "two_sample_ttest_income.csv")
    return result

# Tests whether defaulters have significantly lower average income than non-defaulters
def one_tailed_income_ttest(loan_data: pd.DataFrame, equal_var: bool) -> dict:
    """
    H0: mu_default >= mu_non_default (defaulters do NOT have lower income)
    H1: mu_default <  mu_non_default (defaulters have lower income)

    Justification stated BEFORE testing: Phase 3's descriptive comparison
    and the credit-risk literature (Section 2.1, Altman-style scorecards)
    both give a specific directional expectation -- income predicts
    REPAYMENT ability -- so a one-tailed test is justified a priori,
    not adopted after seeing the two-tailed result to gain power.
    """
    default_income = loan_data.loc[loan_data["outcome"] == "default", "annual_inc"]
    paid_income = loan_data.loc[loan_data["outcome"] == "paid off", "annual_inc"]

    t_stat, p_two_tailed = stats.ttest_ind(default_income, paid_income, equal_var=equal_var)
    correct_direction = default_income.mean() < paid_income.mean()
    p_one_tailed = (p_two_tailed / 2) if correct_direction else (1 - p_two_tailed / 2)

    result = {
        "group1_mean_default": default_income.mean(),
        "group2_mean_paid_off": paid_income.mean(),
        "t_statistic": t_stat,
        "p_value_one_tailed": p_one_tailed,
        "direction_matches_h1": correct_direction,
        "decision": "Reject H0" if p_one_tailed < ALPHA else "Fail to reject H0",
    }

    print("\n" + "=" * 60)
    print("TEST 2: ONE-TAILED T-TEST -- DEFAULTERS HAVE LOWER INCOME")
    print("=" * 60)
    print("H0: mu_default >= mu_non_default")
    print("H1: mu_default <  mu_non_default")
    print(f"alpha = {ALPHA}\n")
    print(f"Sample difference is in the H1 direction: {correct_direction}")
    print(f"t-statistic       : {t_stat:.4f}")
    print(f"one-tailed p-value: {p_one_tailed:.4g}")
    print(f"Decision          : {result['decision']} at alpha={ALPHA}")

    pd.Series(result).to_csv(RESULTS_DIR / "one_tailed_ttest_income.csv")
    return result

# Tests whether two selected sectors have different average daily price changes
def sector_returns_ttest(prices: pd.DataFrame, sector_a: str = "Consumer Staples",
                          sector_b: str = "Technology") -> dict:
    """
    H0: mu_sector_a = mu_sector_b (no difference in mean daily $ price change)
    H1: mu_sector_a != mu_sector_b (two-tailed)

    Sector choice justified BEFORE testing: Phase 2 identified Consumer
    Staples as the lowest-std sector and Technology as one of the
    higher-std sectors by raw dollar price change -- a natural,
    pre-existing "defensive vs. growth" pairing, not cherry-picked here
    for a significant result.
    """
    a = prices.loc[prices["sector_label"] == sector_a, "price_change"]
    b = prices.loc[prices["sector_label"] == sector_b, "price_change"]

    levene_stat, levene_p = stats.levene(a, b)
    equal_var = levene_p >= ALPHA
    test_name = "Student's t-test (equal variance)" if equal_var else "Welch's t-test (unequal variance)"

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=equal_var)

    result = {
        "sector_a": sector_a, "sector_a_mean": a.mean(), "sector_a_n": len(a),
        "sector_b": sector_b, "sector_b_mean": b.mean(), "sector_b_n": len(b),
        "test": test_name, "levene_p": levene_p, "equal_var_assumed": equal_var,
        "t_statistic": t_stat, "p_value": p_value,
        "decision": "Reject H0" if p_value < ALPHA else "Fail to reject H0",
    }

    print("\n" + "=" * 60)
    print(f"TEST 3: TWO-SAMPLE T-TEST -- {sector_a} vs {sector_b} DAILY PRICE CHANGE")
    print("=" * 60)
    print(f"H0: mu_{sector_a} = mu_{sector_b}")
    print(f"H1: mu_{sector_a} != mu_{sector_b} (two-tailed)")
    print(f"alpha = {ALPHA}\n")
    print(f"Levene's test for equal variances: stat={levene_stat:.3f}, p={levene_p:.4g}")
    print(f"  -> using {test_name}\n")
    print(f"{sector_a}: mean=${result['sector_a_mean']:.4f}, n={result['sector_a_n']:,}")
    print(f"{sector_b}: mean=${result['sector_b_mean']:.4f}, n={result['sector_b_n']:,}")
    print(f"t-statistic : {t_stat:.4f}")
    print(f"p-value     : {p_value:.4g}")
    print(f"Decision    : {result['decision']} at alpha={ALPHA}")
    print("NOTE: statistically significant does not necessarily mean practically")
    print("large here -- both are raw dollar price changes near zero on average;")
    print("see the effect-size magnitude ($ difference) alongside the p-value.")

    fig, ax = plt.subplots(figsize=(6, 5))
    plot_data = prices[prices["sector_label"].isin([sector_a, sector_b])].copy()
    plot_data["sector_label"] = plot_data["sector_label"].astype(str)  # drop unused category levels
    sns.boxplot(data=plot_data, x="sector_label", y="price_change", ax=ax, showfliers=False,
                hue="sector_label", legend=False)
    ax.set_title(f"Daily Price Change: {sector_a} vs {sector_b}\n(t={t_stat:.2f}, p={p_value:.2e})")
    ax.set_ylabel("Daily Price Change ($)")
    ax.set_xlabel("")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "ttest_sector_returns.png")
    plt.close(fig)

    pd.Series(result).to_csv(RESULTS_DIR / "sector_returns_ttest.csv")
    return result


if __name__ == "__main__":
    loan_data = load_loan_data()
    prices = load_sp500_data()

    res1 = two_sample_income_ttest(loan_data)
    res2 = one_tailed_income_ttest(loan_data, equal_var=res1["equal_var_assumed"])
    res3 = sector_returns_ttest(prices)

    print("\n" + "=" * 60)
    print("PHASE 5 HYPOTHESIS TESTING COMPLETE")
    print("=" * 60)