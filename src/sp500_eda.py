"""
An ETF is a fund that can hold a collection of stocks rather than representing one company.

Observations per sector_label:
This tells you how much data you have for each sector. 

Financials has the most observations because it has many stocks and/or longer histories. Telecom has fewer observations because it has fewer stocks.
This does not necessarily mean Financials performed better. It only tells us, How many stock-day records belong to each sector.

Sector composition:
This tells you how many different companies belong to each sector.

For example:
    Financials
    88 out of 500 stocks are Financial companies.
    88 / 500 × 100 = 17.6%
So Financials makes up 17.6% of the regular stocks in your dataset. Why this matters? This tells us whether the dataset is heavily concentrated in certain sectors.
Your biggest sectors are:
Financials → Consumer Discretionary → Industrials → Technology

Daily price-change distribution (stocks only, in $):
Your data represents -> How much a stock's price changed from one day to the next, in dollars?
    Mean = $0.0015
    On average, the daily price change is very close to $0. That makes sense because some days prices go up and some days they go down. They roughly balance each other out.
    Median = $0
    This means that the middle observation is $0. In simple words -> A large number of stock-day observations had no change in price.
    Standard deviation = $1.35
    This tells us how spread out the daily price changes are around the average.
    Roughly speaking, the typical variation is around $1.35, although this is influenced heavily by extreme observations.

Think of percentiles as cut-off points.
    25th percentile = -$0.185 -> 25% of observations are below approximately -$0.185.
    50th percentile = $0 -> This is the median. 50% are below this and 50% are above it.
    75th percentile = $0.20 -> 75% of observations are below approximately +$0.20.
    95th percentile = $0.93 -> Only about 5% of observations are above $0.93.
    99th percentile = $2.29 -> Only about 1% of observations are above $2.29.
This helps us understand the normal range and identify unusually large movements.

IQR outlier analysis (daily $ price change, stocks only):
This means the middle 50% of Utilities observations lie approximately between -$0.108 and +$0.123

Sector-level daily $ price-change statistics:
A larger IQR means the middle 50% of daily price changes are more spread out. Again, because these are dollar changes, we should interpret this cautiously across sectors

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------
DATA_DIR = Path("data/processed")
FIG_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------
def load_processed_data():
    """Load the Phase-1 processed price-change and sector-mapping data.
    Loaded once; reused by every function below (no repeat disk reads,
    important given limited hardware).
    """
    prices = pd.read_pickle(DATA_DIR/ "sp500_price_changes_long.pkl")
    sectors = pd.read_pickle(DATA_DIR / "sp500_sectors_clean.pkl")
    return prices, sectors


# ---------------------------------------------------------------------
# 2. Dataset overview
# ---------------------------------------------------------------------
def summarize_dataset(prices: pd.DataFrame, sectors: pd.DataFrame) -> dict:
    """Print and return a concise dataset overview."""
    n_etf_tickers = sectors["is_etf"].sum()
    n_stock_tickers = sectors["is_etf"].eq(0).sum()

    summary = {
        "n_observations": len(prices),
        "n_unique_tickers": prices["symbol"].nunique(),
        "n_stock_tickers": int(n_stock_tickers),
        "n_etf_tickers": int(n_etf_tickers),
        "n_sectors_incl_etf_bucket": prices["sector_label"].nunique(),
        "date_min": prices["date"].min(),
        "date_max": prices["date"].max(),
    }

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Observations (ticker-days) : {summary['n_observations']:,}")
    print(f"Unique tickers              : {summary['n_unique_tickers']}")
    print(f"  - Regular stocks          : {summary['n_stock_tickers']}")
    print(f"  - ETFs                    : {summary['n_etf_tickers']}")
    print(f"Sector labels (incl. 'EFTs'): {summary['n_sectors_incl_etf_bucket']}")
    print(f"Date range                  : {summary['date_min'].date()} to {summary['date_max'].date()}")
    print("\nObservations per sector_label:")
    print(prices["sector_label"].value_counts())

    pd.Series(summary).to_csv(RESULTS_DIR / "dataset_overview.csv")
    return summary


# ---------------------------------------------------------------------
# 3. Sector composition
# ---------------------------------------------------------------------
def analyze_sector_distribution(sectors: pd.DataFrame) -> pd.DataFrame:
    """Count of tickers per sector (real GICS sectors only, ETFs excluded
    since 'EFTs' is a fund wrapper, not a GICS sector, and would distort
    a 'stocks per sector' comparison)."""
    stocks_only = sectors[sectors["is_etf"] == 0]
    counts = (
        stocks_only["sector_label"]
        .value_counts()
        .rename_axis("sector_label")
        .reset_index(name="n_tickers")
        .sort_values("n_tickers", ascending=False)
    )
    counts["pct"] = (counts["n_tickers"] / counts["n_tickers"].sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=counts, y="sector_label", x="n_tickers", ax=ax, color="#4C72B0")
    ax.set_title("Number of S&P 500 Stocks per Sector (ETFs excluded)")
    ax.set_xlabel("Number of Tickers")
    ax.set_ylabel("Sector")
    for i, v in enumerate(counts["n_tickers"]):
        ax.text(v + 2, i, str(v), va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sector_stock_counts.png")
    plt.close(fig)

    counts.to_csv(RESULTS_DIR / "sector_stock_counts.csv", index=False)
    print("\nSector composition:")
    print(counts.to_string(index=False))
    return counts


# ---------------------------------------------------------------------
# 4. Price-change distribution (overall)
# ---------------------------------------------------------------------
def analyze_price_changes(prices: pd.DataFrame) -> pd.Series:
    """Descriptive stats of daily dollar price changes, stocks only
    (ETFs excluded so fund-level price changes don't mix with individual
    stock price changes in the same distribution)."""
    stock_changes = prices.loc[prices["is_etf"] == 0, "price_change"]

    desc = stock_changes.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print("\nDaily price-change distribution (stocks only, in $):")
    print(desc)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    lo, hi = stock_changes.quantile([0.005, 0.995])
    sns.histplot(stock_changes[(stock_changes >= lo) & (stock_changes <= hi)],
                 bins=100, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Distribution of Daily Price Changes\n(middle 99% shown for readability)")
    axes[0].set_xlabel("Daily Price Change ($)")

    sns.boxplot(x=stock_changes, ax=axes[1], color="#55A868", fliersize=1)
    axes[1].set_title("Boxplot of Daily Price Changes (full data)")
    axes[1].set_xlabel("Daily Price Change ($)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_change_distribution.png")
    plt.close(fig)

    desc.to_csv(RESULTS_DIR / "price_change_distribution_stats.csv")
    return desc


# ---------------------------------------------------------------------
# 5. IQR outlier detection
# ---------------------------------------------------------------------
def detect_iqr_outliers(prices: pd.DataFrame) -> dict:
    """Tukey's-fence IQR outlier detection on daily dollar price changes
    (stocks only)."""
    stock_changes = prices.loc[prices["is_etf"] == 0, "price_change"]

    q1, q3 = stock_changes.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    is_outlier = (stock_changes < lower) | (stock_changes > upper)
    n_outliers = int(is_outlier.sum())
    pct_outliers = n_outliers / len(stock_changes) * 100

    result = {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "lower_fence": lower, "upper_fence": upper,
        "n_outliers": n_outliers, "pct_outliers": pct_outliers,
        "n_total": len(stock_changes),
    }

    print("\nIQR outlier analysis (daily $ price change, stocks only):")
    for k, v in result.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v}")

    fig, ax = plt.subplots(figsize=(8, 3.5))
    sns.boxplot(x=stock_changes, ax=ax, color="#C44E52", fliersize=1)
    ax.axvline(lower, color="black", linestyle="--", linewidth=1, label=f"Lower fence ({lower:.2f})")
    ax.axvline(upper, color="black", linestyle="--", linewidth=1, label=f"Upper fence ({upper:.2f})")
    ax.set_title("IQR Outlier Fences — Daily Price Change ($)")
    ax.set_xlabel("Daily Price Change ($)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "price_change_iqr_outliers.png")
    plt.close(fig)

    pd.Series(result).to_csv(RESULTS_DIR / "iqr_outliers.csv")
    return result


# ---------------------------------------------------------------------
# 6. Sector-level statistics
# ---------------------------------------------------------------------
def analyze_sector_statistics(prices: pd.DataFrame) -> pd.DataFrame:
    """Per-sector descriptive stats of daily $ price change (ETFs excluded)."""
    stock_prices = prices[prices["is_etf"] == 0]

    def q1(x): return x.quantile(0.25)
    def q3(x): return x.quantile(0.75)

    stats = (
        stock_prices.groupby("sector_label", observed=True)["price_change"]
        .agg(mean="mean", median="median", std="std", q1=q1, q3=q3, n_obs="count")
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats = stats.sort_values("std", ascending=False).round(4)

    print("\nSector-level daily $ price-change statistics:")
    print(stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 6))
    order = stock_prices.groupby("sector_label", observed=True)["price_change"].median().sort_values().index
    sns.boxplot(data=stock_prices, y="sector_label", x="price_change", order=order, ax=ax, showfliers=False)
    ax.set_title("Daily Price Change by Sector (outliers hidden for readability)")
    ax.set_xlabel("Daily Price Change ($)")
    ax.set_ylabel("Sector")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sector_price_change_boxplot.png")
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    stats_sorted = stats.sort_values("mean")
    colors = ["#C44E52" if v < 0 else "#55A868" for v in stats_sorted["mean"]]
    ax2.barh(stats_sorted.index, stats_sorted["mean"], color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.set_title("Mean Daily Price Change by Sector ($)")
    ax2.set_xlabel("Mean Daily Price Change ($)")
    fig2.tight_layout()
    fig2.savefig(FIG_DIR / "sector_mean_price_change.png")
    plt.close(fig2)

    stats.to_csv(RESULTS_DIR / "sector_statistics.csv")
    return stats


# ---------------------------------------------------------------------
# 7. Correlation analysis (representative subset)
# ---------------------------------------------------------------------
def create_correlation_analysis(prices: pd.DataFrame, n_per_sector: int = 2,
                                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Correlation of daily price changes across a REPRESENTATIVE SUBSET of
    stocks, not all 500 (unreadable heatmap + heavy pivot on limited hardware).

    Subset selection: from each real GICS sector (ETFs excluded), pick the
    `n_per_sector` tickers with the most complete date coverage in this
    dataset, so the sample is both sector-diverse and has enough
    overlapping history to compute a meaningful correlation.
    """
    stock_prices = prices[prices["is_etf"] == 0]

    coverage = stock_prices.groupby("symbol", observed=True)["date"].count()
    sector_map = stock_prices.drop_duplicates("symbol").set_index("symbol")["sector_label"]

    chosen = []
    for sector, grp in sector_map.groupby(sector_map):
        tickers = grp.index
        top = coverage.loc[tickers].sort_values(ascending=False).head(n_per_sector).index.tolist()
        chosen.extend(top)

    print(f"\nRepresentative subset for correlation ({len(chosen)} tickers, "
          f"{n_per_sector} per sector, chosen by longest available history):")
    print(sorted(chosen))

    subset = stock_prices[stock_prices["symbol"].isin(chosen)]
    pivot = subset.pivot_table(index="date", columns="symbol", values="price_change")
    corr = pivot.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, ax=ax, square=True, cbar_kws={"label": "Pearson r"})
    ax.set_title("Correlation of Daily Price Changes\n(representative subset, 2 stocks/sector)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap_subset.png")
    plt.close(fig)

    corr.to_csv(RESULTS_DIR / "correlation_matrix_subset.csv")
    return corr


# ---------------------------------------------------------------------
# 8. Pair plot (small subset)
# ---------------------------------------------------------------------
def create_pairplot(prices: pd.DataFrame, tickers: list = None) -> None:
    """
    Pair plot of daily price changes for a small (4-6 ticker) subset.
    Kept intentionally small: cost grows roughly with the square of the
    number of variables, and beyond ~6 series it gets slow/unreadable
    on limited hardware anyway.

    Default selection: the single largest (by trading-day coverage)
    stock from each of Technology, Financials, Energy, Health Care, and
    Consumer Staples — sector-diverse, so the plot speaks to
    cross-sector co-movement, matching the correlation analysis above.
    """
    stock_prices = prices[prices["is_etf"] == 0]

    if tickers is None:
        target_sectors = ["Technology", "Financials", "Energy", "Health Care", "Consumer Staples"]
        coverage = stock_prices.groupby(["sector_label", "symbol"], observed=True)["date"].count()
        tickers = [coverage.loc[s].idxmax() for s in target_sectors]

    print(f"\nPair plot tickers (largest-coverage stock per target sector): {tickers}")

    subset = stock_prices[stock_prices["symbol"].isin(tickers)]
    pivot = subset.pivot_table(index="date", columns="symbol", values="price_change").dropna()

    g = sns.pairplot(pivot, diag_kind="hist", plot_kws={"alpha": 0.3, "s": 10})
    g.fig.suptitle("Pair Plot — Daily Price Changes, Representative Stocks", y=1.02)
    g.savefig(FIG_DIR / "pairplot_representative_stocks.png")
    plt.close(g.fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    prices, sectors = load_processed_data()
    summarize_dataset(prices, sectors)
    analyze_sector_distribution(sectors)
    analyze_price_changes(prices)
    detect_iqr_outliers(prices)
    analyze_sector_statistics(prices)
    create_correlation_analysis(prices)
    create_pairplot(prices)