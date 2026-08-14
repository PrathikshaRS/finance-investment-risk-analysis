"""
Phase 4 — Central Limit Theorem Demonstration
Finance & Investment Risk Analysis Project

Run from the project root as:
    python src/clt_demo.py

Demonstrates the CLT using annual_inc from loan_data.csv (45,342 rows).
Income was already confirmed right-skewed in Phase 3 (mean $68,212 >
median $60,000), which makes it a good candidate for this demo: the
whole point of the CLT is that the SAMPLING DISTRIBUTION OF THE MEAN
approaches normality even though the raw variable itself is skewed.

IMPORTANT — what this script does NOT claim:
  - It does NOT claim raw income itself becomes normal. It stays skewed
    at every sample size; only the distribution of repeated SAMPLE
    MEANS approaches normal as n grows.
  - Sampling here is WITHOUT replacement within each individual sample
    (drawing n distinct borrowers), the standard textbook CLT demo;
    "with replacement across repetitions" in the report brief refers to
    reusing the same population pool for each of the 1,000 independent
    draws, not duplicate rows within one draw.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

DATA_DIR = Path("data/raw")
FIG_DIR = Path("outputs/figures")
RESULTS_DIR = Path("outputs/results")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

RANDOM_SEED = 42
SAMPLE_SIZES = [10, 30, 50, 100]
N_REPETITIONS = 1000

# Loads the annual income data from the loan dataset
def load_data() -> pd.Series:
    """Load annual_inc from loan_data.csv as the population for the CLT demo."""
    loan_data = pd.read_csv(DATA_DIR / "loan_data.csv", index_col=0)
    return loan_data["annual_inc"]

# Calculates the population mean, standard deviation, and skewness
def population_stats(income: pd.Series) -> dict:
    """Population mean, std, and skewness of raw income."""
    mu = income.mean()
    sigma = income.std()
    skew = stats.skew(income)

    print("=" * 60)
    print("POPULATION (raw income) STATISTICS")
    print("=" * 60)
    print(f"Population mean (mu)     : ${mu:,.2f}")
    print(f"Population std (sigma)   : ${sigma:,.2f}")
    print(f"Population skewness      : {skew:.3f}  (0 = symmetric; "
          f"positive = right-skewed, matches Phase 3 finding)")

    result = {"mu": mu, "sigma": sigma, "skewness": skew, "n_population": len(income)}
    pd.Series(result).to_csv(RESULTS_DIR / "clt_population_stats.csv")
    return result

# Generates random samples and calculates the mean of each sample
def simulate_sampling_distribution(income: pd.Series, n: int,
                                    reps: int = N_REPETITIONS,
                                    seed: int = RANDOM_SEED) -> np.ndarray:
    """Draw `reps` independent random samples of size n (without
    replacement within each sample) from `income`; return the array of
    `reps` sample means."""
    rng = np.random.RandomState(seed)
    values = income.values
    means = np.empty(reps)
    for i in range(reps):
        sample = rng.choice(values, size=n, replace=False)
        means[i] = sample.mean()
    return means

# Runs the CLT simulation for different sample sizes and summarizes the results
def run_clt_simulation(income: pd.Series, sample_sizes: list = SAMPLE_SIZES,
                        reps: int = N_REPETITIONS) -> pd.DataFrame:
    """Run the simulation for every sample size and build the summary table."""
    mu = income.mean()
    sigma = income.std()

    # Track how often an extreme high-income borrower (>$300k, ~99.7th
    # population percentile) lands in a given sample -- this is what
    # actually drives the occasional very-high sample mean / long right
    # tail in the sampling distribution, worth quantifying explicitly.
    extreme_threshold = 300_000
    values = income.values

    rows = []
    results = {}
    for n in sample_sizes:
        means = simulate_sampling_distribution(income, n, reps)
        results[n] = means
        theoretical_se = sigma / np.sqrt(n)
        empirical_se = means.std()

        rng_check = np.random.RandomState(RANDOM_SEED)
        n_reps_with_extreme = sum(
            (rng_check.choice(values, size=n, replace=False) > extreme_threshold).any()
            for _ in range(reps)
        )

        rows.append({
            "n": n,
            "reps": reps,
            "mean_of_sample_means": means.mean(),
            "theoretical_se_sigma_over_sqrt_n": theoretical_se,
            "empirical_se_of_sample_means": empirical_se,
            "skewness_of_sampling_dist": stats.skew(means),
            "pct_reps_containing_gt300k_earner": n_reps_with_extreme / reps * 100,
        })

    summary = pd.DataFrame(rows).round(3)
    print("\nCLT SIMULATION SUMMARY")
    print(f"(population mu=${mu:,.2f}, sigma=${sigma:,.2f}, {reps} repetitions per n)\n")
    print(summary.to_string(index=False))
    print("\nInterpretation:")
    print("  - mean_of_sample_means stays close to population mu at every n,")
    print("    consistent with the sample mean being an unbiased estimator.")
    print("  - empirical_se_of_sample_means tracks theoretical sigma/sqrt(n)")
    print("    reasonably and shrinks as n grows, as CLT predicts.")
    print("  - Population skewness is extreme (~47), driven mostly by a")
    print("    handful of very high earners (161 borrowers >$300k out of")
    print("    45,342, incl. one $7.14M outlier). pct_reps_containing_gt300k_earner")
    print("    shows that even at n=100, roughly a quarter to a third of the")
    print("    1000 simulated samples happen to include at least one such")
    print("    earner -- pulling that sample's mean noticeably higher and")
    print("    producing the long right tail visible in the histograms.")
    print("    This is WHY skewness_of_sampling_dist doesn't fall smoothly")
    print("    to 0 by n=30 the way the 'n=30 rule of thumb' suggests --")
    print("    that folklore assumes a population without such heavy tails.")
    print("  - The core CLT claim still holds -- the sampling distribution's")
    print("    bulk visibly tightens and centers on mu as n grows -- it just")
    print("    converges more slowly here than 'n=30' folklore suggests.")

    summary.to_csv(RESULTS_DIR / "clt_simulation_summary.csv", index=False)
    return summary, results

# Creates graphs showing how the sampling distribution approaches normality
def plot_clt_grid(income: pd.Series, sampling_results: dict, mu: float, sigma: float) -> None:
    """2x3 grid: raw population distribution + sampling distribution of
    the mean at each sample size, each with a fitted normal curve overlay."""
    sample_sizes = list(sampling_results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    cap = income.quantile(0.99)
    sns.histplot(income[income <= cap], bins=60, stat="density", ax=axes[0], color="#4C72B0")
    axes[0].set_title(f"Population: Raw Income\n(skewed, n={len(income):,})")
    axes[0].set_xlabel("Annual Income ($)")

    # x-axis clipped to the 1st-99th percentile of the SIMULATED MEANS
    # (not raw data) so the bulk shape is visible -- rare samples that
    # include a >$300k earner produce very-high means that would
    # otherwise stretch the axis into an unreadable sliver. Full range
    # is still used for every statistic in the summary table.
    for i, n in enumerate(sample_sizes, start=1):
        means = sampling_results[n]
        ax = axes[i]

        lo, hi = np.percentile(means, [1, 99])
        clipped = means[(means >= lo) & (means <= hi)]
        n_clipped = len(means) - len(clipped)

        sns.histplot(clipped, bins=40, stat="density", ax=ax, color="#55A868")

        x = np.linspace(lo, hi, 200)
        theoretical_se = sigma / np.sqrt(n)
        ax.plot(x, stats.norm.pdf(x, mu, theoretical_se), color="black",
                linewidth=1.5, label="Theoretical N(mu, sigma^2/n)")
        ax.set_title(f"Sampling Distribution of Mean\n(n={n}, 1000 reps, middle 98% shown)")
        ax.set_xlabel("Sample Mean ($)")
        ax.legend(fontsize=7)
        ax.text(0.98, 0.02, f"{n_clipped} extreme reps clipped\nfor display only",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5, style="italic")

    if len(sample_sizes) < 5:
        axes[5].axis("off")

    fig.suptitle("Central Limit Theorem: Sampling Distribution of Mean Income "
                  "Converging to Normal as n Increases", y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clt_sampling_distributions.png", bbox_inches="tight")
    plt.close(fig)

# Plots how the theoretical and actual standard error decrease as sample size increases
def plot_se_convergence(summary: pd.DataFrame) -> None:
    """Line plot: theoretical vs empirical standard error shrinking as n grows."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["n"], summary["theoretical_se_sigma_over_sqrt_n"],
            marker="o", label="Theoretical SE = sigma / sqrt(n)", color="black")
    ax.plot(summary["n"], summary["empirical_se_of_sample_means"],
            marker="s", linestyle="--", label="Empirical SE (simulation)", color="#C44E52")
    ax.set_xlabel("Sample Size (n)")
    ax.set_ylabel("Standard Error of the Mean ($)")
    ax.set_title("Standard Error Shrinks as Sample Size Grows")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "clt_standard_error_convergence.png")
    plt.close(fig)


if __name__ == "__main__":
    income = load_data()
    pop_stats = population_stats(income)
    summary, sampling_results = run_clt_simulation(income)
    plot_clt_grid(income, sampling_results, pop_stats["mu"], pop_stats["sigma"])
    plot_se_convergence(summary)
