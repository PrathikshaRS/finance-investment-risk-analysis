"""
Phase 1 preprocessing pipeline: S&P 500 price-change data + sector mapping.

Design notes (kept deliberately lightweight for a memory-constrained machine):
- Each file is read from disk exactly once.
- The wide price table is converted to long format only after pre-listing
  zero-padding has been identified, using a vectorised mask (no per-ticker
  Python loop, no repeated full-frame copies).
- Numeric columns are downcast (float32 / category) to shrink memory before
  the wide-to-long reshape, since that step is the memory peak of this
  pipeline.
- No cumulative sums / price reconstruction is performed anywhere here.
  Values are kept as daily price CHANGES (Close_t - Close_{t-1}) throughout.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SP500_PRICE_CSV = Path("data/raw/sp500_data.csv")
SP500_SECTORS_CSV = Path("data/raw/sp500_sectors.csv")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_sector_map() -> pd.DataFrame:
    """Load and clean the sector classification file."""
    sec = pd.read_csv(SP500_SECTORS_CSV, dtype="category")
    # sector_label contains embedded newlines, e.g. "Consumer\nDiscretionary"
    sec["sector_label"] = (
        sec["sector_label"].astype(str).str.replace("\n", " ", regex=False).str.strip()
    )
    sec["sector_label"] = sec["sector_label"].astype("category")
    # Flag non-company rows (ETFs) rather than silently dropping them -
    # downstream phases can decide whether to include them.
    sec["is_etf"] = (sec["sector"] == "etf").astype("int8")
    sec = sec.drop_duplicates(subset="symbol").reset_index(drop=True)
    return sec


def load_price_wide() -> pd.DataFrame:
    """Load the wide-format daily price-change table (one read, downcast)."""
    df = pd.read_csv(SP500_PRICE_CSV)
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])
    # Downcast all ticker columns to float32 in place to roughly halve
    # memory footprint before the reshape step below.
    ticker_cols = df.columns.drop("date")
    df[ticker_cols] = df[ticker_cols].astype("float32")
    df = df.set_index("date").sort_index()
    return df


def trim_pre_listing_padding(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Replace leading zero-padding (pre-IPO / not-yet-listed placeholder rows)
    with NaN, per ticker, using each ticker's own first non-zero observation.
    Zeros that occur AFTER a ticker's first real observation are left alone,
    since a $0.00 day-over-day change is a legitimate value once a stock is
    actually trading.
    """
    values = wide.to_numpy()
    nonzero = values != 0
    has_any = nonzero.any(axis=0)
    # index of first non-zero row per column; columns with no non-zero
    # value at all are marked to exclude entirely (first_idx = n_rows)
    first_idx = np.where(has_any, nonzero.argmax(axis=0), values.shape[0])

    row_idx = np.arange(values.shape[0])[:, None]
    pre_listing_mask = row_idx < first_idx[None, :]

    trimmed = values.copy()
    trimmed[pre_listing_mask] = np.nan
    trimmed_df = pd.DataFrame(trimmed, index=wide.index, columns=wide.columns)
    return trimmed_df, pre_listing_mask.sum(axis=0), first_idx


def wide_to_long(trimmed_wide: pd.DataFrame) -> pd.DataFrame:
    """Melt to long format, dropping the pre-listing NaNs (not real data)."""
    long_df = trimmed_wide.reset_index().melt(
        id_vars="date", var_name="symbol", value_name="price_change"
    )
    long_df = long_df.dropna(subset=["price_change"])
    long_df["symbol"] = long_df["symbol"].astype("category")
    long_df["price_change"] = long_df["price_change"].astype("float32")
    return long_df.reset_index(drop=True)


def merge_with_sectors(long_df: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    merged = long_df.merge(
        sector_map[["symbol", "sector", "sector_label", "sub_sector", "is_etf"]],
        on="symbol",
        how="left",
    )
    return merged


def run_pipeline(save: bool = True) -> dict:
    report = {}

    sector_map = load_sector_map()
    report["sector_map_shape"] = sector_map.shape

    wide_raw = load_price_wide()
    report["price_wide_raw_shape"] = wide_raw.shape

    trimmed_wide, n_trimmed_per_col, first_idx = trim_pre_listing_padding(wide_raw)
    report["total_pre_listing_cells_removed"] = int(n_trimmed_per_col.sum())
    report["tickers_with_no_data_at_all"] = int((first_idx == wide_raw.shape[0]).sum())

    long_df = wide_to_long(trimmed_wide)
    report["long_shape_before_sector_merge"] = long_df.shape

    merged = merge_with_sectors(long_df, sector_map)
    report["long_shape_after_sector_merge"] = merged.shape
    report["unmatched_symbols"] = int(merged["sector"].isna().sum())

    if save:
        # pyarrow/fastparquet aren't installed and adding one just for this
        # is unnecessary weight on a constrained machine - pickle is fine
        # for an intermediate, same-environment cache file.
        merged.to_pickle(OUT_DIR / "sp500_price_changes_long.pkl")
        sector_map.to_pickle(OUT_DIR / "sp500_sectors_clean.pkl")

    return {
        "report": report,
        "sector_map": sector_map,
        "trimmed_wide": trimmed_wide,
        "long_df": long_df,
        "merged": merged,
        "first_idx": first_idx,
    }


if __name__ == "__main__":
    result = run_pipeline()
    for k, v in result["report"].items():
        print(f"{k}: {v}")