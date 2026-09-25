"""
Stock Semiannual Base-Price Analysis

For every stock CSV:

1. Identify 20 March and 20 September for every available year.
2. Starting from that date, take the first 15 trading sessions.
3. Calculate the average Close of those 15 sessions.
4. Treat this average as the Base Price.
5. Define the following ~4 months as the analysis period.
6. Calculate:
       - Highest High
       - Lowest Low
       - Upside from Base
       - Downside from Base
       - High-Low spread from Base
7. Generate results for every stock and every valid year/event.

Input:
    /Users/umanggohil/Documents/BGYRT/dev-work/nifty500/daily

Output:
    stock_semiannual_analysis.csv
    stock_semiannual_analysis_by_stock/
"""


import os
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIRECTORY = Path(
    "/Users/umanggohil/Documents/BGYRT/dev-work/nifty500/daily"
)

OUTPUT_FILE = Path(
    "/Users/umanggohil/Documents/BGYRT/dev-work/General/equinox/output/"
    "stock_semiannual_analysis.csv"
)

OUTPUT_DIRECTORY = Path(
    "/Users/umanggohil/Documents/BGYRT/dev-work/General/equinox/output/"
    "stock_semiannual_analysis_by_stock"
)

# Number of trading sessions used to calculate base price
BASE_TRADING_DAYS = 15

# Approximate length of analysis period
ANALYSIS_MONTHS = 4


# ============================================================
# HELPERS
# ============================================================

def get_analysis_dates(df):
    """
    Generate all 20 March and 20 September reference dates
    for the years covered by the stock data.
    """

    min_year = df["Date"].dt.year.min()
    max_year = df["Date"].dt.year.max()

    dates = []

    for year in range(min_year, max_year + 1):

        dates.append(
            pd.Timestamp(
                year=year,
                month=3,
                day=20
            )
        )

        dates.append(
            pd.Timestamp(
                year=year,
                month=9,
                day=20
            )
        )

    return dates


def get_first_trading_day_on_or_after(
    df,
    target_date
):
    """
    Return the first available trading day on or after target_date.
    """

    matching = df[
        df["Date"] >= target_date
    ]

    if matching.empty:
        return None

    return matching.iloc[0]["Date"]


def add_months(timestamp, months):
    """
    Add calendar months while preserving the day where possible.

    Example:
        2026-03-20 + 4 months
        -> 2026-07-20
    """

    return timestamp + pd.DateOffset(
        months=months
    )


# ============================================================
# ANALYSIS FOR ONE EVENT
# ============================================================

def analyze_event(
    df,
    stock,
    reference_date,
    event_name
):
    """
    Analyze one 20 March / 20 September event.
    """

    # --------------------------------------------------------
    # Find actual first trading day on/after 20 March/20 Sep
    # --------------------------------------------------------

    actual_start_date = get_first_trading_day_on_or_after(
        df,
        reference_date
    )

    if actual_start_date is None:
        return None

    # --------------------------------------------------------
    # Get 15 trading sessions
    # --------------------------------------------------------

    base_period = df[
        df["Date"] >= actual_start_date
    ].head(BASE_TRADING_DAYS)

    if len(base_period) < BASE_TRADING_DAYS:

        return {
            "Stock": stock,
            "Year": reference_date.year,
            "Event": event_name,
            "Reference Date": reference_date,
            "Actual Start Date": actual_start_date,
            "Status": "INSUFFICIENT_BASE_DATA"
        }

    # --------------------------------------------------------
    # Base period dates
    # --------------------------------------------------------

    base_start_date = base_period.iloc[0]["Date"]
    base_end_date = base_period.iloc[-1]["Date"]

    # --------------------------------------------------------
    # Base price
    #
    # Average of 15 trading days' CLOSE
    # --------------------------------------------------------

    base_price = base_period["Close"].mean()

    # --------------------------------------------------------
    # Base period statistics
    # --------------------------------------------------------

    base_period_high = base_period["High"].max()

    base_period_low = base_period["Low"].min()

    # --------------------------------------------------------
    # Analysis period
    #
    # Starts AFTER the 15th trading session.
    #
    # Ends approximately 4 calendar months after
    # the original reference date.
    # --------------------------------------------------------

    analysis_start_date = (
        base_end_date
        + pd.Timedelta(days=1)
    )

    analysis_end_date = add_months(
        reference_date,
        ANALYSIS_MONTHS
    )

    analysis_period = df[
        (df["Date"] >= analysis_start_date)
        &
        (df["Date"] <= analysis_end_date)
    ]

    if analysis_period.empty:

        return {
            "Stock": stock,
            "Year": reference_date.year,
            "Event": event_name,
            "Reference Date": reference_date,
            "Actual Start Date": actual_start_date,
            "Base Start Date": base_start_date,
            "Base End Date": base_end_date,
            "Base Trading Days": len(base_period),
            "Base Price": base_price,
            "Status": "NO_ANALYSIS_DATA"
        }

    # --------------------------------------------------------
    # 4-month high / low
    # --------------------------------------------------------

    highest_price = analysis_period["High"].max()

    lowest_price = analysis_period["Low"].min()

    # Dates on which high / low occurred
    high_row = analysis_period.loc[
        analysis_period["High"].idxmax()
    ]

    low_row = analysis_period.loc[
        analysis_period["Low"].idxmin()
    ]

    high_date = high_row["Date"]

    low_date = low_row["Date"]

    # --------------------------------------------------------
    # Percentage calculations
    # --------------------------------------------------------

    upside_pct = (
        (highest_price - base_price)
        / base_price
        * 100
    )

    downside_pct = (
        (lowest_price - base_price)
        / base_price
        * 100
    )

    high_low_pct = (
        (highest_price - lowest_price)
        / base_price
        * 100
    )

    # --------------------------------------------------------
    # Additional useful statistics
    # --------------------------------------------------------

    highest_close = analysis_period["Close"].max()

    lowest_close = analysis_period["Close"].min()

    highest_close_pct = (
        (highest_close - base_price)
        / base_price
        * 100
    )

    lowest_close_pct = (
        (lowest_close - base_price)
        / base_price
        * 100
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {
        "Stock": stock,

        "Year": reference_date.year,

        "Event": event_name,

        "Reference Date": reference_date,

        "Actual Start Date": actual_start_date,

        "Base Start Date": base_start_date,

        "Base End Date": base_end_date,

        "Base Trading Days": len(base_period),

        "Base Price": base_price,

        "Base Period High": base_period_high,

        "Base Period Low": base_period_low,

        "Analysis Start Date": analysis_start_date,

        "Analysis End Date": analysis_end_date,

        "Analysis Trading Days": len(
            analysis_period
        ),

        "Highest High": highest_price,

        "Highest High Date": high_date,

        "Lowest Low": lowest_price,

        "Lowest Low Date": low_date,

        "Upside %": upside_pct,

        "Downside %": downside_pct,

        "High-Low %": high_low_pct,

        "Highest Close": highest_close,

        "Highest Close %": highest_close_pct,

        "Lowest Close": lowest_close,

        "Lowest Close %": lowest_close_pct,

        "Status": "SUCCESS"
    }


# ============================================================
# ANALYZE ONE STOCK
# ============================================================

def analyze_stock(
    file_path
):
    """
    Perform complete analysis for one stock CSV.
    """

    stock = file_path.stem

    print(
        f"Processing: {stock}"
    )

    # --------------------------------------------------------
    # Read data
    # --------------------------------------------------------

    try:

        df = pd.read_csv(
            file_path
        )

    except Exception as e:

        print(
            f"ERROR reading {file_path}: {e}"
        )

        return []

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    required_columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print(
            f"{stock}: Missing columns "
            f"{missing_columns}"
        )

        return []

    # --------------------------------------------------------
    # Convert data types
    # --------------------------------------------------------

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    # CSV contains dates such as:
    # 2008-03-20 00:00:00+05:30
    #
    # Remove timezone information but preserve
    # the original local timestamp.
    if df["Date"].dt.tz is not None:
        df["Date"] = df["Date"].dt.tz_localize(None)

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Remove invalid rows
    # --------------------------------------------------------

    df = df.dropna(
        subset=[
            "Date",
            "High",
            "Low",
            "Close"
        ]
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if df.empty:

        print(
            f"{stock}: No valid data"
        )

        return []

    # --------------------------------------------------------
    # Generate March / September dates
    # --------------------------------------------------------

    analysis_dates = get_analysis_dates(
        df
    )

    results = []

    # --------------------------------------------------------
    # Analyze each event
    # --------------------------------------------------------

    for reference_date in analysis_dates:

        if reference_date.month == 3:

            event_name = "20-March"

        else:

            event_name = "20-September"

        result = analyze_event(
            df=df,
            stock=stock,
            reference_date=reference_date,
            event_name=event_name
        )

        if result is not None:

            results.append(
                result
            )

    print(
        f"{stock}: {len(results)} events"
    )

    return results


# ============================================================
# PROCESS ALL STOCKS
# ============================================================

def run_analysis():

    print("=" * 70)

    print(
        "STOCK SEMIANNUAL ANALYSIS"
    )

    print("=" * 70)

    print(
        f"Input directory: {DATA_DIRECTORY}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find CSV files
    # --------------------------------------------------------

    files = sorted(
        DATA_DIRECTORY.glob("*.csv")
    )

    if not files:

        raise RuntimeError(
            f"No CSV files found in {DATA_DIRECTORY}"
        )

    print(
        f"Stocks found: {len(files)}"
    )

    # --------------------------------------------------------
    # Analyze all stocks
    # --------------------------------------------------------

    all_results = []

    for index, file_path in enumerate(
        files,
        start=1
    ):

        print(
            f"\n[{index}/{len(files)}]"
        )

        results = analyze_stock(
            file_path
        )

        all_results.extend(
            results
        )

    # --------------------------------------------------------
    # Create final DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    if results_df.empty:

        print(
            "No analysis results generated."
        )

        return

    # --------------------------------------------------------
    # Sort results
    # --------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "Stock",
                "Reference Date"
            ]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Round numerical values
    # --------------------------------------------------------

    percentage_columns = [
        "Upside %",
        "Downside %",
        "High-Low %",
        "Highest Close %",
        "Lowest Close %"
    ]

    price_columns = [
        "Base Price",
        "Base Period High",
        "Base Period Low",
        "Highest High",
        "Lowest Low",
        "Highest Close",
        "Lowest Close"
    ]

    results_df[percentage_columns] = (
        results_df[percentage_columns]
        .round(2)
    )

    results_df[price_columns] = (
        results_df[price_columns]
        .round(4)
    )

    # --------------------------------------------------------
    # Save master result
    # --------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nMaster output saved:"
    )

    print(
        OUTPUT_FILE
    )

    # --------------------------------------------------------
    # Save individual stock files
    # --------------------------------------------------------

    for stock, stock_df in results_df.groupby(
        "Stock"
    ):

        stock_output = (
            OUTPUT_DIRECTORY
            / f"{stock}.csv"
        )

        stock_df.to_csv(
            stock_output,
            index=False
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "ANALYSIS COMPLETE"
    )

    print("=" * 70)

    print(
        f"Stocks analyzed : "
        f"{results_df['Stock'].nunique()}"
    )

    print(
        f"Events generated: "
        f"{len(results_df)}"
    )

    print(
        f"Successful       : "
        f"{(results_df['Status'] == 'SUCCESS').sum()}"
    )

    print(
        f"Output           : "
        f"{OUTPUT_FILE}"
    )

    print(
        f"Per-stock output : "
        f"{OUTPUT_DIRECTORY}"
    )

    print("=" * 70)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_analysis()