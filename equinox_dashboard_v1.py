
import os
from pathlib import Path

import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

DEFAULT_DATA_FILE = os.getenv(
    "EQUINOX_ANALYSIS_FILE",
    "output/stock_semiannual_analysis.csv",
)

REQUIRED_COLUMNS = [
    "Stock",
    "Year",
    "Event",
    "Status",
    "Base Price",
    "Highest High",
    "Lowest Low",
    "Upside %",
    "Downside %",
    "High-Low %",
    "Analysis Trading Days",
]


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the event-analysis CSV and keep only valid observations.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    # Types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")

    numeric_cols = [
        "Base Price",
        "Highest High",
        "Lowest Low",
        "Upside %",
        "Downside %",
        "High-Low %",
        "Analysis Trading Days",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep only successful, usable observations.
    df = df[
        df["Status"].eq("SUCCESS")
        & df["Upside %"].notna()
        & df["Downside %"].notna()
        & df["High-Low %"].notna()
        & df["Base Price"].gt(0)
    ].copy()

    # Normalize event labels
    df["Cycle"] = df["Event"].map(
        {
            "20-March": "20-March",
            "20-September": "20-September",
        }
    )

    df = df[df["Cycle"].notna()].copy()

    return df


# ============================================================
# AGGREGATIONS
# ============================================================

def aggregate_stock_metrics(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Aggregate filtered observations to one row per stock.
    """

    if df.empty:
        return pd.DataFrame()

    grouped = df.groupby("Stock", as_index=False)

    out = grouped.agg(
        Observations=("Upside %", "count"),
        Avg_Upside=("Upside %", "mean"),
        Median_Upside=("Upside %", "median"),
        Avg_Downside=("Downside %", "mean"),
        Median_Downside=("Downside %", "median"),
        Avg_High_Low=("High-Low %", "mean"),
        Max_Upside=("Upside %", "max"),
        Min_Upside=("Upside %", "min"),
        Worst_Downside=("Downside %", "min"),
        Best_Downside=("Downside %", "max"),
        Avg_Base_Price=("Base Price", "mean"),
        Positive_High_Rate=(
            "Upside %",
            lambda x: (x > 0).mean() * 100,
        ),
        Threshold_Hit_Rate=(
            "Upside %",
            lambda x: (x >= threshold).mean() * 100,
        ),
        Down_10_Rate=(
            "Downside %",
            lambda x: (x <= -10).mean() * 100,
        ),
    )

    # Transparent composite:
    # average upside * positive-observation rate.
    # This rewards both magnitude and consistency.
    out["Consistency_Score"] = (
        out["Avg_Upside"]
        * out["Positive_High_Rate"]
        / 100.0
    )

    return out


def cycle_summary(
    df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """
    Summary split by March / September cycle.
    """

    grouped = df.groupby("Cycle")

    return (
        grouped.agg(
            Stocks=("Stock", "nunique"),
            Observations=("Upside %", "count"),
            Avg_Upside=("Upside %", "mean"),
            Median_Upside=("Upside %", "median"),
            Avg_Downside=("Downside %", "mean"),
            Positive_Rate=(
                "Upside %",
                lambda x: (x > 0).mean() * 100,
            ),
            Threshold_Hit_Rate=(
                "Upside %",
                lambda x: (x >= threshold).mean() * 100,
            ),
        )
        .reset_index()
    )


def cycle_comparison(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-stock March vs September comparison.
    """

    tmp = (
        df.groupby(["Stock", "Cycle"])
        .agg(
            Observations=("Upside %", "count"),
            Avg_Upside=("Upside %", "mean"),
            Median_Upside=("Upside %", "median"),
            Positive_Rate=(
                "Upside %",
                lambda x: (x > 0).mean() * 100,
            ),
            Avg_Downside=("Downside %", "mean"),
            Avg_High_Low=("High-Low %", "mean"),
        )
        .reset_index()
    )

    # Pivot to wide columns.
    wide = tmp.pivot(
        index="Stock",
        columns="Cycle",
        values=[
            "Observations",
            "Avg_Upside",
            "Median_Upside",
            "Positive_Rate",
            "Avg_Downside",
            "Avg_High_Low",
        ],
    )

    wide.columns = [
        f"{metric}_{cycle}"
        for metric, cycle in wide.columns
    ]

    wide = wide.reset_index()

    # Convenience classification.
    def classify(row):
        march = row.get("Avg_Upside_20-March")
        sept = row.get("Avg_Upside_20-September")

        if pd.isna(march) and pd.isna(sept):
            return "No data"
        if pd.isna(march):
            return "September only"
        if pd.isna(sept):
            return "March only"

        if march > 0 and sept > 0:
            return "Positive in both"
        if march < 0 and sept < 0:
            return "Negative in both"
        if march > 0 and sept < 0:
            return "March positive / September negative"
        if march < 0 and sept > 0:
            return "March negative / September positive"

        return "Mixed / zero"

    wide["Cycle_Profile"] = wide.apply(classify, axis=1)

    return wide


# ============================================================
# FORMATTING
# ============================================================

def pct_style(df: pd.DataFrame, columns):
    """
    Apply percentage formatting to numeric columns.
    """

    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df

    return df.style.format(
        {col: "{:.2f}%" for col in existing},
        na_rep="-",
    )


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="NSE Equinox Stock Analysis",
    page_icon="📈",
    layout="wide",
)

st.title("NSE Equinox / Semiannual Stock Analysis")
st.caption(
    "Base price = average Close of first 15 trading sessions "
    "from 20 March / 20 September. Subsequent ~4-month "
    "High/Low performance is measured from that base."
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Filters")

data_file = st.sidebar.text_input(
    "Analysis CSV",
    value=DEFAULT_DATA_FILE,
    help="Path to stock_semiannual_analysis.csv",
)

daily_data_directory = st.sidebar.text_input(
    "Daily stock CSV directory",
    value=os.getenv(
        "EQUINOX_DAILY_DATA_DIRECTORY",
        "daily",
    ),
    help="Directory containing STOCK.csv files such as RELIANCE.csv.",
)

# Keep the directory available to the raw-data viewer.
os.environ["EQUINOX_DAILY_DATA_DIRECTORY"] = daily_data_directory

try:
    data = load_data(data_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

min_year_available = int(data["Year"].min())
max_year_available = int(data["Year"].max())

from_year = st.sidebar.number_input(
    "From year (inclusive)",
    min_value=min_year_available,
    max_value=max_year_available,
    value=max(min_year_available, 2020),
    step=1,
)

to_year = st.sidebar.number_input(
    "To year (inclusive)",
    min_value=min_year_available,
    max_value=max_year_available,
    value=max_year_available,
    step=1,
)

cycle = st.sidebar.selectbox(
    "Cycle",
    options=["Both", "20-March", "20-September"],
)

min_observations = st.sidebar.number_input(
    "Minimum observations per stock",
    min_value=1,
    max_value=46,
    value=6,
    step=1,
    help=(
        "Prevents stocks with very short histories from dominating "
        "the ranking."
    ),
)

threshold = st.sidebar.number_input(
    "High threshold %",
    min_value=-100.0,
    max_value=1000.0,
    value=20.0,
    step=5.0,
    help="Used for Threshold Hit Rate.",
)

ranking_metric = st.sidebar.selectbox(
    "Rank stocks by",
    options=[
        "Consistency_Score",
        "Avg_Upside",
        "Median_Upside",
        "Positive_High_Rate",
        "Threshold_Hit_Rate",
        "Avg_Downside",
    ],
    format_func=lambda x: {
        "Consistency_Score": "Consistency Score",
        "Avg_Upside": "Average Upside %",
        "Median_Upside": "Median Upside %",
        "Positive_High_Rate": "Positive High Rate %",
        "Threshold_Hit_Rate": "Threshold Hit Rate %",
        "Avg_Downside": "Average Downside %",
    }[x],
)

# ============================================================
# APPLY FILTERS
# ============================================================

filtered = data[
    (data["Year"] >= from_year)
    & (data["Year"] <= to_year)
].copy()

if cycle != "Both":
    filtered = filtered[
        filtered["Cycle"].eq(cycle)
    ].copy()

st.sidebar.markdown("---")
st.sidebar.caption(
    f"Valid observations: {len(filtered):,}"
)
st.sidebar.caption(
    f"Stocks in filtered data: {filtered['Stock'].nunique():,}"
)

if filtered.empty:
    st.warning("No valid observations match the selected filters.")
    st.stop()

# ============================================================
# STOCK AGGREGATION
# ============================================================

ranking = aggregate_stock_metrics(
    filtered,
    threshold=threshold,
)

ranking = ranking[
    ranking["Observations"] >= min_observations
].copy()

if ranking.empty:
    st.warning(
        "No stocks satisfy the minimum observation filter. "
        "Reduce Minimum observations per stock."
    )
    st.stop()

ranking = ranking.sort_values(
    ranking_metric,
    ascending=False,
).reset_index(drop=True)

ranking.insert(
    0,
    "Rank",
    range(1, len(ranking) + 1),
)

# ============================================================
# KPI ROW
# ============================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Stocks",
        f"{ranking['Stock'].nunique():,}",
    )

with col2:
    st.metric(
        "Observations",
        f"{int(ranking['Observations'].sum()):,}",
    )

with col3:
    st.metric(
        "Avg Upside",
        f"{ranking['Avg_Upside'].mean():.2f}%",
    )

with col4:
    st.metric(
        "Positive Rate",
        f"{ranking['Positive_High_Rate'].mean():.1f}%",
    )

with col5:
    st.metric(
        f"≥ {threshold:.0f}% Hit Rate",
        f"{ranking['Threshold_Hit_Rate'].mean():.1f}%",
    )

# ============================================================
# TABS
# ============================================================

tab_overview, tab_ranking, tab_cycles, tab_stock = st.tabs(
    [
        "Overview",
        "Stock Ranking",
        "March vs September",
        "Stock Detail",
    ]
)

# ============================================================
# OVERVIEW
# ============================================================

with tab_overview:

    st.subheader("How the filtered universe behaves")

    summary = cycle_summary(
        filtered,
        threshold=threshold,
    )

    summary_display = summary.rename(
        columns={
            "Cycle": "Cycle",
            "Stocks": "Stocks",
            "Observations": "Observations",
            "Avg_Upside": "Average Upside %",
            "Median_Upside": "Median Upside %",
            "Avg_Downside": "Average Downside %",
            "Positive_Rate": "Positive High Rate %",
            "Threshold_Hit_Rate": "Threshold Hit Rate %",
        }
    )

    st.dataframe(
        pct_style(
            summary_display,
            [
                "Average Upside %",
                "Median Upside %",
                "Average Downside %",
                "Positive High Rate %",
                "Threshold Hit Rate %",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Average Upside by Year")

    yearly = (
        filtered.groupby(["Year", "Cycle"])["Upside %"]
        .mean()
        .reset_index()
        .pivot(
            index="Year",
            columns="Cycle",
            values="Upside %",
        )
    )

    if not yearly.empty:
        st.line_chart(yearly)

    st.subheader("Positive High Rate by Year")

    yearly_positive = (
        filtered.assign(
            Positive=filtered["Upside %"] > 0
        )
        .groupby(["Year", "Cycle"])["Positive"]
        .mean()
        .mul(100)
        .reset_index()
        .pivot(
            index="Year",
            columns="Cycle",
            values="Positive",
        )
    )

    if not yearly_positive.empty:
        st.line_chart(yearly_positive)

    st.info(
        "Interpretation: Average Upside measures the magnitude of the "
        "best price excursion. Positive High Rate measures how often "
        "that 4-month High was above the base price. Threshold Hit Rate "
        f"measures how often the High reached at least +{threshold:.0f}%."
    )

# ============================================================
# STOCK RANKING
# ============================================================

with tab_ranking:

    st.subheader(
        f"Ranking by {ranking_metric.replace('_', ' ')}"
    )

    display_cols = [
        "Rank",
        "Stock",
        "Observations",
        "Avg_Upside",
        "Median_Upside",
        "Positive_High_Rate",
        "Threshold_Hit_Rate",
        "Avg_Downside",
        "Worst_Downside",
        "Avg_High_Low",
        "Consistency_Score",
    ]

    ranking_display = ranking[display_cols].rename(
        columns={
            "Avg_Upside": "Avg Upside %",
            "Median_Upside": "Median Upside %",
            "Positive_High_Rate": "Positive Rate %",
            "Threshold_Hit_Rate": "Threshold Hit %",
            "Avg_Downside": "Avg Downside %",
            "Worst_Downside": "Worst Downside %",
            "Avg_High_Low": "Avg High-Low %",
            "Consistency_Score": "Consistency Score",
        }
    )

    ranking_event = st.dataframe(
        pct_style(
            ranking_display,
            [
                "Avg Upside %",
                "Median Upside %",
                "Positive Rate %",
                "Threshold Hit %",
                "Avg Downside %",
                "Worst Downside %",
                "Avg High-Low %",
            ],
        ),
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="stock_ranking_table",
    )

    csv_data = ranking_display.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download filtered ranking CSV",
        data=csv_data,
        file_name="equinox_filtered_stock_ranking.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # Open the underlying daily CSV when a ranking row is clicked
    # --------------------------------------------------------

    selected_stock_from_ranking = None

    try:
        selected_rows = ranking_event.selection.rows
    except AttributeError:
        selected_rows = []

    if selected_rows:
        selected_index = selected_rows[0]
        if 0 <= selected_index < len(ranking_display):
            selected_stock_from_ranking = ranking_display.iloc[
                selected_index
            ]["Stock"]

    if selected_stock_from_ranking:
        st.markdown("---")
        st.subheader(
            f"Daily CSV Data — {selected_stock_from_ranking}"
        )

        raw_csv_path = (
            Path(daily_data_directory)
            / f"{selected_stock_from_ranking}.csv"
        )

        if raw_csv_path.exists():
            try:
                raw_df = pd.read_csv(raw_csv_path)

                st.caption(
                    f"{len(raw_df):,} daily rows | "
                    f"{raw_df['Date'].min()} → {raw_df['Date'].max()}"
                )

                st.dataframe(
                    raw_df,
                    use_container_width=True,
                    hide_index=True,
                    height=550,
                )

                raw_csv_data = raw_df.to_csv(index=False).encode(
                    "utf-8"
                )

                st.download_button(
                    label=f"Download {selected_stock_from_ranking}.csv",
                    data=raw_csv_data,
                    file_name=f"{selected_stock_from_ranking}.csv",
                    mime="text/csv",
                    key=f"download_raw_{selected_stock_from_ranking}",
                )

            except Exception as exc:
                st.error(
                    f"Unable to read {raw_csv_path}: {exc}"
                )
        else:
            st.warning(
                f"Underlying daily CSV not found: {raw_csv_path}"
            )

    st.markdown("---")

    st.subheader("Top stocks")

    top_n = min(25, len(ranking))

    top_chart = (
        ranking.head(top_n)
        .set_index("Stock")[[ranking_metric]]
        .rename(columns={ranking_metric: ranking_metric.replace("_", " ")})
    )

    st.bar_chart(top_chart)

    st.caption(
        "Consistency Score = Average Upside × Positive High Rate / 100. "
        "It is intentionally transparent rather than a statistical forecast."
    )

# ============================================================
# CYCLE COMPARISON
# ============================================================

with tab_cycles:

    st.subheader("March vs September behavior")

    comparison = cycle_comparison(
        filtered,
    )

    comparison_filtered = comparison[
        comparison["Stock"].isin(
            ranking["Stock"]
        )
    ].copy()

    profile_counts = (
        comparison_filtered["Cycle_Profile"]
        .value_counts()
        .rename_axis("Profile")
        .reset_index(name="Stocks")
    )

    st.dataframe(
        profile_counts,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Per-stock cycle comparison")

    cycle_cols = [
        "Stock",
        "Avg_Upside_20-March",
        "Positive_Rate_20-March",
        "Avg_Downside_20-March",
        "Avg_Upside_20-September",
        "Positive_Rate_20-September",
        "Avg_Downside_20-September",
        "Cycle_Profile",
    ]

    cycle_display = comparison_filtered[
        [
            c
            for c in cycle_cols
            if c in comparison_filtered.columns
        ]
    ].rename(
        columns={
            "Avg_Upside_20-March": "March Avg Upside %",
            "Positive_Rate_20-March": "March Positive Rate %",
            "Avg_Downside_20-March": "March Avg Downside %",
            "Avg_Upside_20-September": "September Avg Upside %",
            "Positive_Rate_20-September": "September Positive Rate %",
            "Avg_Downside_20-September": "September Avg Downside %",
        }
    )

    cycle_display = cycle_display.sort_values(
        ["Cycle_Profile", "Stock"]
    )

    st.dataframe(
        pct_style(
            cycle_display,
            [
                "March Avg Upside %",
                "March Positive Rate %",
                "March Avg Downside %",
                "September Avg Upside %",
                "September Positive Rate %",
                "September Avg Downside %",
            ],
        ),
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# STOCK DETAIL
# ============================================================

with tab_stock:

    stocks = sorted(filtered["Stock"].unique())

    selected_stock = st.selectbox(
        "Select stock",
        stocks,
    )

    st.caption(
        "Tip: click any stock row in the Stock Ranking tab to inspect "
        "its historical analysis and underlying daily CSV data."
    )

    stock_data = filtered[
        filtered["Stock"].eq(selected_stock)
    ].sort_values(
        ["Year", "Cycle"]
    ).copy()

    st.subheader(selected_stock)

    detail_cols = [
        "Year",
        "Cycle",
        "Base Price",
        "Highest High",
        "Lowest Low",
        "Upside %",
        "Downside %",
        "High-Low %",
        "Analysis Trading Days",
    ]

    detail = stock_data[detail_cols].copy()

    st.dataframe(
        detail.style.format(
            {
                "Base Price": "{:.4f}",
                "Highest High": "{:.4f}",
                "Lowest Low": "{:.4f}",
                "Upside %": "{:.2f}%",
                "Downside %": "{:.2f}%",
                "High-Low %": "{:.2f}%",
            },
            na_rep="-",
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Year / cycle chart
    chart_df = (
        stock_data[["Year", "Cycle", "Upside %"]]
        .pivot(
            index="Year",
            columns="Cycle",
            values="Upside %",
        )
    )

    st.subheader("Historical Upside by cycle")

    if not chart_df.empty:
        st.line_chart(chart_df)

    # Summary for selected stock
    stock_summary = aggregate_stock_metrics(
        stock_data,
        threshold=threshold,
    )

    if not stock_summary.empty:

        st.subheader("Selected stock summary")

        summary_row = stock_summary.iloc[0]

        a, b, c, d = st.columns(4)

        with a:
            st.metric(
                "Average Upside",
                f"{summary_row['Avg_Upside']:.2f}%",
            )

        with b:
            st.metric(
                "Median Upside",
                f"{summary_row['Median_Upside']:.2f}%",
            )

        with c:
            st.metric(
                "Positive Rate",
                f"{summary_row['Positive_High_Rate']:.1f}%",
            )

        with d:
            st.metric(
                f"≥ {threshold:.0f}% Hit Rate",
                f"{summary_row['Threshold_Hit_Rate']:.1f}%",
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.caption(
    "This dashboard describes historical observations in the supplied "
    "dataset. It does not forecast future returns."
)
