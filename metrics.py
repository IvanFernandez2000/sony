"""
Consumption metrics for one movie (dataset C).

Grain of consumption: 1 row per movie × month × country × platform.
All aggregations start from this table alone (never joined with
availability), so sums are never inflated by duplicated rows.
"""
import pandas as pd

METRIC_LABELS = {"streams": "Streams", "total_minutes": "Total minutes"}
SPLIT_OPTIONS = {"Total": None, "By country": "country", "By platform": "platform"}


def filter_consumption(consumption, title_id, countries=None, platforms=None):
    """Rows for one movie, optionally restricted to some countries/platforms.

    An empty or None filter means "all".
    """
    df = consumption[consumption["title_id"] == title_id]
    if countries:
        df = df[df["country"].isin(countries)]
    if platforms:
        df = df[df["platform"].isin(platforms)]
    return df


def summarize(df):
    """Headline numbers for the filtered rows."""
    total_streams = df["streams"].sum()
    total_minutes = df["total_minutes"].sum()

    # Ratio of sums, NOT the average of per-row ratios:
    # averaging ratios would give a tiny month the same weight as a huge one.
    minutes_per_stream = total_minutes / total_streams if total_streams > 0 else None

    monthly = df.groupby("month")["streams"].sum()
    peak_month = monthly.idxmax() if not monthly.empty and monthly.max() > 0 else None

    return {
        "total_streams": total_streams,
        "total_minutes": total_minutes,
        "minutes_per_stream": minutes_per_stream,
        "peak_month": peak_month,
    }


def monthly_series(df, metric, split=None):
    """Monthly totals of `metric`, optionally one series per country/platform.

    Months with no records are kept as empty values (NaN) instead of being
    dropped, so the chart shows a gap. Otherwise the line would connect the
    surrounding months and suggest data that doesn't exist. We don't fill
    them with 0 because the dataset doesn't say whether a missing month
    means zero consumption or no data.
    """
    group_cols = ["month"] + ([split] if split else [])
    agg = df.groupby(group_cols, as_index=False)[metric].sum()

    all_months = pd.date_range(df["month"].min(), df["month"].max(), freq="MS")
    if split:
        full_index = pd.MultiIndex.from_product(
            [all_months, sorted(df[split].unique())], names=["month", split]
        )
        agg = agg.set_index(["month", split]).reindex(full_index).reset_index()
    else:
        agg = agg.set_index("month").reindex(all_months).rename_axis("month").reset_index()
    return agg