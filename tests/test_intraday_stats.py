import pandas as pd

from swing_data.intraday_stats import summarize_morning_lows


def _sample_lows():
    return pd.DataFrame(
        {
            "Ticker": ["7203.T", "7203.T", "6758.T", "6758.T"],
            "Date": [
                "2026-09-14",
                "2026-09-15",
                "2026-09-14",
                "2026-09-15",
            ],
            "MinutesFromOpen": [2, 8, 12, 18],
        }
    )


def test_statistics_include_overall_and_per_ticker():
    statistics, _ = summarize_morning_lows(_sample_lows())

    overall = statistics[statistics["Scope"] == "ALL"].iloc[0]
    assert overall["Ticker"] == "ALL"
    assert overall["Observations"] == 4
    assert overall["MeanMinutesFromOpen"] == 10.0
    assert overall["MedianMinutesFromOpen"] == 10.0
    assert overall["MinMinutesFromOpen"] == 2
    assert overall["MaxMinutesFromOpen"] == 18

    toyota = statistics[
        (statistics["Scope"] == "TICKER") & (statistics["Ticker"] == "7203.T")
    ].iloc[0]
    assert toyota["Observations"] == 2
    assert toyota["MeanMinutesFromOpen"] == 5.0
    assert toyota["MedianMinutesFromOpen"] == 5.0


def test_distribution_uses_five_minute_buckets_and_percentages():
    _, distribution = summarize_morning_lows(_sample_lows())
    overall = distribution[distribution["Scope"] == "ALL"]

    bucket_0_4 = overall[overall["BucketStartMinute"] == 0].iloc[0]
    bucket_5_9 = overall[overall["BucketStartMinute"] == 5].iloc[0]
    bucket_10_14 = overall[overall["BucketStartMinute"] == 10].iloc[0]
    bucket_15_19 = overall[overall["BucketStartMinute"] == 15].iloc[0]
    bucket_20_24 = overall[overall["BucketStartMinute"] == 20].iloc[0]

    assert bucket_0_4["Count"] == 1
    assert bucket_5_9["Count"] == 1
    assert bucket_10_14["Count"] == 1
    assert bucket_15_19["Count"] == 1
    assert bucket_0_4["Percentage"] == 25.0
    assert bucket_20_24["Count"] == 0
    assert bucket_0_4["ClockRangeJST"] == "09:00-09:04"


def test_distribution_keeps_1130_as_final_bucket():
    lows = pd.DataFrame({"Ticker": ["7203.T"], "MinutesFromOpen": [150]})
    _, distribution = summarize_morning_lows(lows)
    overall = distribution[distribution["Scope"] == "ALL"]
    final_bucket = overall[overall["BucketStartMinute"] == 150].iloc[0]

    assert final_bucket["BucketEndMinute"] == 150
    assert final_bucket["MinutesFromOpenRange"] == "150"
    assert final_bucket["ClockRangeJST"] == "11:30"
    assert final_bucket["Count"] == 1
    assert final_bucket["Percentage"] == 100.0


def test_invalid_minutes_are_removed_before_aggregation():
    lows = pd.DataFrame(
        {
            "Ticker": ["7203.T", "7203.T", "7203.T", "7203.T"],
            "MinutesFromOpen": [-1, 0, 150, 151],
        }
    )
    statistics, _ = summarize_morning_lows(lows)
    overall = statistics[statistics["Scope"] == "ALL"].iloc[0]

    assert overall["Observations"] == 2
    assert overall["MeanMinutesFromOpen"] == 75.0
    assert overall["MedianMinutesFromOpen"] == 75.0
