import numpy as np
import pandas as pd

from src import anomalies


def _front_df(prices, product="WTI"):
    """Build a 2-contract panel from a front-month price path."""
    dates = pd.bdate_range("2021-01-01", periods=len(prices))
    rows = []
    for d, p in zip(dates, prices):
        rows.append({"trade_date": d, "product": product, "contract_rank": 1,
                     "expiry": "x", "settle": p, "unit": "u"})
        rows.append({"trade_date": d, "product": product, "contract_rank": 2,
                     "expiry": "x", "settle": p * 1.01, "unit": "u"})
    return pd.DataFrame(rows)


def test_zscore_flags_obvious_spike():
    rng = np.random.default_rng(0)
    prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 300)))
    prices[200] *= 1.20  # a 20% one-day jump should never be normal
    feats = anomalies.detect(_front_df(prices), "WTI")

    spike_day = pd.bdate_range("2021-01-01", periods=300)[200]
    assert feats.loc[spike_day, "z_flag"]
    assert feats.loc[spike_day, "anomaly"]


def test_events_separate_when_far_apart():
    # Flat baseline with two isolated one-day blips far apart -> two events.
    prices = np.full(400, 100.0)
    prices[150] = 118.0
    prices[300] = 118.0
    events = anomalies.group_events(anomalies.detect(_front_df(prices), "WTI"))

    assert len(events) == 2
    assert events["days"].max() < 10


def test_consecutive_flags_collapse_to_one_event():
    # A single blip moves price up then back; the two adjacent flagged days
    # (the jump and the reversal) should fold into one event.
    prices = np.full(300, 100.0)
    prices[150] = 118.0
    events = anomalies.group_events(anomalies.detect(_front_df(prices), "WTI"))

    assert len(events) == 1
    assert events.iloc[0]["days"] == 2
