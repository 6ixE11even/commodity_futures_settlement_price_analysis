import pandas as pd

from src import term_structure as ts


def _curve(rows, product="WTI"):
    """rows: list of (trade_date, contract_rank, settle)."""
    return pd.DataFrame([
        {"trade_date": pd.Timestamp(d), "product": product, "contract_rank": r,
         "expiry": "2021-06", "settle": s, "unit": "$/bbl"}
        for d, r, s in rows
    ])


def test_classify_states():
    assert ts.classify(0.05) == "contango"
    assert ts.classify(-0.05) == "backwardation"
    assert ts.classify(0.0) == "flat"


def test_front_month_picks_rank_one():
    df = _curve([
        ("2021-01-04", 1, 100), ("2021-01-04", 2, 98),
        ("2021-01-05", 1, 101), ("2021-01-05", 2, 99),
    ])
    fm = ts.front_month(df, "WTI")
    assert list(fm.values) == [100, 101]


def test_backwardation_when_front_richest():
    df = _curve([("2021-01-04", 1, 100), ("2021-01-04", 2, 98), ("2021-01-04", 3, 96)])
    row = ts.term_structure(df, "WTI").iloc[0]
    assert row["back_rank"] == 3
    assert row["slope"] < 0
    assert row["state"] == "backwardation"


def test_contango_when_deferred_richest():
    df = _curve([("2021-01-04", 1, 100), ("2021-01-04", 2, 102), ("2021-01-04", 3, 105)])
    assert ts.term_structure(df, "WTI").iloc[0]["state"] == "contango"
