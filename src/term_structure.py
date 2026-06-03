"""Term-structure metrics: front month, curve slope, and settlement deviation.

Convention: slope is measured front-to-back as (deferred - front) / front.
A positive slope means deferred contracts trade above the front (contango);
negative means the front is richer (backwardation).
"""

from __future__ import annotations

import pandas as pd

import config

_FLAT_BAND = 0.005  # |slope| below this is treated as a flat curve


def pivot_curve(df: pd.DataFrame, product: str) -> pd.DataFrame:
    """Wide panel: rows = trade_date, columns = contract_rank, values = settle."""
    sub = df[df["product"] == product]
    wide = sub.pivot_table(index="trade_date", columns="contract_rank", values="settle")
    return wide.sort_index()


def front_month(df: pd.DataFrame, product: str) -> pd.Series:
    wide = pivot_curve(df, product)
    return wide[1].rename("settle")


def classify(slope: float) -> str:
    if slope > _FLAT_BAND:
        return "contango"
    if slope < -_FLAT_BAND:
        return "backwardation"
    return "flat"


def term_structure(df: pd.DataFrame, product: str, back_rank: int | None = None) -> pd.DataFrame:
    """Front vs deferred spread and curve state for every trade date."""
    wide = pivot_curve(df, product)
    if back_rank is None:
        back_rank = int(wide.columns.max())

    m1, mb = wide[1], wide[back_rank]
    out = pd.DataFrame({
        "trade_date": wide.index,
        "front": m1.values,
        "deferred": mb.values,
        "back_rank": back_rank,
    })
    out["spread"] = out["deferred"] - out["front"]
    out["slope"] = out["spread"] / out["front"]
    out["state"] = out["slope"].map(classify)
    return out


def curve_on(df: pd.DataFrame, product: str, date) -> pd.DataFrame:
    """The full settlement curve as of a single trade date."""
    date = pd.Timestamp(date)
    sub = df[(df["product"] == product) & (df["trade_date"] == date)]
    return sub[["contract_rank", "expiry", "settle"]].sort_values("contract_rank").reset_index(drop=True)


def settlement_deviation(df: pd.DataFrame, product: str,
                         window: int = config.ROLL_WINDOW) -> pd.DataFrame:
    """Front-month level vs its rolling mean, plus the rolling z-score."""
    s = front_month(df, product)
    roll_mean = s.rolling(window, min_periods=window // 3).mean()
    roll_std = s.rolling(window, min_periods=window // 3).std()
    out = pd.DataFrame({
        "trade_date": s.index,
        "settle": s.values,
        "roll_mean": roll_mean.values,
        "roll_std": roll_std.values,
    })
    out["deviation"] = out["settle"] - out["roll_mean"]
    out["z"] = out["deviation"] / out["roll_std"]
    return out
