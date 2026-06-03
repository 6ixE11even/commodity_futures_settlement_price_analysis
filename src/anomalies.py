"""Settlement anomaly detection.

Two independent views, then a consensus:

  1. Z-score rule on the rolling-standardised front-month log return. This is the
     interpretable "the settlement moved N sigma" test.
  2. Isolation Forest over [log return, curve-slope change, rolling vol]. A
     multivariate view that also reacts to curve and volatility dislocations.

The z-score rule defines the flagged events; the forest runs independently and
its agreement is recorded per event. Consecutive flagged days are collapsed into
a single event (a multi-day spike is one event, not five).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import config
from src import term_structure as ts

_FEATURES = ["log_ret", "slope_change", "vol"]


def build_features(df: pd.DataFrame, product: str,
                   window: int = config.ROLL_WINDOW) -> pd.DataFrame:
    front = ts.front_month(df, product)
    slope = ts.term_structure(df, product).set_index("trade_date")["slope"]

    log_ret = np.log(front).diff()
    roll_mean = log_ret.rolling(window, min_periods=window // 3).mean()
    roll_std = log_ret.rolling(window, min_periods=window // 3).std()

    feats = pd.DataFrame({
        "settle": front,
        "log_ret": log_ret,
        "ret_z": (log_ret - roll_mean) / roll_std,
        "vol": log_ret.rolling(window, min_periods=window // 3).std(),
        "slope": slope,
        "slope_change": slope.diff(),
    })
    return feats.dropna(subset=["log_ret"])


def detect(df: pd.DataFrame, product: str,
           z_threshold: float = config.ZSCORE_THRESHOLD,
           contamination: float = config.IFOREST_CONTAMINATION) -> pd.DataFrame:
    feats = build_features(df, product).copy()

    feats["z_flag"] = feats["ret_z"].abs() > z_threshold

    # Independent multivariate view. Standardise first so no single feature
    # dominates the tree splits, then score every day (higher = more isolated).
    X = StandardScaler().fit_transform(feats[_FEATURES].fillna(0.0))
    forest = IsolationForest(contamination=contamination, random_state=config.IFOREST_SEED)
    forest.fit(X)
    feats["iso_flag"] = forest.predict(X) == -1
    feats["iso_score"] = -forest.score_samples(X)

    # Headline anomalies are the z-score exceedances; the forest is a second
    # opinion we record per event rather than a hard gate.
    feats["anomaly"] = feats["z_flag"]
    feats["product"] = product
    return feats


def group_events(feats: pd.DataFrame, gap: int = config.EVENT_GAP_DAYS) -> pd.DataFrame:
    flagged = feats[feats["anomaly"]]
    if flagged.empty:
        return pd.DataFrame(columns=[
            "product", "start", "end", "days", "peak_date",
            "peak_z", "peak_return", "direction", "iso_confirmed",
        ])

    pos = feats.index.get_indexer(flagged.index)
    group = (np.diff(pos, prepend=pos[0]) > gap).cumsum()

    events = []
    for _, chunk in flagged.groupby(group):
        peak = chunk.loc[chunk["ret_z"].abs().idxmax()]
        events.append({
            "product": chunk["product"].iloc[0],
            "start": chunk.index.min().date(),
            "end": chunk.index.max().date(),
            "days": len(chunk),
            "peak_date": peak.name.date(),
            "peak_z": round(float(peak["ret_z"]), 2),
            "peak_return": round(float(peak["log_ret"]), 4),
            "direction": "up" if peak["log_ret"] > 0 else "down",
            # Did the Isolation Forest independently flag any day in the window?
            "iso_confirmed": bool(chunk["iso_flag"].any()),
        })
    return pd.DataFrame(events)


def detect_all(df: pd.DataFrame, products: list[str] | None = None):
    """Run detection across products. Returns (per_day_features, events)."""
    products = products or config.PRODUCTS
    feats = [detect(df, p) for p in products]
    events = [group_events(f) for f in feats]
    events = pd.concat(events, ignore_index=True).sort_values("start").reset_index(drop=True)
    return feats, events
