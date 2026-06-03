"""Generate a realistic sample forward curve for WTI and Henry Hub.

The live pipeline pulls real settlements from EIA, but that feed only exposes
the front four contracts and needs an API key. To keep the repo runnable out of
the box (and to give the dashboard a full curve to draw), this module builds a
calibrated simulation instead:

  * the front-month path follows hand-set anchor levels through 2021-2025 and is
    nudged by the macro moves that actually happened (the Nov-2021 Omicron drop,
    the 2022 Russia/Ukraine oil spike, Winter Storm Uri and the 2022 gas run-up,
    the Apr-2023 OPEC+ cut, etc.);
  * WTI carries a level-dependent backwardation/contango slope;
  * Henry Hub carries a seasonal winter premium across the curve.

Output is a tidy frame: trade_date, product, contract_rank, expiry, settle, unit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

SIM_SEED = 20240131
N_CONTRACTS = 18  # forward months per product

# Approximate front-month settle at a handful of anchor dates ($/bbl, $/MMBtu).
# Daily levels are interpolated between these.
_WTI_ANCHORS = {
    "2021-01-01": 48, "2021-03-01": 60, "2021-07-01": 73, "2021-10-01": 78,
    "2021-12-01": 71, "2022-02-01": 90, "2022-03-15": 108, "2022-06-01": 115,
    "2022-09-01": 88, "2022-12-01": 78, "2023-03-01": 75, "2023-06-01": 70,
    "2023-09-01": 85, "2023-12-01": 73, "2024-04-01": 83, "2024-09-01": 72,
    "2024-12-01": 70, "2025-04-01": 66, "2025-09-01": 68, "2025-12-31": 70,
}
_HH_ANCHORS = {
    "2021-01-01": 2.6, "2021-04-01": 2.6, "2021-07-01": 3.6, "2021-10-01": 5.6,
    "2021-12-01": 4.0, "2022-02-01": 4.6, "2022-05-01": 7.0, "2022-08-15": 9.0,
    "2022-10-01": 6.5, "2022-12-01": 5.5, "2023-02-01": 2.6, "2023-06-01": 2.2,
    "2023-09-01": 2.6, "2023-12-01": 2.5, "2024-03-01": 1.6, "2024-07-01": 2.2,
    "2024-11-01": 2.8, "2025-01-01": 3.6, "2025-04-01": 3.8, "2025-08-01": 3.2,
    "2025-12-31": 3.6,
}

# One-off log-return shocks layered on top of the trend (they decay over a few
# days). These line up with real settlement dislocations.
_WTI_SHOCKS = {
    "2021-11-26": -0.13,   # Omicron demand scare
    "2022-02-24": 0.05,    # invasion begins
    "2022-03-07": 0.07,    # spike toward $120s
    "2022-03-09": -0.11,   # sharp reversal
    "2023-04-03": 0.06,    # surprise OPEC+ cut
    "2024-04-15": 0.05,    # Middle East escalation
}
_HH_SHOCKS = {
    "2021-02-17": 0.13,    # Winter Storm Uri
    "2021-02-22": -0.09,   # post-Uri normalisation
    "2021-09-28": 0.09,    # global gas crunch
    "2022-01-27": 0.15,    # expiry-day squeeze
    "2022-08-22": 0.08,    # summer storage scare
    "2023-01-04": -0.12,   # warm-winter collapse
    "2025-01-21": 0.10,    # January cold snap
}

# Henry Hub winter premium by delivery month (Jan/Feb/Dec rich, shoulder cheap).
_HH_SEASONAL = {
    1: 1.18, 2: 1.14, 3: 1.02, 4: 0.93, 5: 0.92, 6: 0.94,
    7: 0.97, 8: 0.99, 9: 0.95, 10: 0.99, 11: 1.06, 12: 1.15,
}

_DAILY_VOL = {"WTI": 0.013, "HH": 0.022}
_ROUND = {"WTI": 2, "HH": 3}


def _trend(anchors: dict, bdates: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series(anchors)
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    merged = s.reindex(s.index.union(bdates)).interpolate("time")
    return merged.reindex(bdates).ffill().bfill()


def _front_path(anchors: dict, shocks: dict, sigma: float,
                bdates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.Series:
    n = len(bdates)
    trend = _trend(anchors, bdates).to_numpy()

    # Stationary AR(1) wiggle around the trend (in log space).
    rho = 0.9
    eps = rng.normal(0.0, sigma, n)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = rho * u[i - 1] + eps[i]

    # Event shocks placed on the nearest trading day, decaying afterwards.
    impulse = np.zeros(n)
    for date, val in shocks.items():
        pos = bdates.get_indexer([pd.Timestamp(date)], method="nearest")[0]
        impulse[pos] += val
    decay, bump = 0.6, np.zeros(n)
    for i in range(1, n):
        bump[i] = decay * bump[i - 1] + impulse[i]

    return pd.Series(np.exp(np.log(trend) + u + bump), index=bdates)


def _wti_curve(front: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    f = front.to_numpy()
    ranks = np.arange(1, N_CONTRACTS + 1)
    # Backwardation when prices are high, mild contango when low.
    slope = np.clip(-0.004 * (f - 65.0), -0.012, 0.004)
    factor = 1.0 + np.outer(slope, ranks - 1)
    noise = rng.normal(0.0, 0.002, (len(f), N_CONTRACTS))
    noise[:, 0] = 0.0  # keep M1 equal to the front path
    settle = f[:, None] * factor * np.exp(noise)
    return pd.DataFrame(settle, index=front.index, columns=ranks)


def _hh_curve(front: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    ranks = np.arange(1, N_CONTRACTS + 1)
    front_month = front.index.to_period("M") + 1
    front_factor = front_month.month.map(_HH_SEASONAL).to_numpy(dtype=float)
    deseason = front.to_numpy() / front_factor

    seasonal = np.empty((len(front), N_CONTRACTS))
    for j, k in enumerate(ranks):
        months = (front.index.to_period("M") + k).month
        seasonal[:, j] = months.map(_HH_SEASONAL).to_numpy(dtype=float)
    noise = rng.normal(0.0, 0.0025, (len(front), N_CONTRACTS))
    noise[:, 0] = 0.0
    settle = deseason[:, None] * seasonal * np.exp(noise)
    return pd.DataFrame(settle, index=front.index, columns=ranks)


def _to_long(product: str, wide: pd.DataFrame) -> pd.DataFrame:
    wide = wide.round(_ROUND[product])
    long = (
        wide.rename_axis("trade_date")
        .reset_index()
        .melt(id_vars="trade_date", var_name="contract_rank", value_name="settle")
    )
    long["contract_rank"] = long["contract_rank"].astype(int)
    long["expiry"] = (
        long["trade_date"].dt.to_period("M") + long["contract_rank"]
    ).astype(str)
    long["product"] = product
    long["unit"] = config.EIA_SERIES[product]["unit"]
    return long


def simulate(start: str = config.START, end: str = config.END,
             seed: int = SIM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bdates = pd.bdate_range(start, end)

    wti = _wti_curve(
        _front_path(_WTI_ANCHORS, _WTI_SHOCKS, _DAILY_VOL["WTI"], bdates, rng), rng
    )
    hh = _hh_curve(
        _front_path(_HH_ANCHORS, _HH_SHOCKS, _DAILY_VOL["HH"], bdates, rng), rng
    )

    out = pd.concat([_to_long("WTI", wti), _to_long("HH", hh)], ignore_index=True)
    cols = ["trade_date", "product", "contract_rank", "expiry", "settle", "unit"]
    return out[cols].sort_values(["product", "trade_date", "contract_rank"]).reset_index(drop=True)
