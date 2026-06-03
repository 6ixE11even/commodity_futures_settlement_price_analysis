"""Client for the EIA Open Data v2 API.

Pulls daily NYMEX (CME Group) futures settlement prices for WTI crude and
Henry Hub natural gas. EIA publishes the front four contract months for each,
which is enough to characterise the short end of the curve. A free API key is
required and is read from the EIA_API_KEY environment variable.
"""

from __future__ import annotations

import time

import pandas as pd
import requests

import config

_TIMEOUT = 30
_MAX_RETRIES = 3
_PAGE = 5000  # EIA returns at most 5000 rows per request


class EIAError(RuntimeError):
    pass


def _request(route: str, params: dict) -> dict:
    url = f"{config.EIA_BASE}/{route}/data/"
    for attempt in range(_MAX_RETRIES):
        resp = requests.get(url, params=params, timeout=_TIMEOUT)
        if resp.ok:
            return resp.json()["response"]
        # 429 = rate limited. Back off and retry; anything else is fatal.
        if resp.status_code == 429 and attempt < _MAX_RETRIES - 1:
            time.sleep(2 ** attempt)
            continue
        raise EIAError(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
    raise EIAError("retries exhausted")


def _fetch_one(route: str, series_id: str, start: str, end: str) -> pd.DataFrame:
    params = {
        "api_key": config.EIA_API_KEY,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": series_id,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": _PAGE,
    }
    rows: list[dict] = []
    while True:
        payload = _request(route, params)
        rows.extend(payload["data"])
        params["offset"] += _PAGE
        if params["offset"] >= int(payload["total"]):
            break
    if not rows:
        return pd.DataFrame(columns=["trade_date", "settle", "series"])
    df = pd.DataFrame(rows)[["period", "value"]]
    df.columns = ["trade_date", "settle"]
    df["series"] = series_id
    return df


def fetch_product(product: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Return tidy daily settlements for one product across its EIA series.

    Output columns: trade_date, series, settle.
    """
    if not config.EIA_API_KEY:
        raise EIAError("EIA_API_KEY is not set. Get a free key at eia.gov/opendata or use the sample data.")

    spec = config.EIA_SERIES[product]
    start = start or config.START
    end = end or config.END

    frames = [_fetch_one(spec["route"], sid, start, end) for sid in spec["series"]]
    out = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    out["trade_date"] = pd.to_datetime(out["trade_date"])
    out["settle"] = pd.to_numeric(out["settle"], errors="coerce")
    out = out.dropna(subset=["settle"])
    return out.sort_values(["series", "trade_date"]).reset_index(drop=True)
