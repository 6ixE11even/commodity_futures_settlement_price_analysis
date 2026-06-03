"""Load settlement data from the bundled sample or the live EIA API.

Both paths return the same tidy schema so the rest of the pipeline doesn't care
where the data came from:

    trade_date | product | contract_rank | expiry | settle | unit
"""

from __future__ import annotations

import pandas as pd

import config
from src import eia_client

SCHEMA = ["trade_date", "product", "contract_rank", "expiry", "settle", "unit"]

_SAMPLE_FILES = {"WTI": "wti_settlements.csv", "HH": "henry_hub_settlements.csv"}


def _series_to_rank(series_id: str) -> int:
    # RCLC1 -> 1, RNGC4 -> 4, etc.
    return int(series_id[-1])


def load_sample(products: list[str] | None = None) -> pd.DataFrame:
    products = products or config.PRODUCTS
    frames = []
    for p in products:
        path = config.SAMPLE / _SAMPLE_FILES[p]
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run scripts/build_sample_data.py first.")
        frames.append(pd.read_csv(path, parse_dates=["trade_date"]))
    return pd.concat(frames, ignore_index=True)


def load_live(products: list[str] | None = None, start: str | None = None,
              end: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    products = products or config.PRODUCTS
    frames = []
    for p in products:
        cache = config.RAW / f"{p.lower()}_live.csv"
        if use_cache and cache.exists():
            frames.append(pd.read_csv(cache, parse_dates=["trade_date"]))
            continue

        raw = eia_client.fetch_product(p, start, end)
        raw["product"] = p
        raw["contract_rank"] = raw["series"].map(_series_to_rank)
        raw["unit"] = config.EIA_SERIES[p]["unit"]
        # EIA's generic "contract N" series aren't tagged with a delivery month.
        raw["expiry"] = pd.NA
        df = raw[SCHEMA]
        df.to_csv(cache, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load(source: str = "sample", **kwargs) -> pd.DataFrame:
    if source == "sample":
        return load_sample(kwargs.get("products"))
    if source == "live":
        return load_live(**kwargs)
    raise ValueError(f"unknown source {source!r} (expected 'sample' or 'live')")
