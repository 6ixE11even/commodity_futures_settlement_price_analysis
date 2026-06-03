"""Project configuration: paths, data sources, and detector thresholds."""

from pathlib import Path
import os

try:
    # Optional: load EIA_API_KEY (and anything else) from a local .env file.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"            # cached live API pulls (not committed)
SAMPLE = DATA / "sample"      # bundled simulated dataset (committed)
PROCESSED = DATA / "processed"  # pipeline outputs (not committed)

for _d in (RAW, SAMPLE, PROCESSED):
    _d.mkdir(parents=True, exist_ok=True)

# --- EIA Open Data API (https://www.eia.gov/opendata/) -----------------------
# Free key, read from the environment so it never lands in the repo.
EIA_API_KEY = os.environ.get("EIA_API_KEY")
EIA_BASE = "https://api.eia.gov/v2"

# NYMEX/CME front-of-curve daily settlements published by EIA (contracts 1-4).
# WTI crude is quoted in $/bbl, Henry Hub natural gas in $/MMBtu.
EIA_SERIES = {
    "WTI": {
        "name": "WTI Crude Oil",
        "route": "petroleum/pri/fut",
        "series": ["RCLC1", "RCLC2", "RCLC3", "RCLC4"],
        "unit": "$/bbl",
    },
    "HH": {
        "name": "Henry Hub Natural Gas",
        "route": "natural-gas/pri/fut",
        "series": ["RNGC1", "RNGC2", "RNGC3", "RNGC4"],
        "unit": "$/MMBtu",
    },
}

PRODUCTS = list(EIA_SERIES)  # ["WTI", "HH"]

# Analysis window: roughly five years of settlements.
START = "2021-01-01"
END = "2025-12-31"

# --- Anomaly detection --------------------------------------------------------
ROLL_WINDOW = 63          # ~one trading quarter for rolling mean/std
ZSCORE_THRESHOLD = 3.2    # |z| on the rolling-standardised front-month return
IFOREST_CONTAMINATION = 0.05  # fraction of days the forest treats as outliers
IFOREST_SEED = 7
EVENT_GAP_DAYS = 5        # flags within this many trading days = one event
