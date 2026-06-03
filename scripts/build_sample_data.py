"""Generate the bundled sample settlements and write them under data/sample/.

Run once after cloning if you don't have an EIA key:
    python scripts/build_sample_data.py
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from src import simulate

_FILES = {"WTI": "wti_settlements.csv", "HH": "henry_hub_settlements.csv"}


def main() -> None:
    df = simulate.simulate()
    for product, fname in _FILES.items():
        sub = df[df["product"] == product]
        path = config.SAMPLE / fname
        sub.to_csv(path, index=False)
        print(f"wrote {len(sub):,} rows -> {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
