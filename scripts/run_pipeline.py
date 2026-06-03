"""Load settlements, scan for anomalies, and write the event list to disk.

    python scripts/run_pipeline.py                 # bundled sample data
    python scripts/run_pipeline.py --source live   # needs EIA_API_KEY
    python scripts/run_pipeline.py --product WTI    # one product only
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import config
from src import anomalies, data_loader


def main() -> None:
    ap = argparse.ArgumentParser(description="Commodity futures settlement anomaly scan")
    ap.add_argument("--source", choices=["sample", "live"], default="sample")
    ap.add_argument("--product", choices=config.PRODUCTS, action="append",
                    help="limit to one product (repeatable); default is all")
    args = ap.parse_args()

    products = args.product or config.PRODUCTS
    df = data_loader.load(args.source, products=products)
    _, events = anomalies.detect_all(df, products)

    out = config.PROCESSED / "anomaly_events.csv"
    events.to_csv(out, index=False)

    n = len(events)
    confirmed = int(events["iso_confirmed"].sum()) if n else 0
    print(f"{n} anomalous settlement events ({config.START[:4]}-{config.END[:4]}); "
          f"{confirmed} corroborated by Isolation Forest\n")
    if n:
        print(events.to_string(index=False))
    print(f"\nsaved -> {out.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
