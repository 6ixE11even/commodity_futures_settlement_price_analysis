# Commodity Futures Settlement Price Analysis

Anomaly detection and term-structure analysis on WTI crude and Henry Hub natural
gas futures settlements, 2021–2025. The pipeline pulls daily NYMEX/CME
settlements from the EIA REST API, flags anomalous settlement days with a
z-score + Isolation Forest model, and serves an interactive dashboard for the
forward curve, contango/backwardation regime, and settlement deviations.

The dashboard is a Streamlit + Plotly app (it replaces an earlier Tableau
workbook), so the whole project runs from a single `pip install`.

## What it does

- Pulls ~5 years of WTI and Henry Hub settlements from the EIA Open Data API
  (front four contracts per product).
- Builds the forward curve and the term-structure metrics: front-vs-deferred
  slope, contango/backwardation state, and rolling settlement deviation.
- Flags anomalous settlements two ways — a rolling z-score on front-month
  returns, and an Isolation Forest over `[return, slope change, volatility]`. On
  the bundled data it surfaces **12 events** (11 independently corroborated by
  the forest) that line up with real dislocations: Winter Storm Uri, the
  Nov-2021 Omicron drop, the 2022 Russia/Ukraine oil spike and reversal, the SVB
  selloff, the Apr-2023 OPEC+ surprise cut, and the warm-winter gas collapses.
- Four dashboard views: settlements with flagged anomalies, the term-structure
  curve compared across two dates, the full forward curve over time, and the
  event log.

## Data

Two interchangeable sources sit behind one schema
(`trade_date, product, contract_rank, expiry, settle, unit`):

- **Live** — EIA Open Data API (`RCLC1–4` for WTI, `RNGC1–4` for Henry Hub).
  Real NYMEX/CME settlements; needs a free key in `EIA_API_KEY`. EIA publishes
  the front four contract months.
- **Sample (bundled, default)** — a simulated 18-contract forward curve
  calibrated to the actual 2021–2025 price path and the events above. It exists
  because the free EIA feed is only four contracts deep while the term-structure
  views want the whole curve. The front month still rolls through 77 distinct
  contract months over the window. It is clearly simulated — don't read the
  deep-curve levels as real settlements.

## Project layout

```
config.py               paths, EIA series IDs, detector thresholds
src/
  eia_client.py         EIA v2 REST client
  simulate.py           calibrated sample-curve generator
  data_loader.py        unified loader (sample | live) with caching
  term_structure.py     curve slope, contango/backwardation, deviation
  anomalies.py          z-score + Isolation Forest, event grouping
dashboard/app.py        Streamlit + Plotly dashboard
scripts/
  build_sample_data.py  write the bundled sample CSVs
  run_pipeline.py       run detection, write the event list
tests/                  unit tests
data/sample/            committed sample data
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# 1. (re)generate the bundled sample data
python scripts/build_sample_data.py

# 2. scan for anomalies -> data/processed/anomaly_events.csv
python scripts/run_pipeline.py
python scripts/run_pipeline.py --product WTI     # single product

# 3. launch the dashboard
streamlit run dashboard/app.py
```

To use live EIA data instead of the sample, get a free key at
<https://www.eia.gov/opendata/register.php>, then:

```bash
cp .env.example .env      # paste your key into .env
python scripts/run_pipeline.py --source live
```

## Method notes

- **Returns z-score.** Front-month log returns standardised by a 63-day rolling
  mean and std; a day is flagged when `|z| > 3.2`. This is the interpretable,
  headline detector.
- **Isolation Forest.** Fit on standardised `[log return, slope change, rolling
  vol]` at 5% contamination. It's an independent multivariate view, recorded per
  event as corroboration rather than used as a hard gate — the two methods catch
  overlapping but not identical things.
- Consecutive flagged days within one trading week collapse into a single event.
- **Term structure.** Slope is `(deferred − front) / front`; positive is
  contango, negative is backwardation, with a small flat band around zero.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```
