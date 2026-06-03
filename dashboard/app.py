"""Streamlit dashboard for the settlement analysis.

Run from the project root:
    streamlit run dashboard/app.py

Four views: front-month settlements with flagged anomalies, the term-structure
curve on a chosen date, the whole forward curve over time, and the event log.
"""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from src import anomalies, data_loader
from src import term_structure as ts

st.set_page_config(page_title="Commodity Futures Settlement Analysis", layout="wide")

NAMES = {p: config.EIA_SERIES[p]["name"] for p in config.PRODUCTS}


@st.cache_data
def get_data(source: str) -> pd.DataFrame:
    return data_loader.load(source)


@st.cache_data
def get_events(source: str) -> pd.DataFrame:
    _, events = anomalies.detect_all(get_data(source))
    return events


def snap(dates: pd.DatetimeIndex, target) -> pd.Timestamp:
    """Nearest available trading date to a picked calendar date."""
    pos = dates.get_indexer([pd.Timestamp(target)], method="nearest")[0]
    return dates[pos]


# --- sidebar ------------------------------------------------------------------
st.sidebar.title("Settings")
source = st.sidebar.radio(
    "Data source", ["sample", "live"],
    help="'sample' uses the bundled simulated curve. 'live' pulls EIA data and needs EIA_API_KEY.",
)

df = events = pd.DataFrame()
try:
    df = get_data(source)
    events = get_events(source)
except Exception as exc:  # surface the reason instead of a blank page
    st.error(f"Could not load {source} data: {exc}")
    st.stop()

product = st.sidebar.selectbox("Product", config.PRODUCTS, format_func=NAMES.get)
unit = config.EIA_SERIES[product]["unit"]

panel = df[df["product"] == product]
dates = pd.DatetimeIndex(sorted(panel["trade_date"].unique()))
lo, hi = st.sidebar.slider(
    "Date range",
    min_value=dates.min().to_pydatetime(), max_value=dates.max().to_pydatetime(),
    value=(dates.min().to_pydatetime(), dates.max().to_pydatetime()),
    format="YYYY-MM",
)
window = (panel["trade_date"] >= lo) & (panel["trade_date"] <= hi)
view = panel[window]

with st.sidebar.expander("About the data"):
    st.markdown(
        "Live data is NYMEX/CME WTI and Henry Hub settlements from the EIA API "
        "(front four contracts). The bundled **sample** is a simulated forward "
        "curve (18 contracts) calibrated to real 2021-2025 moves, so the full "
        "term structure renders without an API key."
    )

st.title("Commodity Futures Settlement Price Analysis")
st.caption(f"{NAMES[product]} · settlements in {unit} · {lo:%b %Y} – {hi:%b %Y}")

fm = ts.front_month(view, product)
ev = events[events["product"] == product]
ev = ev[(pd.to_datetime(ev["peak_date"]) >= lo) & (pd.to_datetime(ev["peak_date"]) <= hi)]

c1, c2, c3 = st.columns(3)
c1.metric("Latest front settle", f"{fm.iloc[-1]:,.2f}")
c2.metric("Anomalies in range", len(ev))
c3.metric("Contract months loaded", panel["contract_rank"].nunique())

tab_px, tab_curve, tab_surface, tab_events = st.tabs(
    ["Settlements & anomalies", "Term structure", "Curve over time", "Events"]
)

# --- settlements + flagged anomalies -----------------------------------------
with tab_px:
    fig = go.Figure()
    fig.add_scatter(x=fm.index, y=fm.values, mode="lines",
                    name="Front-month settle", line=dict(color="#1f77b4", width=1.3))
    if not ev.empty:
        peaks = pd.to_datetime(ev["peak_date"])
        fig.add_scatter(
            x=peaks, y=fm.reindex(peaks).values, mode="markers", name="Flagged anomaly",
            marker=dict(color="crimson", size=9, symbol="x"),
            text=ev["peak_z"].map(lambda z: f"z={z}"), hovertemplate="%{x|%Y-%m-%d}<br>%{text}<extra></extra>",
        )
    fig.update_layout(height=430, margin=dict(t=30, b=10),
                      yaxis_title=unit, legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, width="stretch")

    dev = ts.settlement_deviation(view, product)
    figd = go.Figure()
    figd.add_scatter(x=dev["trade_date"], y=dev["z"], mode="lines",
                     line=dict(color="#555", width=1), name="deviation z")
    for level in (config.ZSCORE_THRESHOLD, -config.ZSCORE_THRESHOLD):
        figd.add_hline(y=level, line=dict(color="crimson", dash="dot", width=1))
    figd.update_layout(height=240, margin=dict(t=30, b=10),
                       title="Front-month deviation from rolling mean (z-score)",
                       yaxis_title="z")
    st.plotly_chart(figd, width="stretch")

# --- term-structure curve on chosen dates ------------------------------------
with tab_curve:
    st.write("Compare the settlement curve on two dates to see the structure shift.")
    a, b = st.columns(2)
    d1 = snap(dates, a.date_input("Date A", value=dates.min(),
                                  min_value=dates.min(), max_value=dates.max()))
    d2 = snap(dates, b.date_input("Date B", value=dates.max(),
                                  min_value=dates.min(), max_value=dates.max()))

    fig = go.Figure()
    for d, color in ((d1, "#1f77b4"), (d2, "#d62728")):
        curve = ts.curve_on(df, product, d)
        state = ts.term_structure(df, product).set_index("trade_date").loc[d, "state"]
        fig.add_scatter(x=curve["contract_rank"], y=curve["settle"], mode="lines+markers",
                        name=f"{d:%Y-%m-%d} ({state})", line=dict(color=color),
                        text=curve["expiry"], hovertemplate="M%{x} · %{text}<br>%{y:.2f}<extra></extra>")
    fig.update_layout(height=460, margin=dict(t=30),
                      xaxis_title="contract (1 = front month)", yaxis_title=unit,
                      legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, width="stretch")

# --- forward curve over time (normalised to front) ---------------------------
with tab_surface:
    st.write(
        "Each contract's settle relative to the front month, by month. "
        "Red = deferred richer (contango); blue = front richer (backwardation)."
    )
    wide = ts.pivot_curve(view, product)
    monthly = wide.resample("ME").last()
    ratio = monthly.div(monthly[1], axis=0)
    fig = go.Figure(go.Heatmap(
        z=ratio.T.values, x=monthly.index, y=ratio.columns,
        colorscale="RdBu_r", zmid=1.0, colorbar=dict(title="settle / M1"),
    ))
    fig.update_layout(height=460, margin=dict(t=30),
                      yaxis_title="contract month", xaxis_title="")
    st.plotly_chart(fig, width="stretch")

# --- event log ---------------------------------------------------------------
with tab_events:
    confirmed = int(events["iso_confirmed"].sum())
    st.write(f"**{len(events)}** flagged events across both products; "
             f"**{confirmed}** corroborated by the Isolation Forest.")
    st.dataframe(events, width="stretch", hide_index=True)
