import os
import json
import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
import pydeck as pdk
from pathlib import Path
from datetime import date, datetime
from sqlalchemy import create_engine, text


#reads st.secrets first (how Streamlit Community Cloud's secrets UI works),
#falling back to a plain env var, then a local-dev default - so the same
#code runs unchanged locally and once deployed.
def get_config(key: str, default: str) -> str:
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)


API_URL = get_config("API_URL", "http://localhost:8000")
DB_URL = get_config(
    "DATABASE_URL",
    "postgresql://climate:climate123@localhost:5433/climate_dw"
)
REGRESSOR_METRICS_PATH = (Path(__file__).resolve().parent.parent
                           / "modeling" / "saved_models"
                           / "xgb_tmax_regressor_h3_metrics.json")

# Live Forecast only offers stations that reported within this many days -
# a rolling window rather than a hardcoded year, so it stays correct as
# time passes and whenever the data gets refreshed.
FORECAST_MAX_AGE_DAYS = 60

# ...and that have at least this many distinct years of real history.
# ECCC's near-real-time feed (ingestion/fetch_recent.py) can surface a
# station that has never had a full GHCN download - it'd have only a
# couple of weeks of data, so its climatology/p95 threshold would be
# based on ~1 sample per day-of-year and be statistically meaningless.
MIN_HISTORY_YEARS = 5

# Station map default view - zoomed out to all of Canada; users pan/zoom
# in themselves toward whichever region they want.
MAP_CENTER_LAT = 56.1304
MAP_CENTER_LON = -106.3468
MAP_DEFAULT_ZOOM = 2.8

#page config
st.set_page_config(
    page_title="Canadian Extreme Heat Predictor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

#trim Streamlit's default top whitespace above the sidebar title and above
#the tabs. The default header bar (Deploy / menu) reserves space that only
#sits over the main content, not the sidebar, so equal padding alone would
#leave the two misaligned - hide that header entirely, then give both
#panels the same small top padding so the title and tabs line up.
#the sidebar title uses the same orange-red as Streamlit's active-tab
#indicator (its default theme's primaryColor, #FF4B4B - no custom theme
#is configured in this project, so that's the color actually shown)
#"Station context", "Day by day outlook", and the tab labels are unified
#to the same larger size (1.4rem) - bigger than Streamlit's defaults for
#each, but still smaller than the sidebar title's h1 size, so the title
#stays visually the largest element on the page.
st.markdown("""
    <style>
        header[data-testid="stHeader"] { display: none; }
        .block-container { padding-top: 1rem; }
        section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
        section[data-testid="stSidebar"] h1 { color: #FF4B4B; }
        .section-heading { font-size: 1.4rem; font-weight: 600; margin: 0.5rem 0; }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.4rem;
        }
    </style>
""", unsafe_allow_html=True)


#cached so it only runs once.
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)


#Load all stations from API.
@st.cache_data
def load_stations():
    engine = get_engine()
    query = text("""
        SELECT
            s.station_id,
            s.name,
            s.province,
            s.lat,
            s.lon,
            s.elevation,
            s.first_year,
            s.last_year
        FROM dim_stations s
        WHERE EXISTS (
            SELECT 1 FROM features_daily f
            WHERE f.station_id = s.station_id
        )
        ORDER BY s.province, s.name
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


#Stations reporting within the last FORECAST_MAX_AGE_DAYS days AND with at
#least MIN_HISTORY_YEARS of real history - the only ones a live forecast
#can be honestly anchored on (recent enough to be "live", enough history
#for climatology/p95 to mean anything).
@st.cache_data
def load_forecast_ready_stations(max_age_days: int = FORECAST_MAX_AGE_DAYS,
                                  min_history_years: int = MIN_HISTORY_YEARS):
    engine = get_engine()
    query = text("""
        SELECT
            s.station_id,
            s.name,
            s.province,
            s.lat,
            s.lon,
            s.elevation,
            s.first_year,
            s.last_year
        FROM dim_stations s
        JOIN features_latest fl ON fl.station_id = s.station_id
        JOIN (
            SELECT station_id, count(DISTINCT year) AS n_years
            FROM features_daily_base
            GROUP BY station_id
        ) hist ON hist.station_id = s.station_id
        WHERE CURRENT_DATE - fl.date_id <= :max_age_days
          AND hist.n_years >= :min_history_years
        ORDER BY s.province, s.name
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={
            "max_age_days": max_age_days,
            "min_history_years": min_history_years,
        })


#Load the 60 calendar days up to and including anchor_date (the live
#as_of_date, or the Historical Explorer's selected_date) - filtered by
#actual date range, not just "the last 60 rows with data". A station can
#have a long reporting gap (rows exist, just not within 60 real days of
#each other); grabbing the last 60 available rows regardless of gap size
#would pull in data from years earlier, and the chart would draw one
#straight line across the whole gap since Plotly just connects whatever
#points are in the dataframe.
@st.cache_data
def load_station_history(station_id: str, anchor_date: date):
    engine = get_engine()
    query = text("""
        SELECT
            date_id,
            tmax_c,
            clim_tmax_mean,
            p95_tmax,
            is_extreme_h1,
            tmax_anomaly
        FROM features_daily
        WHERE station_id = :sid
          AND date_id BETWEEN :anchor_date - INTERVAL '60 days' AND :anchor_date
        ORDER BY date_id
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"sid": station_id, "anchor_date": anchor_date})


#Station-level stats for context panel.
@st.cache_data
def load_station_context(station_id: str, year: int):
    engine = get_engine()
    query = text("""
        SELECT
            COUNT(*) FILTER (WHERE is_extreme_h1 = 1
                AND EXTRACT(year FROM date_id) = :year)
                AS extreme_this_year,
            COUNT(*) FILTER (WHERE is_extreme_h1 = 1
                AND EXTRACT(year FROM date_id) = :year
                AND EXTRACT(month FROM date_id) = EXTRACT(month FROM CURRENT_DATE))
                AS extreme_this_month,
            MAX(tmax_c) AS record_tmax,
            MIN(tmax_c) AS record_tmin
        FROM features_daily
        WHERE station_id = :sid
    """)
    with engine.connect() as conn:
        result = conn.execute(query,
                              {"sid": station_id, "year": year})
        return result.mappings().first()


#Actual min/max backtest-able date - queried rather than hardcoded so the
#Historical Explorer's date picker is correct whether pointed at the full
#local database or a deployment with a trimmed history window.
@st.cache_data
def load_date_range():
    engine = get_engine()
    query = text("SELECT MIN(date_id) AS min_date, MAX(date_id) AS max_date FROM features_daily")
    with engine.connect() as conn:
        row = conn.execute(query).mappings().first()
    return row["min_date"], row["max_date"]


#Load the regressor's own validation MAE per horizon, so predicted
#temperatures can be shown with an honest error bar instead of a bare number.
@st.cache_data
def load_regressor_mae():
    try:
        with open(REGRESSOR_METRICS_PATH) as f:
            metrics = json.load(f)
        by_horizon = metrics["test"]["by_horizon"]
        return {int(h[1:]): v["mae"] for h, v in by_horizon.items()}
    except (FileNotFoundError, KeyError):
        return {}


#map a probability to a risk band (colour / label / message)
def risk_band(prob: float):
    if prob >= 0.75:
        return "#ff4444", "HIGH RISK",     "Extreme heat very likely"
    if prob >= 0.5:
        return "#ff8800", "ELEVATED RISK", "Conditions favour extreme heat"
    if prob >= 0.3:
        return "#ffcc00", "MODERATE RISK", "Some chance of extreme heat"
    return "#00cc44", "LOW RISK", "Extreme heat unlikely"


#Clickable map of every station in stations_df, zoomed out to all of Canada.
#Uses pydeck/deck.gl (GPU-accelerated, free CARTO "dark" basemap - no
#Mapbox token needed) rather than Plotly's Scattermapbox, for a more
#polished look. Clicking a marker is read by render_station_picker (same
#map_key) to sync the province/station dropdowns - this function only
#draws the map.
def map_layer_id(map_key: str) -> str:
    return f"{map_key}_layer"


def render_station_map(stations_df: pd.DataFrame, map_key: str,
                        selected_station_id: str = None):
    df = stations_df.copy()
    is_selected = df["station_id"] == selected_station_id
    # pixel-based radius (not meters) so markers stay a sensible, clickable
    # size whether zoomed out to all of Canada or into a single city
    df["radius"] = is_selected.map({True: 12, False: 6})
    df["color"]  = is_selected.map({
        True:  [255, 68, 68, 230],
        False: [64, 170, 255, 170],
    })

    layer = pdk.Layer(
        "ScatterplotLayer",
        id=map_layer_id(map_key),
        data=df,
        get_position=["lon", "lat"],
        get_radius="radius",
        radius_units="pixels",
        radius_min_pixels=4,
        radius_max_pixels=18,
        get_fill_color="color",
        get_line_color=[255, 255, 255, 90],
        line_width_min_pixels=1,
        stroked=True,
        pickable=True,
        auto_highlight=True,
    )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=pdk.ViewState(
            latitude=MAP_CENTER_LAT,
            longitude=MAP_CENTER_LON,
            zoom=MAP_DEFAULT_ZOOM,
        ),
        map_style="dark",
        tooltip={"text": "{name}"},
    )

    st.pydeck_chart(deck, on_select="rerun", selection_mode="single-object",
                     key=map_key, height=380)


#Live Forecast and Historical Explorer draw from different station sets
#(fresh-only vs. all), so each gets its own province/station picker rather
#than sharing one in the sidebar. key_prefix keeps the two sets of widgets
#(and the map) from colliding. Clicking a station on the map updates the
#dropdowns below it, and vice versa.
def render_station_picker(stations_df: pd.DataFrame, key_prefix: str,
                           default_province: str = "BC"):
    map_key         = f"{key_prefix}_map"
    province_key    = f"{key_prefix}_province"
    station_key     = f"{key_prefix}_station"
    last_synced_key = f"{key_prefix}_map_last_synced"

    # A widget's session_state entry is already updated by the time the
    # script reruns after the user interacts with it, so this map click
    # (if any) is visible here even before render_station_map() is
    # called again further down - only sync on a NEW click (not the
    # stale selection Streamlit keeps around from the last one) so a
    # manual dropdown change after a map click isn't immediately
    # overwritten by the map's still-remembered previous selection.
    map_state = st.session_state.get(map_key)
    selection = (map_state or {}).get("selection", {}) or {}
    clicked_rows = (selection.get("objects", {}) or {}).get(map_layer_id(map_key), [])
    if clicked_rows:
        clicked_station_id = clicked_rows[0]["station_id"]
        if clicked_station_id != st.session_state.get(last_synced_key):
            match = stations_df[stations_df["station_id"] == clicked_station_id]
            if not match.empty:
                st.session_state[province_key] = match.iloc[0]["province"]
                st.session_state[station_key]  = match.iloc[0]["name"]
            st.session_state[last_synced_key] = clicked_station_id

    provinces = sorted(stations_df["province"].dropna().unique())
    province = st.selectbox("Province", provinces,
                             index=provinces.index(default_province)
                             if default_province in provinces else 0,
                             key=province_key)

    filtered = stations_df[stations_df["province"] == province]
    station_options = filtered["name"].tolist()
    station_ids     = filtered["station_id"].tolist()

    # A stored selection from a different province (e.g. the map click
    # synced one province+station together, then the user manually
    # switched province afterwards) won't be in today's options - reset
    # it rather than let the selectbox choke on an invalid stored value.
    if st.session_state.get(station_key) not in station_options and station_options:
        st.session_state[station_key] = station_options[0]

    station_name = st.selectbox("Station", station_options,
                                 key=station_key)
    station_id   = station_ids[station_options.index(station_name)]

    render_station_map(stations_df, map_key, selected_station_id=station_id)

    return station_id, station_name


def render_sidebar(num_fresh_stations: int, num_all_stations: int):
    st.sidebar.title("Canadian Extreme Heat Predictor")
    st.sidebar.markdown(
        "Predicts extreme heat risk and expected daily high temperatures "
        "1 to 3 days ahead for Canadian weather stations, using an XGBoost "
        "model trained on decades of historical climate records."
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Model: XGBoost |\n Data: NOAA GHCN Daily and ECCC near real time obs\n\n"
        f"Live Forecast covers the {num_fresh_stations} stations that have "
        f"reported within the last {FORECAST_MAX_AGE_DAYS} days and have "
        f"{MIN_HISTORY_YEARS}+ years of history. "
        f"Historical Explorer covers all {num_all_stations} stations."
    )


#headline card: extreme heat somewhere in the next 3 days
def render_window_card(window: dict):
    prob = window["probability"]
    color, label, msg = risk_band(prob)

    st.markdown(f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 14px; letter-spacing: 1px;
                        color: #888;">EXTREME HEAT WITHIN 3 DAYS</div>
            <div style="font-size: 28px; font-weight: bold;
                        color: {color};">{label}</div>
            <div style="font-size: 48px; font-weight: bold;
                        margin: 8px 0;">{prob*100:.0f}%</div>
            <div style="font-size: 16px; color: #666;">{msg} in the next 3 days</div>
        </div>
    """, unsafe_allow_html=True)


#per-day strip: one small card for each of the next 3 days.
#show_actual=True (Historical Explorer) shows what really happened;
#show_actual=False (Live Forecast) shows the regressor's predicted temperature.
def render_horizon_strip(horizons: list, show_actual: bool = True,
                          mae_by_horizon: dict = None):
    st.markdown('<p class="section-heading">Day by day outlook</p>', unsafe_allow_html=True)
    cols = st.columns(len(horizons))
    for col, h in zip(cols, horizons):
        prob = h["probability"]
        color, label, _ = risk_band(prob)
        day_label = datetime.strptime(h["date"], "%Y-%m-%d").strftime("%a %b %d")

        if show_actual:
            if h["actual"] is None:
                bottom_line = "actual: —"
            elif h["actual"] == 1:
                bottom_line = "actual: Extreme"
            else:
                bottom_line = "actual: Normal"
        else:
            predicted = h.get("predicted_tmax")
            mae = (mae_by_horizon or {}).get(h["horizon"])
            if predicted is None:
                bottom_line = "predicted: —"
            elif mae is not None:
                bottom_line = f"Predicted: {predicted:.1f}°C (± {mae:.1f}°C)"
            else:
                bottom_line = f"Predicted: {predicted:.1f}°C"

        with col:
            st.markdown(f"""
                <div style="
                    background: {color}18;
                    border: 1px solid {color};
                    border-radius: 10px;
                    padding: 14px;
                    text-align: center;
                ">
                    <div style="font-size: 13px; color: #888;">
                        Day +{h['horizon']} · {day_label}</div>
                    <div style="font-size: 26px; font-weight: bold;
                                color: {color};">{prob*100:.0f}%</div>
                    <div style="font-size: 12px; color: #888;">{label}</div>
                    <div style="font-size: 12px; color: #666;
                                margin-top: 6px;">{bottom_line}</div>
                </div>
            """, unsafe_allow_html=True)


def render_history_chart(history_df: pd.DataFrame,
                          p95_tmax: float,
                          anchor_date: date,
                          horizons: list,
                          forecast_field: str = "future_tmax",
                          forecast_label: str = "Next 3 days (actual)"):
    fig = go.Figure()

    # actual tmax line
    fig.add_trace(go.Scatter(
        x=history_df["date_id"],
        y=history_df["tmax_c"],
        name="Daily Tmax",
        line=dict(color="#4a90d9", width=2),
        fill="tozeroy",
        fillcolor="rgba(74, 144, 217, 0.1)"
    ))

    # climatology mean
    fig.add_trace(go.Scatter(
        x=history_df["date_id"],
        y=history_df["clim_tmax_mean"],
        name="Historical average",
        line=dict(color="#888", width=1, dash="dot")
    ))

    # p95 threshold line
    fig.add_hline(
        y=p95_tmax,
        line_color="red",
        line_dash="dash",
        annotation_text=f"Extreme threshold ({p95_tmax:.1f}°C)",
        annotation_position="top right"
    )

    # highlight the anchor date (selected date, or latest real data for forecasts)
    fig.add_vline(
        x=anchor_date.strftime("%Y-%m-%d"),
        line_color="orange",
        line_dash="dash"
    )

    # forecast points for the next 3 days
    fc = [h for h in horizons if h.get(forecast_field) is not None]
    if fc:
        fig.add_trace(go.Scatter(
            x=[h["date"] for h in fc],
            y=[h[forecast_field] for h in fc],
            name=forecast_label,
            mode="markers",
            marker=dict(color="#ff8800", size=10, symbol="diamond"),
        ))

    fig.update_layout(
        title="60 day temperature history and 3 day outlook",
        xaxis_title="Date",
        yaxis_title="Max Temperature (°C)",
        legend=dict(orientation="h", y=-0.35, yanchor="top"),
        height=380,
        margin=dict(t=40, b=75)
    )

    st.plotly_chart(fig, use_container_width=True)


def render_station_context(station_id: str, year: int):
    ctx = load_station_context(station_id, year)
    if ctx:
        st.markdown('<p class="section-heading">Station context</p>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Extreme days this year",
                  ctx["extreme_this_year"] or 0)
        c2.metric("Extreme days this month",
                  ctx["extreme_this_month"] or 0)
        c3.metric("Record high at station",
                  f"{ctx['record_tmax']}°C")
        c4.metric("Record low at station",
                  f"{ctx['record_tmin']}°C")


#Live Forecast tab: genuinely forward-looking, anchored on each station's
#most recent real observation (no date picker — the future can't be typed in).
def render_forecast_tab(fresh_stations_df: pd.DataFrame):
    if fresh_stations_df.empty:
        st.warning(
            f"No stations have reported within the last "
            f"{FORECAST_MAX_AGE_DAYS} days. Refresh the data (see README) "
            f"or use the Historical Explorer tab instead."
        )
        return

    station_id, station_name = render_station_picker(fresh_stations_df, "forecast")

    st.caption(
        f"Station: **{station_name}**. Forecasts 1-3 days ahead from each "
        f"station's most recent NOAA report (only stations reporting within "
        f"the last {FORECAST_MAX_AGE_DAYS} days are listed here). Data is "
        f"refreshed manually, so the anchor date below may still lag behind "
        f"today's real calendar date."
    )

    with st.spinner("Loading forecast..."):
        resp = requests.get(f"{API_URL}/forecast",
                             params={"station_id": station_id})

    if resp.status_code == 404:
        st.error(f"No data available for {station_name}.")
        return

    result = resp.json()
    as_of_date = datetime.strptime(result["as_of_date"], "%Y-%m-%d").date()
    age = result["data_age_days"]

    age_msg = (f"Latest data for **{station_name}**: **{as_of_date}** "
               f"({age} day{'s' if age != 1 else ''} old)")
    if age > 5:
        st.warning(age_msg + " This station hasn't reported recently; "
                   "the forecast below is only as current as this date.")
    else:
        st.info(age_msg)

    window   = result["window"]
    horizons = result["horizons"]
    mae_by_horizon = load_regressor_mae()

    col1, col2 = st.columns([1, 1])
    with col1:
        render_window_card(window)
    with col2:
        st.markdown("### Station readings")
        m1, m2 = st.columns(2)
        m1.metric("Latest Tmax", f"{result['todays_tmax']}°C",
                  delta=f"{result['tmax_anomaly']:+.1f}°C vs normal")
        m2.metric("Extreme threshold", f"{result['p95_tmax']:.1f}°C")

        m3, m4 = st.columns(2)
        m3.metric("3-day risk", f"{window['probability']*100:.1f}%")
        m4.metric("Predicted high (tomorrow)",
                  f"{horizons[0]['predicted_tmax']:.1f}°C")

    # per-day strip
    st.markdown("---")
    render_horizon_strip(horizons, show_actual=False,
                          mae_by_horizon=mae_by_horizon)

    # history chart
    st.markdown("---")
    history = load_station_history(station_id, as_of_date)
    if not history.empty:
        render_history_chart(history,
                              result["p95_tmax"],
                              as_of_date,
                              horizons,
                              forecast_field="predicted_tmax",
                              forecast_label="Next 3 days (predicted)")

    # context stats
    st.markdown("---")
    render_station_context(station_id, as_of_date.year)


#Historical Explorer tab: today's original date-picker/backtest experience,
#unchanged — pick any past date and compare the model's call to what happened.
def render_explorer_tab(all_stations_df: pd.DataFrame):
    station_id, station_name = render_station_picker(all_stations_df, "explorer")

    st.caption(
        f"Station: **{station_name}**. Pick any past date to see what the "
        f"model would have predicted, compared with what actually happened."
    )

    min_date, max_date = load_date_range()
    default_date = min(max(date(2021, 6, 27), min_date), max_date)
    selected_date = st.date_input(
        "Date",
        value=default_date,
        min_value=min_date,
        max_value=max_date,
        key="explorer_date"
    )
    predict_btn = st.button("Predict", type="primary", key="explorer_predict")

    if not predict_btn:
        st.info("Pick a date, then click **Predict**.")
        return

    with st.spinner("Running prediction..."):
        resp = requests.get(
            f"{API_URL}/predict",
            params={"station_id": station_id,
                    "date": str(selected_date)}
        )

    if resp.status_code == 404:
        st.error(f"No data for {station_name} on {selected_date}. "
                 f"Try a different date.")
        return

    result = resp.json()
    window   = result["window"]
    horizons = result["horizons"]

    # layout: window card left, station readings right
    col1, col2 = st.columns([1, 1])

    with col1:
        render_window_card(window)

    with col2:
        st.markdown("### Station readings")
        m1, m2 = st.columns(2)
        todays_tmax = result["todays_tmax"]
        tmax_anomaly = result["tmax_anomaly"]
        m1.metric("Today's Tmax",
                  f"{todays_tmax}°C" if todays_tmax is not None else "—",
                  delta=(f"{tmax_anomaly:+.1f}°C vs normal"
                         if tmax_anomaly is not None else None))
        m2.metric("Extreme threshold",
                  f"{result['p95_tmax']:.1f}°C")

        m3, m4 = st.columns(2)
        m3.metric("3-day risk", f"{window['probability']*100:.1f}%")
        m4.metric("Actual (next 3d)",
                  "Extreme" if window["actual"] == 1
                  else ("Normal" if window["actual"] == 0 else "—"))

    # per-day strip
    st.markdown("---")
    render_horizon_strip(horizons, show_actual=True)

    # history chart
    st.markdown("---")
    history = load_station_history(station_id, selected_date)
    if not history.empty:
        render_history_chart(history,
                              result["p95_tmax"],
                              selected_date,
                              horizons)

    # context stats
    st.markdown("---")
    render_station_context(station_id, selected_date.year)


def main():
    all_stations_df   = load_stations()
    fresh_stations_df = load_forecast_ready_stations()

    render_sidebar(len(fresh_stations_df), len(all_stations_df))

    tab_forecast, tab_explorer = st.tabs(["Live Forecast", "Historical Explorer"])

    with tab_forecast:
        render_forecast_tab(fresh_stations_df)

    with tab_explorer:
        render_explorer_tab(all_stations_df)


if __name__ == "__main__":
    main()
