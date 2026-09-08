import streamlit as st 
import pandas as pd 
import requests 
import plotly.graph_objects as go
from datetime import date, datetime, timedelta
from sqlalchemy import create_engine, text

API_URL = "http://localhost:8000"
DB_URL = "postgresql://climate:climate123@localhost:5433/climate_dw"

#page config
st.set_page_config(
    page_title="Canadian Extreme Heat Predictor",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


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


#Load last 60 days of tmax + the p95 threshold for this station.
@st.cache_data
def load_station_history(station_id: str):
    engine = get_engine()
    query = text("""
        SELECT
            date_id,
            tmax_c,
            clim_tmax_mean,
            p95_tmax,
            is_extreme_tomorrow,
            tmax_anomaly
        FROM features_daily
        WHERE station_id = :sid
        ORDER BY date_id DESC
        LIMIT 60
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"sid": station_id})
    return df.sort_values("date_id")


#Station-level stats for context panel.
@st.cache_data
def load_station_context(station_id: str, year: int):
    engine = get_engine()
    query = text("""
        SELECT
            COUNT(*) FILTER (WHERE is_extreme_tomorrow = 1
                AND EXTRACT(year FROM date_id) = :year)
                AS extreme_this_year,
            COUNT(*) FILTER (WHERE is_extreme_tomorrow = 1
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


def render_sidebar(stations_df):
    st.sidebar.title("🌡️ Heat Risk Predictor")
    st.sidebar.markdown("---")

    # province filter
    provinces = sorted(stations_df["province"].dropna().unique())
    province = st.sidebar.selectbox("Province", provinces,
                                    index=provinces.index("BC")
                                    if "BC" in provinces else 0)

    # station filter — only show stations in selected province
    filtered = stations_df[stations_df["province"] == province]
    station_options = filtered["name"].tolist()
    station_ids     = filtered["station_id"].tolist()

    station_name = st.sidebar.selectbox("Station", station_options)
    station_id   = station_ids[station_options.index(station_name)]

    # date picker — default to most recent available date
    selected_date = st.sidebar.date_input(
        "Date",
        value=date(2021, 6, 27),
        min_value=date(1873, 1, 1),
        max_value=date(2026, 8, 29)
    )

    predict_btn = st.sidebar.button("🔍 Predict", type="primary",
                                     use_container_width=True)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Model: XGBoost | AUC-ROC: 0.875 | Recall: 86%\n\n"
        "Data: NOAA GHCN-Daily | 1,299 Canadian stations"
    )

    return station_id, station_name, selected_date, predict_btn


def render_risk_card(result: dict):
    prob      = result["probability"]
    is_extreme = result["is_extreme"]

    if prob >= 0.75:
        color  = "#ff4444"
        emoji  = "🔴"
        label  = "HIGH RISK"
        msg    = "Extreme heat very likely tomorrow"
    elif prob >= 0.5:
        color  = "#ff8800"
        emoji  = "🟠"
        label  = "ELEVATED RISK"
        msg    = "Conditions favour extreme heat tomorrow"
    elif prob >= 0.3:
        color  = "#ffcc00"
        emoji  = "🟡"
        label  = "MODERATE RISK"
        msg    = "Some chance of extreme heat tomorrow"
    else:
        color  = "#00cc44"
        emoji  = "🟢"
        label  = "LOW RISK"
        msg    = "Extreme heat unlikely tomorrow"

    st.markdown(f"""
        <div style="
            background: {color}22;
            border: 2px solid {color};
            border-radius: 12px;
            padding: 24px;
            text-align: center;
            margin-bottom: 16px;
        ">
            <div style="font-size: 48px">{emoji}</div>
            <div style="font-size: 28px; font-weight: bold;
                        color: {color};">{label}</div>
            <div style="font-size: 48px; font-weight: bold;
                        margin: 8px 0;">{prob*100:.0f}%</div>
            <div style="font-size: 16px; color: #666;">{msg}</div>
        </div>
    """, unsafe_allow_html=True)



def render_history_chart(history_df: pd.DataFrame,
                          p95_tmax: float,
                          selected_date: date):
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

    # highlight selected date
    fig.add_vline(
    x=selected_date.strftime("%Y-%m-%d"),
    line_color="orange",
    line_dash="dash"
    )

    fig.update_layout(
        title="60-day temperature history",
        xaxis_title="Date",
        yaxis_title="Max Temperature (°C)",
        legend=dict(orientation="h", y=-0.2),
        height=350,
        margin=dict(t=40, b=40)
    )

    st.plotly_chart(fig, use_container_width=True)



def main():
    stations_df = load_stations()
    station_id, station_name, selected_date, predict_btn = \
        render_sidebar(stations_df)

    st.title(f"🌡️ Extreme Heat Prediction")
    st.caption(f"Station: **{station_name}** | Date: **{selected_date}**")

    if predict_btn:
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

        # layout: risk card left, stats right
        col1, col2 = st.columns([1, 1])

        with col1:
            render_risk_card(result)

        with col2:
            st.markdown("### Station readings")
            m1, m2 = st.columns(2)
            m1.metric("Today's Tmax",
                      f"{result['todays_tmax']}°C",
                      delta=f"+{result['tmax_anomaly']:.1f}°C vs normal")
            m2.metric("Extreme threshold",
                      f"{result['p95_tmax']:.1f}°C")

            m3, m4 = st.columns(2)
            m3.metric("Probability", f"{result['probability']*100:.1f}%")
            m4.metric("Actual outcome",
                      "Extreme ✓" if result["actual"] == 1 else "Normal ✓")

        # history chart
        st.markdown("---")
        history = load_station_history(station_id)
        if not history.empty:
            render_history_chart(history,
                                  result["p95_tmax"],
                                  selected_date)

        # context stats
        st.markdown("---")
        ctx = load_station_context(station_id, selected_date.year)
        if ctx:
            st.markdown("### Station context")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Extreme days this year",
                      ctx["extreme_this_year"] or 0)
            c2.metric("Extreme days this month",
                      ctx["extreme_this_month"] or 0)
            c3.metric("Record high at station",
                      f"{ctx['record_tmax']}°C")
            c4.metric("Record low at station",
                      f"{ctx['record_tmin']}°C")
    else:
        st.info("👈 Select a province, station, and date in the sidebar, "
                "then click **Predict**.")

if __name__ == "__main__":
    main()