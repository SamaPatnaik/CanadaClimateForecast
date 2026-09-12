import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, date
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from typing import Optional

# add this to config.py
DB_URL = "postgresql://climate:climate123@localhost:5433/climate_dw"
MODEL_DIR = Path("modeling/saved_models")
HORIZON_MODEL_PATH = MODEL_DIR / "xgb_extreme_heat_h3.joblib"
WINDOW_MODEL_PATH  = MODEL_DIR / "xgb_extreme_window3.joblib"
TMAX_MODEL_PATH    = MODEL_DIR / "xgb_tmax_regressor_h3.joblib"

HORIZONS = [1, 2, 3]
THRESHOLD = 0.5

# base features known at prediction time (day 0)
FEATURE_COLS = [
    "month", "day_of_year",
    "lat", "lon", "elevation",
    "tmax_c", "tmin_c", "prcp_mm", "snow_mm",
    "tmax_lag1", "tmax_lag2", "tmax_lag3",
    "tmin_lag1", "prcp_lag1",
    "tmax_roll7", "tmax_roll30", "prcp_roll7",
    "diurnal_range_lag1",
    "clim_tmax_mean", "clim_tmax_std",
    "tmax_anomaly", "p95_tmax",
]
# the per-day model also takes the horizon it is predicting
HORIZON_FEATURE_COLS = FEATURE_COLS + ["horizon"]

#setup
app = FastAPI(title="Canadian Extreme Heat Prediction API")
engine = create_engine(DB_URL)

print("loading models...")
model_h    = joblib.load(HORIZON_MODEL_PATH)   # per-day, horizons 1-3
model_w    = joblib.load(WINDOW_MODEL_PATH)    # any extreme in the next 3 days
model_tmax = joblib.load(TMAX_MODEL_PATH)      # per-day predicted tmax, horizons 1-3
print(f"models loaded from {MODEL_DIR}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models_loaded": model_h is not None and model_w is not None
                          and model_tmax is not None,
        "horizon_model": type(model_h).__name__,
        "window_model": type(model_w).__name__,
        "tmax_model": type(model_tmax).__name__,
        "horizons": HORIZONS,
    }


@app.get("/stations")
def get_stations(province: Optional[str] = None):
    query = """
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
    """
    params = {}
    if province:
        query += " AND s.province = :province"
        params["province"] = province.upper()

    query += " ORDER BY s.province, s.name"

    with engine.connect() as conn:
        result = conn.execute(text(query), params)
        rows = result.mappings().all()

    return {
        "count": len(rows),
        "stations": [dict(r) for r in rows]
    }


@app.get("/predict")
def predict(station_id: str, date: str):
    # station_id: GHCN station ID e.g. CA001108487
    # date: date in YYYY-MM-DD format e.g. 2024-08-15
    # returns per-day extreme-heat risk for the next 3 days plus a
    # single "extreme heat within 3 days" probability.

    query = text("""
        SELECT
            month, day_of_year,
            lat, lon, elevation,
            tmax_c, tmin_c, prcp_mm, snow_mm,
            tmax_lag1, tmax_lag2, tmax_lag3,
            tmin_lag1, prcp_lag1,
            tmax_roll7, tmax_roll30, prcp_roll7,
            diurnal_range_lag1,
            clim_tmax_mean, clim_tmax_std,
            tmax_anomaly, p95_tmax,
            tmax_h1, tmax_h2, tmax_h3,
            is_extreme_h1, is_extreme_h2, is_extreme_h3,
            is_extreme_next_3d
        FROM features_daily
        WHERE station_id = :station_id
          AND date_id = :date
    """)

    with engine.connect() as conn:
        result = conn.execute(query,
                              {"station_id": station_id, "date": date})
        row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for station {station_id} on {date}. "
                   f"Check the station ID and date are valid."
        )

    # base feature vector (right column order), shared by both models
    base = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}]).astype("float32")

    base_date = datetime.strptime(date, "%Y-%m-%d").date()

    # per-day horizon predictions
    horizons = []
    for h in HORIZONS:
        feats = base.copy()
        feats["horizon"] = np.float32(h)
        feats = feats[HORIZON_FEATURE_COLS]

        prob = float(model_h.predict_proba(feats)[0][1])
        actual = row[f"is_extreme_h{h}"]
        future_tmax = row[f"tmax_h{h}"]

        horizons.append({
            "horizon":     h,
            "date":        str(base_date + timedelta(days=h)),
            "probability": round(prob, 4),
            "is_extreme":  prob >= THRESHOLD,
            "future_tmax": float(future_tmax) if future_tmax is not None else None,
            "actual":      int(actual) if actual is not None else None,
        })

    # windowed prediction: extreme heat on ANY of the next 3 days
    w_prob = float(model_w.predict_proba(base[FEATURE_COLS])[0][1])
    w_actual = row["is_extreme_next_3d"]

    # station name for the response
    name_query = text("""
        SELECT name, province
        FROM dim_stations
        WHERE station_id = :station_id
    """)
    with engine.connect() as conn:
        name_row = conn.execute(name_query,
                                {"station_id": station_id}).mappings().first()

    return {
        "station_id":   station_id,
        "station_name": name_row["name"] if name_row else None,
        "province":     name_row["province"] if name_row else None,
        "date":         date,
        "window": {
            "days":        3,
            "probability": round(w_prob, 4),
            "is_extreme":  w_prob >= THRESHOLD,
            "actual":      int(w_actual) if w_actual is not None else None,
        },
        "horizons":     horizons,
        "threshold":    THRESHOLD,
        "p95_tmax":     float(row["p95_tmax"]),
        # a station can report other elements without a max-temp reading
        # on a given day, so tmax_c/tmax_anomaly can be null here
        "todays_tmax":  float(row["tmax_c"]) if row["tmax_c"] is not None else None,
        "tmax_anomaly": float(row["tmax_anomaly"]) if row["tmax_anomaly"] is not None else None,
    }


@app.get("/forecast")
def forecast(station_id: str):
    # station_id: GHCN station ID e.g. CA001108487
    # Genuine forward-looking forecast: anchors on the most recent real
    # observation for this station (features_latest) and projects 1-3 days
    # ahead, rather than looking up a date the caller supplies. The future
    # is unknown here, so there are no "actual"/"future_tmax" fields.

    query = text("""
        SELECT
            date_id,
            month, day_of_year,
            lat, lon, elevation,
            tmax_c, tmin_c, prcp_mm, snow_mm,
            tmax_lag1, tmax_lag2, tmax_lag3,
            tmin_lag1, prcp_lag1,
            tmax_roll7, tmax_roll30, prcp_roll7,
            diurnal_range_lag1,
            clim_tmax_mean, clim_tmax_std,
            tmax_anomaly, p95_tmax
        FROM features_latest
        WHERE station_id = :station_id
    """)

    with engine.connect() as conn:
        result = conn.execute(query, {"station_id": station_id})
        row = result.mappings().first()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for station {station_id}. "
                   f"Check the station ID is valid."
        )

    as_of_date = row["date_id"]
    data_age_days = (date.today() - as_of_date).days

    # base feature vector (right column order), shared by both models
    base = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}]).astype("float32")

    # per-day horizon predictions, genuinely forward from as_of_date
    horizons = []
    for h in HORIZONS:
        feats = base.copy()
        feats["horizon"] = np.float32(h)
        feats = feats[HORIZON_FEATURE_COLS]

        prob = float(model_h.predict_proba(feats)[0][1])
        predicted_tmax = float(model_tmax.predict(feats)[0])

        horizons.append({
            "horizon":        h,
            "date":           str(as_of_date + timedelta(days=h)),
            "probability":    round(prob, 4),
            "is_extreme":     prob >= THRESHOLD,
            "predicted_tmax": round(predicted_tmax, 1),
        })

    # windowed prediction: extreme heat on ANY of the next 3 days
    w_prob = float(model_w.predict_proba(base[FEATURE_COLS])[0][1])

    # station name for the response
    name_query = text("""
        SELECT name, province
        FROM dim_stations
        WHERE station_id = :station_id
    """)
    with engine.connect() as conn:
        name_row = conn.execute(name_query,
                                {"station_id": station_id}).mappings().first()

    return {
        "station_id":     station_id,
        "station_name":   name_row["name"] if name_row else None,
        "province":       name_row["province"] if name_row else None,
        "as_of_date":     str(as_of_date),
        "data_age_days":  data_age_days,
        "window": {
            "days":        3,
            "probability": round(w_prob, 4),
            "is_extreme":  w_prob >= THRESHOLD,
        },
        "horizons":     horizons,
        "threshold":    THRESHOLD,
        "p95_tmax":     float(row["p95_tmax"]),
        "todays_tmax":  float(row["tmax_c"]),
        "tmax_anomaly": float(row["tmax_anomaly"]),
    }
