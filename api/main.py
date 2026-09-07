import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from sqlalchemy import Date, create_engine, text

# add this to config.py 
DB_URL = "postgresql://climate:climate123@localhost:5433/climate_dw"
MODEL_PATH = Path("modeling/saved_models/xgb_extreme_heat.joblib")

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

#setup
app = FastAPI(title = "Canadian Extreme Heat Prediction API")
engine = create_engine(DB_URL)

print("load the model...")
model = joblib.load(MODEL_PATH)
print(f"model loaded from {MODEL_PATH}")


# we want to set up 3 different endpoints - 
# check health, get features for a given date and station, and predict extreme heat for a given date and station

@app.get("/health")
def health(): 
    return {
        "status": "ok", 
        "model_loaded": model is not None,
        "model_type": type(model).__name__,
    }

@app.get("/stations")
def get_stations(province):
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

    if province:
            query += f" AND s.province = '{province.upper()}'"

    query += " ORDER BY s.province, s.name"

    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.mappings().all()

    return {
        "count": len(rows),
        "stations": [dict(r) for r in rows]
    }

@app.get("/predict")
def predict(station_id: str, date: str):
    # station_id: GHCN station ID e.g. CA001108487
    # date: date in YYYY-MM-DD format e.g. 2024-08-15

    # fetch the feature row for this station + date
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
            is_extreme_tomorrow
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

    #build feature vector in the right column order
    features = pd.DataFrame([{col: row[col] for col in FEATURE_COLS}])
    features = features.astype("float32")

    #get probability from model
    prob = float(model.predict_proba(features)[0][1])
    pred = int(prob >= 0.5)

    #fetch station name for response
    name_query = text("""
        SELECT name, province 
        FROM dim_stations 
        WHERE station_id = :station_id
    """)
    with engine.connect() as conn:
        name_row = conn.execute(name_query, {"station_id": station_id}).mappings().first()
    return {
        "station_id":    station_id,
        "station_name":  name_row["name"] if name_row else None,
        "province":      name_row["province"] if name_row else None,
        "date":          date,
        "prediction":    pred,
        "probability":   round(prob, 4),
        "is_extreme":    pred == 1,
        "threshold":     0.5,
        "p95_tmax":      float(row["p95_tmax"]),
        "todays_tmax":   float(row["tmax_c"]),
        "tmax_anomaly":  float(row["tmax_anomaly"]),
        "actual":        int(row["is_extreme_tomorrow"])
    }