import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from xgboost import XGBClassifier

#Config
DB_URL = "postgresql://climate:climate123@localhost:5433/climate_dw"
MODELS_DIR = Path("modeling/saved_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = 2015   # train on everything up to & including 2015
VAL_END   = 2020   # validate on 2016- 2020
# test on 2021- 2026

SCALE_POS_WEIGHT = 14.07   # 15,406,839/1,095,116

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

TARGET_COL = "is_extreme_tomorrow"

#load features_daily data from postgres 
def load_data():
    print("connecting to postgres...")
    engine = create_engine(DB_URL)

    query = f"""
        SELECT 
            year, 
            {', '.join(FEATURE_COLS)},
            {TARGET_COL}
        FROM features_daily
        WHERE tmax_lag1 IS NOT NULL 
        AND tmax_roll7 IS NOT NULL
        AND tmax_roll30 IS NOT NULL
        AND tmax_tomorrow IS NOT NULL
    """
    print("loading the data from features_daily...")
    df = pd.read_sql(query, engine)
    print(f"loaded {len(df)} rows, and {df.shape[1]} columns.")

    #downcast to save memory
    for col in FEATURE_COLS:
        if col in df.columns: 
            df[col] = df[col].astype("float32")

    df[TARGET_COL] = df[TARGET_COL].astype("int8")
    return df 


