import joblib
import pandas as pd
import numpy as np
import json
import gc
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
        WITH filtered AS (
        SELECT *
        FROM features_daily
        WHERE tmax_lag1 IS NOT NULL
        AND tmax_roll7 IS NOT NULL
        AND tmax_roll30 IS NOT NULL
    ),

    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY period
                ORDER BY RANDOM()
            ) AS rn
        FROM (
            SELECT *,
                CASE
                    WHEN year <= 2015 THEN 'train'
                    WHEN year <= 2020 THEN 'validation'
                    ELSE 'test'
                END AS period
            FROM filtered
        ) x
    )

    SELECT
        year,
        station_id,
        date_id,
        month,
        day_of_year,
        lat,
        lon,
        elevation,
        tmax_c,
        tmin_c,
        prcp_mm,
        snow_mm,
        tmax_lag1,
        tmax_lag2,
        tmax_lag3,
        tmin_lag1,
        prcp_lag1,
        tmax_roll7,
        tmax_roll30,
        prcp_roll7,
        diurnal_range_lag1,
        clim_tmax_mean,
        clim_tmax_std,
        tmax_anomaly,
        p95_tmax,
        is_extreme_tomorrow
    FROM ranked
    WHERE (period = 'train' AND rn <= 1400000)
    OR (period = 'validation' AND rn <= 400000)
    OR (period = 'test' AND rn <= 200000);

    """
    print("loading 2mil data from features_daily...")
    df = pd.read_sql(query, engine)
    print(f"loaded {len(df)} rows, and {df.shape[1]} columns.")
    print(f"  Extreme days: {df['is_extreme_tomorrow'].sum():,} "
          f"({df['is_extreme_tomorrow'].mean()*100:.2f}%)")

    #downcast to save memory
    for col in FEATURE_COLS:
        if col in df.columns: 
            df[col] = df[col].astype("float32")

    df["is_extreme_tomorrow"] = df["is_extreme_tomorrow"].astype("int8")

    mem = df.memory_usage(deep=True).sum() / 1e9
    print(f"  Memory usage: {mem:.2f} GB")

    return df


#Now we want to split the data into train, validation and test sets
def split_data(df): 
    train_df = df[df["year"] <= TRAIN_END]
    val_df = df[(df["year"] > TRAIN_END) & (df["year"] <= VAL_END)]
    test_df = df[df["year"] > VAL_END]

    print(f"\nSplit summary:")
    print(f"  Train:    {len(train_df):>10,} rows  "
          f"({train_df[TARGET_COL].mean()*100:.2f}% extreme)")
    print(f"  Validate: {len(val_df):>10,} rows  "
          f"({val_df[TARGET_COL].mean()*100:.2f}% extreme)")
    print(f"  Test:     {len(test_df):>10,} rows  "
          f"({test_df[TARGET_COL].mean()*100:.2f}% extreme)")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    X_val = val_df[FEATURE_COLS]
    y_val = val_df[TARGET_COL]

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    #free full df from memory
    del df
    gc.collect()

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_baseline(y_val):  
    y_pred = np.zeros(len(y_val), dtype=int)

    print("\nBaseline (always predict 0):")
    print(classification_report(y_val, y_pred,
          target_names=["normal", "extreme"]))

    return {
        "f1":  f1_score(y_val, y_pred, average="binary"),
        "auc": 0.5,   # random classifier AUC
    }

#training the XGBoost model 
def train_xgboost(X_train, y_train, X_val, y_val):
    print("training xgboost model...")
    model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6, 
        learning_rate = 0.05, 
        subsample = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = SCALE_POS_WEIGHT,
        eval_metric = "aucpr", 
        early_stopping_rounds = 20,
        random_state = 42,
        n_jobs = -1
    )

    model.fit(
        X_train, y_train,
        eval_set = [(X_val, y_val)],
        verbose = 50
    )
    print(f"\n  Best iteration: {model.best_iteration}")
    return model 


#given a split, it evaluates model and gives f1 score, auc & confusion matrix
def evaluate(model, X, y, splitname): 
    y_prob = model.predict_proba(X)[:, 1]
    y_pred_05 = (y_prob >= 0.5).astype(int)
    y_pred_03 = (y_prob >= 0.3).astype(int)

    print(f"\n{'='*50}")
    print(f"{splitname} threshold 0.5")
    print(f"{'='*50}")
    print(classification_report(y, y_pred_05,
          target_names=["normal", "extreme"],
          zero_division=0))

    print(f"\n{splitname} threshold 0.3 (higher recall)")
    print(f"{'='*50}")
    print(classification_report(y, y_pred_03,
          target_names=["normal", "extreme"],
          zero_division=0))

    auc = roc_auc_score(y, y_prob)
    print(f"AUC-ROC: {auc:.4f}")

    cm = confusion_matrix(y, y_pred_05)
    print(f"\nConfusion matrix (threshold 0.5):")
    print(f"                Predicted Normal  Predicted Extreme")
    print(f"Actual Normal:  {cm[0][0]:>14,}  {cm[0][1]:>17,}")
    print(f"Actual Extreme: {cm[1][0]:>14,}  {cm[1][1]:>17,}")

    return {
        "auc":       auc,
        "f1_05":     f1_score(y, y_pred_05, average="binary", zero_division=0),
        "f1_03":     f1_score(y, y_pred_03, average="binary", zero_division=0),
        "recall_05": (cm[1][1]/(cm[1][0] + cm[1][1])) if cm[1].sum() > 0 else 0,
    }


#save model
def save_model(model, metrics: dict):
    path = MODELS_DIR / "xgb_extreme_heat.joblib"
    joblib.dump(model, path)
    print(f"model saved to {path}")

    # save metrics with model
    metrics_path = MODELS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    importance = dict(zip(FEATURE_COLS,
                         [float(x) for x in model.feature_importances_]))
    importance = dict(sorted(importance.items(),
                            key=lambda x: x[1], reverse=True))
    fi_path = MODELS_DIR / "feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"Feature importance saved to {fi_path}")



if __name__ == "__main__":
    df = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(df)

    # establish the floor
    baseline = train_baseline(y_val)

    # train XGBoost
    model = train_xgboost(X_train, y_train, X_val, y_val)

    # evaluate on validation set
    print("\nValidation set evaluation:")
    val_metrics = evaluate(model, X_val, y_val, "Validation")

    #evaluate on test set, final eval
    print("\nTest set evaluation (held out):")
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # save
    save_model(model, {
        "baseline_f1": baseline["f1"],
        "val":  val_metrics,
        "test": test_metrics,
    })

