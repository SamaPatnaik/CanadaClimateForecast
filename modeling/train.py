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
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor

#add this to config.py
DB_URL = "postgresql://climate:climate123@localhost:5433/climate_dw"
MODELS_DIR = Path("modeling/saved_models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = 2015   # train on everything up to & including 2015
VAL_END   = 2020   # validate on 2016- 2020
# test on 2021- 2026

HORIZONS = [1, 2, 3]   # forecast 1, 2 and 3 days ahead

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

# the per-day model also sees which horizon it is predicting
HORIZON_FEATURE_COLS = FEATURE_COLS + ["horizon"]

# targets
TARGET_HORIZON     = "is_extreme"      # per-day: is day+horizon extreme?
TARGET_WINDOW      = "is_extreme_next_3d"  # windowed: any of the next 3 days extreme?
TARGET_REGRESSION  = "tmax_target"     # per-day: actual tmax at day+horizon


#load features_daily data from postgres, reshaped to one row per (station, date, horizon)
def load_data():
    print("connecting to postgres...")
    engine = create_engine(DB_URL)

    query = f"""
        WITH filtered AS (
            SELECT *
            FROM features_daily
            WHERE tmax_lag1  IS NOT NULL
              AND tmax_roll7  IS NOT NULL
              AND tmax_roll30 IS NOT NULL
        ),

        -- unpivot the 3 per-horizon targets into rows
        long AS (
            SELECT
                f.*,
                h.horizon,
                CASE h.horizon
                    WHEN 1 THEN f.is_extreme_h1
                    WHEN 2 THEN f.is_extreme_h2
                    WHEN 3 THEN f.is_extreme_h3
                END AS is_extreme,
                -- raw future tmax at this horizon (regression target)
                CASE h.horizon
                    WHEN 1 THEN f.tmax_h1
                    WHEN 2 THEN f.tmax_h2
                    WHEN 3 THEN f.tmax_h3
                END AS tmax_target
            FROM filtered f
            CROSS JOIN (VALUES (1), (2), (3)) AS h(horizon)
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
                FROM long
                WHERE is_extreme IS NOT NULL
            ) x
        )

        SELECT
            year,
            station_id,
            date_id,
            horizon,
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
            is_extreme,
            is_extreme_next_3d,
            tmax_target
        FROM ranked
        WHERE (period = 'train'      AND rn <= 1400000)
           OR (period = 'validation' AND rn <= 400000)
           OR (period = 'test'       AND rn <= 200000);
    """
    print("loading ~2mil (station, date, horizon) rows from features_daily...")
    df = pd.read_sql(query, engine)
    print(f"loaded {len(df):,} rows, and {df.shape[1]} columns.")
    print(f"  Extreme (per-day): {df['is_extreme'].sum():,} "
          f"({df['is_extreme'].mean()*100:.2f}%)")
    for h in HORIZONS:
        sub = df[df["horizon"] == h]
        print(f"    h{h}: {len(sub):,} rows, {sub['is_extreme'].mean()*100:.2f}% extreme")

    #downcast to save memory
    for col in HORIZON_FEATURE_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float32")

    df["is_extreme"] = df["is_extreme"].astype("int8")
    df["tmax_target"] = df["tmax_target"].astype("float32")

    mem = df.memory_usage(deep=True).sum() / 1e9
    print(f"  Memory usage: {mem:.2f} GB")

    return df


#chronological split by year
def split_by_year(df):
    train_df = df[df["year"] <= TRAIN_END]
    val_df   = df[(df["year"] > TRAIN_END) & (df["year"] <= VAL_END)]
    test_df  = df[df["year"] > VAL_END]
    return train_df, val_df, test_df


def train_baseline(y_val):
    y_pred = np.zeros(len(y_val), dtype=int)

    print("\nBaseline (always predict 0):")
    print(classification_report(y_val, y_pred,
          target_names=["normal", "extreme"], zero_division=0))

    return {
        "f1":  f1_score(y_val, y_pred, average="binary", zero_division=0),
        "auc": 0.5,   # random classifier AUC
    }


#training the XGBoost model; scale_pos_weight is derived from the training labels
def train_xgboost(X_train, y_train, X_val, y_val):
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = neg / max(pos, 1)
    print(f"training xgboost model...  (scale_pos_weight = {spw:.2f})")

    model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = spw,
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


#training the XGBoost regressor for actual tmax at a given horizon
def train_xgboost_regressor(X_train, y_train, X_val, y_val):
    print("training xgboost regressor...")

    model = XGBRegressor(
        n_estimators = 500,
        max_depth = 6,
        learning_rate = 0.05,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = "reg:squarederror",
        eval_metric = "mae",
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


#naive baselines for temperature: persistence (today's tmax) and
#climatology (the historical mean tmax for that day-of-year).
#tmax_c/clim_tmax_mean can be NaN for some rows (a station can report
#tmin without tmax on a given day), so drop those before scoring -
#XGBoost handles NaN features natively but sklearn's metrics can't.
def regression_baselines(df, y):
    y = np.asarray(y)
    tmax_c = df["tmax_c"].to_numpy()
    clim   = df["clim_tmax_mean"].to_numpy()

    valid_p = np.isfinite(y) & np.isfinite(tmax_c)
    valid_c = np.isfinite(y) & np.isfinite(clim)

    persistence_mae = float(mean_absolute_error(y[valid_p], tmax_c[valid_p]))
    climatology_mae = float(mean_absolute_error(y[valid_c], clim[valid_c]))
    print(f"  Persistence baseline (today = tomorrow) MAE: {persistence_mae:.2f} C"
          f"  (n={valid_p.sum():,})")
    print(f"  Climatology baseline (day-of-year mean)  MAE: {climatology_mae:.2f} C"
          f"  (n={valid_c.sum():,})")
    return {"persistence_mae": persistence_mae, "climatology_mae": climatology_mae}


#given a split, evaluate a regressor: MAE/RMSE, optionally broken out by group.
def evaluate_regressor(model, X, y, splitname, groups=None):
    y = np.asarray(y)
    y_pred = model.predict(X)

    mae  = float(mean_absolute_error(y, y_pred))
    rmse = float(mean_squared_error(y, y_pred) ** 0.5)
    print(f"\n{'='*50}")
    print(f"{splitname}: MAE={mae:.2f} C  RMSE={rmse:.2f} C")
    print(f"{'='*50}")

    per_group = {}
    if groups is not None:
        groups = np.asarray(groups)
        print(f"\n{splitname} — by horizon:")
        for g in sorted(np.unique(groups)):
            mask = groups == g
            mae_g  = float(mean_absolute_error(y[mask], y_pred[mask]))
            rmse_g = float(mean_squared_error(y[mask], y_pred[mask]) ** 0.5)
            print(f"  h{int(g)}: MAE={mae_g:.2f} C  RMSE={rmse_g:.2f} C")
            per_group[f"h{int(g)}"] = {"mae": mae_g, "rmse": rmse_g}

    return {"mae": mae, "rmse": rmse, "by_horizon": per_group}


#given a split, evaluate model: f1, auc & confusion matrix. Optionally break out by group.
def evaluate(model, X, y, splitname, groups=None):
    y = np.asarray(y)
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

    per_group = {}
    if groups is not None:
        groups = np.asarray(groups)
        print(f"\n{splitname} — by horizon:")
        for g in sorted(np.unique(groups)):
            mask = groups == g
            auc_g = roc_auc_score(y[mask], y_prob[mask])
            f1_g  = f1_score(y[mask], y_pred_05[mask], average="binary", zero_division=0)
            cm_g  = confusion_matrix(y[mask], y_pred_05[mask])
            rec_g = (cm_g[1][1] / cm_g[1].sum()) if cm_g[1].sum() > 0 else 0.0
            print(f"  h{int(g)}: AUC={auc_g:.4f}  F1@0.5={f1_g:.4f}  recall@0.5={rec_g:.4f}")
            per_group[f"h{int(g)}"] = {"auc": auc_g, "f1_05": f1_g, "recall_05": rec_g}

    return {
        "auc":       auc,
        "f1_05":     f1_score(y, y_pred_05, average="binary", zero_division=0),
        "f1_03":     f1_score(y, y_pred_03, average="binary", zero_division=0),
        "recall_05": (cm[1][1]/(cm[1][0] + cm[1][1])) if cm[1].sum() > 0 else 0,
        "by_horizon": per_group,
    }


#save model + metrics + feature importance under a name prefix
def save_model(model, metrics: dict, feature_cols: list, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"model saved to {path}")

    metrics_path = MODELS_DIR / f"{name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

    importance = dict(zip(feature_cols,
                          [float(x) for x in model.feature_importances_]))
    importance = dict(sorted(importance.items(),
                             key=lambda x: x[1], reverse=True))
    fi_path = MODELS_DIR / f"{name}_feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump(importance, f, indent=2)
    print(f"Feature importance saved to {fi_path}")


if __name__ == "__main__":
    df = load_data()

    # ---------------------------------------------------------------
    # Model 1 — per-day horizon model (one model, horizon as a feature)
    # ---------------------------------------------------------------
    print("\n" + "#"*60)
    print("# Model 1: per-day extreme heat (horizons 1-3)")
    print("#"*60)

    tr, va, te = split_by_year(df)
    print(f"\nSplit: train={len(tr):,}  val={len(va):,}  test={len(te):,}")

    Xh_tr, yh_tr = tr[HORIZON_FEATURE_COLS], tr[TARGET_HORIZON]
    Xh_va, yh_va = va[HORIZON_FEATURE_COLS], va[TARGET_HORIZON]
    Xh_te, yh_te = te[HORIZON_FEATURE_COLS], te[TARGET_HORIZON]

    baseline_h = train_baseline(yh_va)
    model_h = train_xgboost(Xh_tr, yh_tr, Xh_va, yh_va)

    print("\nValidation set evaluation:")
    val_h = evaluate(model_h, Xh_va, yh_va, "Validation", groups=va["horizon"])
    print("\nTest set evaluation (held out):")
    test_h = evaluate(model_h, Xh_te, yh_te, "Test", groups=te["horizon"])

    save_model(model_h,
               {"baseline_f1": baseline_h["f1"], "val": val_h, "test": test_h},
               HORIZON_FEATURE_COLS,
               "xgb_extreme_heat_h3")

    # ---------------------------------------------------------------
    # Model 2 — "any extreme heat in the next 3 days" window model
    # ---------------------------------------------------------------
    print("\n" + "#"*60)
    print("# Model 2: extreme heat within the next 3 days")
    print("#"*60)

    w = df[df["horizon"] == 1].dropna(subset=[TARGET_WINDOW]).copy()
    w[TARGET_WINDOW] = w[TARGET_WINDOW].astype("int8")
    print(f"\nWindow rows: {len(w):,}  ({w[TARGET_WINDOW].mean()*100:.2f}% extreme)")

    wtr, wva, wte = split_by_year(w)
    print(f"Split: train={len(wtr):,}  val={len(wva):,}  test={len(wte):,}")

    Xw_tr, yw_tr = wtr[FEATURE_COLS], wtr[TARGET_WINDOW]
    Xw_va, yw_va = wva[FEATURE_COLS], wva[TARGET_WINDOW]
    Xw_te, yw_te = wte[FEATURE_COLS], wte[TARGET_WINDOW]

    baseline_w = train_baseline(yw_va)
    model_w = train_xgboost(Xw_tr, yw_tr, Xw_va, yw_va)

    print("\nValidation set evaluation:")
    val_w = evaluate(model_w, Xw_va, yw_va, "Validation")
    print("\nTest set evaluation (held out):")
    test_w = evaluate(model_w, Xw_te, yw_te, "Test")

    save_model(model_w,
               {"baseline_f1": baseline_w["f1"], "val": val_w, "test": test_w},
               FEATURE_COLS,
               "xgb_extreme_window3")

    # ---------------------------------------------------------------
    # Model 3 — per-day tmax regression (horizons 1-3)
    # ---------------------------------------------------------------
    print("\n" + "#"*60)
    print("# Model 3: predicted max temperature (horizons 1-3)")
    print("#"*60)

    Xr_tr, yr_tr = tr[HORIZON_FEATURE_COLS], tr[TARGET_REGRESSION]
    Xr_va, yr_va = va[HORIZON_FEATURE_COLS], va[TARGET_REGRESSION]
    Xr_te, yr_te = te[HORIZON_FEATURE_COLS], te[TARGET_REGRESSION]

    print("\nBaselines (validation set):")
    baseline_r = regression_baselines(va, yr_va)

    model_r = train_xgboost_regressor(Xr_tr, yr_tr, Xr_va, yr_va)

    print("\nValidation set evaluation:")
    val_r = evaluate_regressor(model_r, Xr_va, yr_va, "Validation", groups=va["horizon"])
    print("\nTest set evaluation (held out):")
    test_r = evaluate_regressor(model_r, Xr_te, yr_te, "Test", groups=te["horizon"])

    save_model(model_r,
               {"baselines": baseline_r, "val": val_r, "test": test_r},
               HORIZON_FEATURE_COLS,
               "xgb_tmax_regressor_h3")

    del df
    gc.collect()
    print("\nDone. Three models saved to", MODELS_DIR)
