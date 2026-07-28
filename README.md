# Canadian Climate ML

In this end to end data science project, I explore NOAA GHCN-Daily weather data → Postgres star schema → SQL feature engineering → XGBoost regression → FastAPI serving.

**Predicts next-day maximum temperature** for Canadian weather stations using historical climate observations.

## Architecture

```
NOAA GHCN-Daily
      │
      ▼
ingestion/download.py    ← fetches raw station + daily CSVs
      │
      ▼
ingestion/ingest.py      ← parses & loads into Postgres star schema
      │
      ▼
sql/schema.sql           ← dim_stations, dim_dates, fact_observations
      │                     features_daily (materialized view)
      ▼
modeling/train.py        ← pulls features via SQL, trains XGBoost
      │
      ▼
api/main.py              ← FastAPI serving predictions
      │
      ▼
dashboard/app.py         ← Streamlit visualization
```

## Stack

- **Storage**: PostgreSQL 16 (Docker)
- **Ingestion**: Python, psycopg2
- **Feature engineering**: SQL window functions, CTEs, materialized views
- **Modeling**: XGBoost, scikit-learn, chronological train/test split
- **Serving**: FastAPI
- **Dashboard**: Streamlit

## Quickstart

```bash
# 1. Start Postgres
docker compose up -d

# 2. Install Python dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Download data (starts with 50 stations)
python ingestion/download.py

# 4. Ingest into Postgres
python ingestion/ingest.py

# 5. Train the model
python modeling/train.py

# 6. Start the API
uvicorn api.main:app --reload

# 7. Launch dashboard
streamlit run dashboard/app.py
```

## Key design decisions

- **Star schema** (fact + dimension tables) rather than a flat table 
- **Feature engineering in SQL** (window functions, LAG, rolling aggregates) rather than pandas — SQL handles this more efficiently at scale and is closer to how real pipelines work
- **Chronological train/test split** — avoids data leakage that would occur with a random split on time-series data
- **Baseline comparison** — model is evaluated against climatology (historical average) to demonstrate actual predictive skill

![ER Diagram](docs/er_diagram.png)
