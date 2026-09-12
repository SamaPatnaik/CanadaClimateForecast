CREATE TABLE IF NOT EXISTS dim_stations (
    station_id VARCHAR(11) PRIMARY KEY,
    wmo_id VARCHAR(10),
    name VARCHAR(255) NOT NULL,
    lat DECIMAL(7,4) NOT NULL,
    lon DECIMAL(7,4) NOT NULL,
    elevation DECIMAL(7,1),
    province VARCHAR(50),
    first_year SMALLINT,
    last_year SMALLINT
);

CREATE TABLE IF NOT EXISTS dim_dates (
    date_id DATE PRIMARY KEY,
    year SMALLINT NOT NULL,
    month SMALLINT NOT NULL,
    day SMALLINT NOT NULL,
    day_of_year SMALLINT NOT NULL,
    week_of_year SMALLINT NOT NULL,
    quarter SMALLINT NOT NULL,
    season VARCHAR(10) NOT NULL,
    is_leap_year BOOLEAN NOT NULL
);


CREATE TABLE IF NOT EXISTS fact_observations (
    station_id VARCHAR(11) NOT NULL REFERENCES dim_stations(station_id),
    date_id DATE NOT NULL REFERENCES dim_dates(date_id),
    tmax_c NUMERIC(5,1),
    tmin_c NUMERIC(5,1),
    prcp_mm NUMERIC(6,1),
    snow_mm NUMERIC(6,1),
    snwd_mm NUMERIC(6,1),
    tmax_qflag CHAR(1),
    tmin_qflag CHAR(1),
    prcp_qflag CHAR(1),
    PRIMARY KEY (station_id, date_id)
);

CREATE INDEX IF NOT EXISTS idx_obs_date ON fact_observations(date_id);
CREATE INDEX IF NOT EXISTS idx_obs_station ON fact_observations(station_id);


-- features_daily used to be a materialized view; it's now a plain view.
-- Drop it by whichever kind it currently is so this script stays
-- re-runnable both before and after the migration.
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'features_daily' AND relkind = 'm') THEN
        EXECUTE 'DROP MATERIALIZED VIEW features_daily';
    ELSIF EXISTS (SELECT 1 FROM pg_class WHERE relname = 'features_daily' AND relkind = 'v') THEN
        EXECUTE 'DROP VIEW features_daily';
    END IF;
END $$;

DROP VIEW IF EXISTS features_latest;
DROP MATERIALIZED VIEW IF EXISTS features_daily_base;

CREATE MATERIALIZED VIEW features_daily_base AS
WITH base AS (
    SELECT
        o.station_id,
        o.date_id,
        d.year,
        d.month,
        d.day_of_year,
        d.season,
        s.lat,
        s.lon,
        s.elevation,
        o.tmax_c,
        o.tmin_c,
        o.prcp_mm,
        o.snow_mm
    FROM fact_observations o
    JOIN dim_dates d ON d.date_id = o.date_id
    JOIN dim_stations s ON s.station_id = o.station_id
    WHERE o.tmax_qflag IS NULL
      AND o.tmin_qflag IS NULL
),
 

climatology AS (
    SELECT
        station_id,
        day_of_year,
        AVG(tmax_c) AS clim_tmax_mean,
        STDDEV(tmax_c) AS clim_tmax_std,
        PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY tmax_c) AS p95_tmax
    FROM base
    GROUP BY station_id, day_of_year
),
 
-- Lag and rolling window features
-- WINDOW w = same station, ordered by date
lagged AS (
    SELECT
        *,
        -- Lag features (look back N days)
        LAG(tmax_c, 1) OVER w  AS tmax_lag1,
        LAG(tmax_c, 2) OVER w  AS tmax_lag2,
        LAG(tmax_c, 3) OVER w  AS tmax_lag3,
        LAG(tmin_c, 1) OVER w  AS tmin_lag1,
        LAG(prcp_mm, 1) OVER w AS prcp_lag1,
        -- Rolling 7-day average tmax (excludes current row)
        AVG(tmax_c) OVER (
            PARTITION BY station_id ORDER BY date_id
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS tmax_roll7,
        -- Rolling 30-day average tmax (excludes current row)
        AVG(tmax_c) OVER (
            PARTITION BY station_id ORDER BY date_id
            ROWS BETWEEN 29 PRECEDING AND 1 PRECEDING
        ) AS tmax_roll30,
        -- Rolling 7-day precipitation sum (excludes current row)
        SUM(prcp_mm) OVER (
            PARTITION BY station_id ORDER BY date_id
            ROWS BETWEEN 6 PRECEDING AND 1 PRECEDING
        ) AS prcp_roll7,
        -- Yesterday's diurnal range (tmax - tmin)
        LAG(tmax_c - tmin_c, 1) OVER w AS diurnal_range_lag1,
        -- Future tmax at horizons 1-3 days ahead (prediction targets).
        -- LEAD walks rows, not calendar days, so we also capture the lead
        -- date and validate it downstream (see the `targets` CTE).
        LEAD(tmax_c, 1)  OVER w AS tmax_h1_raw,
        LEAD(tmax_c, 2)  OVER w AS tmax_h2_raw,
        LEAD(tmax_c, 3)  OVER w AS tmax_h3_raw,
        LEAD(date_id, 1) OVER w AS date_h1,
        LEAD(date_id, 2) OVER w AS date_h2,
        LEAD(date_id, 3) OVER w AS date_h3
    FROM base
    WINDOW w AS (PARTITION BY station_id ORDER BY date_id)
),

-- Keep a future tmax only when the LEAD landed on the actual calendar day
-- N days ahead. Across data gaps the Nth following row is not day+N, and
-- using it would silently mislabel the target.
targets AS (
    SELECT
        *,
        CASE WHEN date_h1 = date_id + 1 THEN tmax_h1_raw END AS tmax_h1,
        CASE WHEN date_h2 = date_id + 2 THEN tmax_h2_raw END AS tmax_h2,
        CASE WHEN date_h3 = date_id + 3 THEN tmax_h3_raw END AS tmax_h3
    FROM lagged
)

SELECT
    l.station_id,
    l.date_id,
    l.year,
    l.month,
    l.day_of_year,
    l.season,
    l.lat,
    l.lon,
    l.elevation,
    -- Current day observations
    l.tmax_c,
    l.tmin_c,
    l.prcp_mm,
    l.snow_mm,
    -- Lag features
    l.tmax_lag1,
    l.tmax_lag2,
    l.tmax_lag3,
    l.tmin_lag1,
    l.prcp_lag1,
    -- Rolling features
    l.tmax_roll7,
    l.tmax_roll30,
    l.prcp_roll7,
    l.diurnal_range_lag1,
    -- Climatology features
    c.clim_tmax_mean,
    c.clim_tmax_std,
    c.p95_tmax,
    -- Anomaly: how much warmer/cooler than historical normal
    l.tmax_c - c.clim_tmax_mean AS tmax_anomaly,
    -- Raw future tmax at each horizon (used by the dashboard as "actual")
    l.tmax_h1,
    l.tmax_h2,
    l.tmax_h3,
    -- Per-day classification targets: is day+N an extreme heat day?
    -- 1 = that day's tmax exceeds the station's historical 95th percentile
    --     for the *current* day-of-year (a <0.1 C approximation over 3 days),
    -- 0 = normal day, NULL = that future day is unknown (data gap).
    (l.tmax_h1 > c.p95_tmax)::int AS is_extreme_h1,
    (l.tmax_h2 > c.p95_tmax)::int AS is_extreme_h2,
    (l.tmax_h3 > c.p95_tmax)::int AS is_extreme_h3,
    -- Windowed target: will ANY of the next 3 days be extreme?
    -- Only defined when all 3 future days are known.
    CASE
        WHEN l.tmax_h1 IS NOT NULL
         AND l.tmax_h2 IS NOT NULL
         AND l.tmax_h3 IS NOT NULL
        THEN (GREATEST(l.tmax_h1, l.tmax_h2, l.tmax_h3) > c.p95_tmax)::int
    END AS is_extreme_next_3d
FROM targets l
JOIN climatology c
    ON  c.station_id  = l.station_id
    AND c.day_of_year = l.day_of_year
WHERE l.tmax_lag1 IS NOT NULL   -- need at least one lag
;

-- Indexes for fast lookups during training and API queries
DROP INDEX IF EXISTS idx_feat_station_date;
DROP INDEX IF EXISTS idx_feat_year;

CREATE UNIQUE INDEX idx_feat_station_date
    ON features_daily_base(station_id, date_id);

CREATE INDEX idx_feat_year
    ON features_daily_base(year);

-- Training/backtest view: identical contents to the old features_daily
-- (requires future truth, i.e. tmax_h1, to exist for every row).
CREATE VIEW features_daily AS
SELECT *
FROM features_daily_base
WHERE tmax_h1 IS NOT NULL;   -- need at least the next day as a target

-- Live-forecast view: the single most recent forecast-ready row per
-- station, regardless of whether future truth exists yet. This is what
-- makes forecasting "today" possible instead of only ever backtesting
-- dates whose future is already known. tmax_c must be present - a
-- station can report other elements without a max-temp reading on its
-- most recent day, but tmax_c is both a required model feature and the
-- "today's reading" the API/dashboard display, so such a row is useless
-- as a forecast anchor even though it still has valid lag/rolling
-- features from prior days.
CREATE VIEW features_latest AS
SELECT DISTINCT ON (station_id) *
FROM features_daily_base
WHERE tmax_c IS NOT NULL
ORDER BY station_id, date_id DESC;