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


DROP MATERIALIZED VIEW IF EXISTS features_daily;
 
CREATE MATERIALIZED VIEW features_daily AS
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
        -- Tomorrow's tmax (prediction target for regression)
        LEAD(tmax_c, 1) OVER w         AS tmax_tomorrow
    FROM base
    WINDOW w AS (PARTITION BY station_id ORDER BY date_id)
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
    -- Regression target: tomorrow's raw tmax
    l.tmax_tomorrow,
    -- Classification target: will tomorrow be an extreme heat day?
    -- 1 = tomorrow's tmax exceeds station's historical 95th percentile
    -- 0 = normal day
    CASE
        WHEN l.tmax_tomorrow > c.p95_tmax THEN 1
        ELSE 0
    END AS is_extreme_tomorrow
FROM lagged l
JOIN climatology c
    ON  c.station_id  = l.station_id
    AND c.day_of_year = l.day_of_year
WHERE l.tmax_lag1    IS NOT NULL   -- need at least one lag
  AND l.tmax_tomorrow IS NOT NULL  -- need known target
;
 
-- Indexes for fast lookups during training and API queries
DROP INDEX IF EXISTS idx_feat_station_date;
DROP INDEX IF EXISTS idx_feat_year;
 
CREATE INDEX idx_feat_station_date
    ON features_daily(station_id, date_id);
 
CREATE INDEX idx_feat_year
    ON features_daily(year);