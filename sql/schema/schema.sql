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