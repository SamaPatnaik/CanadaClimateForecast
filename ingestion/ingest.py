import os 
import csv 
import psycopg2
import psycopg2.extras
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from config import(RAW_DIR, DAILY_DIR,
    URL_BYSTATION, URL_DAILYSTATION, URL_INV,
    PREFIX, MIN_YRS) 

# config 
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5433")),
    "dbname": os.getenv("DB_NAME", "climate_dw"),
    "user": os.getenv("DB_USER", "climate"),
    "password": os.getenv("DB_PASSWORD", "climate123"),
}

ELEMENTS = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD"}

BAD_QFLAGS = {"D","G","I","K","L","M","N","O","R","S","T","W","X","Z"}


# stations_df = pd.read_csv("canada_stations.csv")
# stations_df = stations_df.rename(columns={
#     "latitude":  "lat",
#     "longitude": "lon",
#     "name": "name"
# })


#establish connection to Postgres database
def get_connection(): 
    try: 
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.Error as e:
        print(f"Error connecting to database: {e}")
        return None


# upsert canadian station data from download.py into dim_stations in Postgres
def upsert_stations(conn, stations_df):
    sql = """
        INSERT INTO dim_stations
            (station_id, name, province, lat, 
             lon, elevation, wmo_id, first_year, last_year)
        VALUES %s
        ON CONFLICT (station_id) DO UPDATE SET
            name      = EXCLUDED.name,
            last_year = EXCLUDED.last_year
    """
    rows = [
        (
            row.station_id,
            row.name.strip() if pd.notna(row.name) else None,
            row.province.strip() if pd.notna(row.province)  else None,
            float(row.lat),
            float(row.lon),
            float(row.elevation) if pd.notna(row.elevation) else None,
            str(row.wmo_id).strip() if pd.notna(row.wmo_id) else None,
            int(row.first_year) if pd.notna(row.first_year) else None,
            int(row.last_year) if pd.notna(row.last_year)  else None,
        )
        for row in stations_df.itertuples()
    ]
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=500)
    conn.commit()
    print(f"  Upserted {len(rows)} stations into dim_stations.")


#populate the dim_dates table with date info from start to end date
def populate_dates(conn, start, end):
    season_map = {
        12: "Winter", 1: "Winter",  2: "Winter",
        3:  "Spring", 4: "Spring",  5: "Spring",
        6:  "Summer", 7: "Summer",  8: "Summer",
        9:  "Fall",   10: "Fall",   11: "Fall"
    }

    rows = []
    current = start
    while current <= end:
        is_leap = (current.year % 4 == 0 and 
                  (current.year % 100 != 0 or 
                   current.year % 400 == 0))
        rows.append((
            current,
            current.year,
            current.month,
            current.day,
            current.timetuple().tm_yday,
            int(current.strftime("%W")),
            (current.month - 1) // 3 + 1,
            season_map[current.month],
            is_leap,
        ))
        current += timedelta(days=1)

    sql = """
        INSERT INTO dim_dates
            (date_id, year, month, day, day_of_year,
             week_of_year, quarter, season, is_leap_year)
        VALUES %s
        ON CONFLICT (date_id) DO NOTHING
    """
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, rows, page_size=1000)
    conn.commit()
    print(f"  Populated dim_dates: {start} to {end} ({len(rows)} rows).")


# parse a raw GHCN station inventory file into a pandas DataFrame
def parse_ghcn_csv(filepath):
    rows_by_date = {}

    with open(filepath, "r") as f:
        reader = csv.reader(f)
        for line in reader:
            if len(line) < 8:
                continue

            station_id, date_str, element, value_raw, mflag, qflag, sflag, *_ = line

            # skip elements we don't care about
            if element not in ELEMENTS:
                continue

            # skip bad quality flags
            if qflag.strip() in BAD_QFLAGS:
                continue

            # skip missing value 
            try:
                value = int(value_raw)
                if value == -9999:
                    continue
            except ValueError:
                continue

            # convert from 10ths to proper units
            if element in ("TMAX", "TMIN"):
                value = round(value / 10.0, 1)   # tenths °C → °C
            elif element in ("PRCP", "SNOW", "SNWD"):
                value = round(value / 10.0, 1)   # tenths mm → mm

            #group by date
            if date_str not in rows_by_date:
                rows_by_date[date_str] = {
                    "station_id": station_id,
                    "qflags": {}
                }

            rows_by_date[date_str][element] = value
            rows_by_date[date_str]["qflags"][element] = qflag.strip() or None

    #list of dicts
    result = []
    for date_str, data in rows_by_date.items():
        try:
            d = date(
                int(date_str[:4]),
                int(date_str[4:6]),
                int(date_str[6:8])
            )
        except ValueError:
            continue

        qflags = data.get("qflags", {})
        result.append({
            "station_id": data["station_id"],
            "date_id": d,
            "tmax_c": data.get("TMAX"),
            "tmin_c": data.get("TMIN"),
            "prcp_mm": data.get("PRCP"),
            "snow_mm": data.get("SNOW"),
            "snwd_mm": data.get("SNWD"),
            "tmax_qflag": qflags.get("TMAX"),
            "tmin_qflag": qflags.get("TMIN"),
            "prcp_qflag": qflags.get("PRCP"),
        })

    return result


#takes the list of dicts and upserts them into the fact_observations table
def upsert_observations(conn, rows):
    if not rows:
        return

    sql = """
        INSERT INTO fact_observations
            (station_id, date_id, tmax_c, tmin_c, prcp_mm,
             snow_mm, snwd_mm, tmax_qflag, tmin_qflag, prcp_qflag)
        VALUES %s
        ON CONFLICT (station_id, date_id) DO UPDATE SET
            tmax_c     = EXCLUDED.tmax_c,
            tmin_c     = EXCLUDED.tmin_c,
            prcp_mm    = EXCLUDED.prcp_mm,
            snow_mm    = EXCLUDED.snow_mm,
            snwd_mm    = EXCLUDED.snwd_mm,
            tmax_qflag = EXCLUDED.tmax_qflag,
            tmin_qflag = EXCLUDED.tmin_qflag,
            prcp_qflag = EXCLUDED.prcp_qflag
    """

    values = [
        (
            r["station_id"], r["date_id"],
            r["tmax_c"],     r["tmin_c"],
            r["prcp_mm"],    r["snow_mm"],   r["snwd_mm"],
            r["tmax_qflag"], r["tmin_qflag"], r["prcp_qflag"]
        )
        for r in rows
    ]

    with conn.cursor() as cur:
        psycopg2.extras.execute_values(cur, sql, values, page_size=2000)
    conn.commit()


#refresh the features_daily materialized view
def refresh_features(conn):
    print("  Refreshing features_daily materialized view...")
    with conn.cursor() as cur:
        cur.execute("REFRESH MATERIALIZED VIEW features_daily;")
    conn.commit()
    print("  Done.")


if __name__ == "__main__":
    stations_file = RAW_DIR / "canada_stations.csv"
    if not stations_file.exists():
        raise FileNotFoundError("Run download.py first.")

    stations_df  = pd.read_csv(stations_file)
    daily_files  = sorted(DAILY_DIR.glob("*.csv"))

    if not daily_files:
        raise FileNotFoundError("No daily files found. Run download.py first.")

    print(f"Connecting to Postgres...")
    conn = get_connection()
    if conn is None:
        raise RuntimeError("Could not connect to database.")
    print("Connected.\n")

    # Step 1: load stations dimension
    print("Loading stations...")
    upsert_stations(conn, stations_df)

    # Step 2: populate dates dimension
    print("Populating dates...")
    populate_dates(conn, date(1840, 1, 1), date(2030, 12, 31))

    # Step 3: parse and load observations
    print(f"Ingesting {len(daily_files)} station files...")
    total_rows = 0
    for i, filepath in enumerate(daily_files, 1):
        rows = parse_ghcn_csv(filepath)
        if rows:
            upsert_observations(conn, rows)
            total_rows += len(rows)
        if i % 10 == 0:
            print(f"  {i}/{len(daily_files)} files processed "
                  f"({total_rows:,} rows so far)...")

    print(f"  Ingested {total_rows:,} total rows.")

    # Step 4: refresh materialized view
    print("Refreshing feature view...")
    refresh_features(conn)

    conn.close()
    print("\nIngestion complete.")
    print("Verify with:")
    print("  SELECT COUNT(*) FROM fact_observations;")
    print("  SELECT COUNT(*) FROM features_daily;")