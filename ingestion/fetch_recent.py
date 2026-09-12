# Fetch near-real-time daily observations from Environment and Climate
# Change Canada's GeoMet OGC API (climate-daily collection) and upsert them
# into fact_observations, so features_latest stays much fresher than the
# NOAA GHCN bulk per-station files allow (which lag by weeks between
# manual download.py/ingest.py runs).
#
# One request per date returns ALL ~8,000+ reporting Canadian stations at
# once (verified: limit=10000 is accepted), so a LOOKBACK_DAYS-day pull is
# just a handful of requests total - not one per station.
#
# Station ID mapping: GHCN's Canadian station IDs are "CA" + padding +
# the 7-digit ECCC climate identifier, e.g. CA001100119 -> climate
# identifier 1100119. Verified against the climate-stations collection
# for multiple stations (different provinces/prefixes) before relying on
# it here - it's just the last 7 characters of the 11-character GHCN ID.
#
# ECCC's near-real-time values are provisional and can later be revised;
# GHCN's periodic full re-download remains the source of truth for
# training data (its ON CONFLICT DO UPDATE upsert will naturally
# supersede a provisional value once that date's finalized GHCN reading
# is re-ingested). This script only exists to keep the live-forecast
# anchor (features_latest) fresh between those full refreshes.

import time
import requests
from datetime import date, timedelta
from ingestion.ingest import get_connection, upsert_observations, refresh_features

GEOMET_URL = "https://api.weather.gc.ca/collections/climate-daily/items"
LOOKBACK_DAYS = 14   # ECCC daily values can take a few days to finalize


def climate_id_for(station_id: str) -> str:
    return station_id[-7:]


#one station_id per climate identifier - only need it for stations we track
def load_station_id_by_climate_id(conn) -> dict:
    with conn.cursor() as cur:
        cur.execute("SELECT station_id FROM dim_stations")
        station_ids = [r[0] for r in cur.fetchall()]
    return {climate_id_for(sid): sid for sid in station_ids}


#fetch every Canadian station's reading for one date, filtered down to
#the stations we track (station_id_by_climate_id)
def fetch_day(day: date, station_id_by_climate_id: dict) -> list:
    resp = requests.get(GEOMET_URL, params={
        "datetime": day.isoformat(),
        "limit": 10000,
        "f": "json",
    }, timeout=60)
    resp.raise_for_status()
    features = resp.json().get("features", [])

    rows = []
    for feat in features:
        p = feat["properties"]
        station_id = station_id_by_climate_id.get(p.get("CLIMATE_IDENTIFIER"))
        if station_id is None:
            continue   # not one of our tracked GHCN stations

        tmax = p.get("MAX_TEMPERATURE")
        tmin = p.get("MIN_TEMPERATURE")
        prcp = p.get("TOTAL_PRECIPITATION")
        snow_cm = p.get("TOTAL_SNOW")

        if tmax is None and tmin is None and prcp is None and snow_cm is None:
            continue   # provisional placeholder row, nothing usable yet

        rows.append({
            "station_id": station_id,
            "date_id": day,
            "tmax_c": tmax,
            "tmin_c": tmin,
            "prcp_mm": prcp,
            "snow_mm": round(snow_cm * 10, 1) if snow_cm is not None else None,  # cm -> mm
            "snwd_mm": None,
            # ECCC flags an estimated/suspect value with a non-blank code;
            # mirror that as a non-null qflag so the same
            # "WHERE qflag IS NULL" filter in features_daily_base excludes
            # it, consistent with how GHCN's own flags are treated.
            "tmax_qflag": p.get("MAX_TEMPERATURE_FLAG") or None,
            "tmin_qflag": p.get("MIN_TEMPERATURE_FLAG") or None,
            "prcp_qflag": p.get("TOTAL_PRECIPITATION_FLAG") or None,
        })
    return rows


if __name__ == "__main__":
    conn = get_connection()
    if conn is None:
        raise RuntimeError("Could not connect to database.")

    print("Loading tracked stations...")
    station_id_by_climate_id = load_station_id_by_climate_id(conn)
    print(f"  Tracking {len(station_id_by_climate_id)} stations.")

    today = date.today()
    total_rows = 0
    for i in range(LOOKBACK_DAYS, 0, -1):
        day = today - timedelta(days=i)
        rows = fetch_day(day, station_id_by_climate_id)
        if rows:
            upsert_observations(conn, rows)
            total_rows += len(rows)
        print(f"  {day}: {len(rows)} station-readings upserted.")
        time.sleep(0.3)

    print(f"\nUpserted {total_rows:,} total rows from ECCC near-real-time data.")

    print("Refreshing feature view...")
    refresh_features(conn)

    conn.close()
    print("\nDone.")
