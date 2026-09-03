# Download the station inventory from NOAA
# Parse that fixed-width file into a dataframe
# Filter to Canadian stations with enough history
# Download one CSV per station

import time
import pandas as pd 
import requests 
from pathlib import Path 
from config import (
    RAW_DIR, DAILY_DIR,
    URL_BYSTATION, URL_DAILYSTATION, URL_INV,
    PREFIX, MIN_YRS
)


#helper to download file from url and save it to specified folder
def download_file(url, dest):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except requests.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


#helper to parse the station inventory fixed width file into pd dataframe 
def parse_station_inventory(path):
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 75), (76,79), (80, 85)]
    column_names = ["station_id", "lat", "lon", "elevation", "province", "name", "gsn_flag", "hcn_flag", "wmo_id"]
    df = pd.read_fwf(path, colspecs=colspecs, names=column_names, header = None)    
    
    return df


#download the station inventory to raw folder if it doesn't already exist
def step1_download_station_list():
    dest = RAW_DIR / "ghcnd-stations.txt"

    if dest.exists():
        print(f"Station list already exists at {dest}, skipping download.")
        return dest

    print("Downloading station inventory...")
    success = download_file(URL_BYSTATION, dest)

    if not success:
        raise RuntimeError("Failed to download station inventory after 3 attempts.")

    print(f"Saved to {dest}")
    return dest


#parse station inventory and filter for only Canadian stations with at least MIN_YRS of TMAX and TMIN data
def step2_filter_canada_stations(station_file):
    print("Filtering Canadian stations...")
    df = parse_station_inventory(station_file)

    # keep only canadian stations
    canada = df[df["station_id"].str.startswith(PREFIX)].copy()

    inv_dest = RAW_DIR / "ghcnd-inventory.txt"

    if not inv_dest.exists():
        print("Downloading element inventory...")
        download_file(URL_INV, inv_dest)

    inv_colspecs = [(0,11),(12,20),(21,30),(31,35),(36,40),(41,45)] 
    inv_names = ["station_id","lat","lon","element","first_year","last_year"]
    inv = pd.read_fwf(inv_dest, colspecs=inv_colspecs, names=inv_names, header=None)

    # keep canadian stations that have TMAX with at least MIN_YRS of data
    has_tmax = inv[
        (inv["station_id"].str.startswith(PREFIX)) &
        (inv["element"] == "TMAX") &
        ((inv["last_year"] - inv["first_year"]) >= MIN_YRS)
    ]["station_id"].unique()

    # also require TMIN
    has_tmin = inv[
        (inv["station_id"].str.startswith(PREFIX)) &
        (inv["element"] == "TMIN")
    ]["station_id"].unique()

    # station must have both
    good = set(has_tmax) & set(has_tmin)
    canada = canada[canada["station_id"].isin(good)].copy()

    # attach first_year and last_year from TMAX rows
    tmax_dates = inv[
        (inv["station_id"].isin(good)) &
        (inv["element"] == "TMAX")
    ][["station_id", "first_year", "last_year"]]

    canada = canada.merge(tmax_dates, on="station_id", how="left")

    # save filtered list
    out = RAW_DIR / "canada_stations.csv"
    canada.to_csv(out, index=False)

    print(f"Found {len(canada)} Canadian stations with {MIN_YRS}+ years of TMAX and TMIN.")
    print(f"Saved to {out}")
    return canada


# download one CSV per station
def step3_download_daily_data(stations, limit=None):
    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    station_ids = stations["station_id"].tolist()

    #small batch first
    if limit:
        station_ids = station_ids[:limit]
        print(f"[DEV MODE] Downloading {limit} stations only.")

    print(f"Downloading daily data for {len(station_ids)} stations...")
    success, failed = 0, []

    for i, sid in enumerate(station_ids, 1):
        dest = DAILY_DIR / f"{sid}.csv"

        if dest.exists():
            success += 1
            continue

        url = f"{URL_DAILYSTATION}{sid}.csv"
        ok  = download_file(url, dest)

        if ok:
            success += 1
        else:
            failed.append(sid)
            print(f"  FAILED: {sid}")

        time.sleep(0.3)

        if i % 10 == 0:
            print(f"  {i}/{len(station_ids)} done...")

    print(f"\nDone: {success} succeeded, {len(failed)} failed.")

    if failed:
        fail_path = RAW_DIR / "failed_stations.txt"
        fail_path.write_text("\n".join(failed))
        print(f"Failed IDs saved to {fail_path}")



if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    station_file = step1_download_station_list()
    stations     = step2_filter_canada_stations(station_file)

    # start with 50 stations to verify the pipeline works
    # once confirmed, remove the limit= argument for the full run
    step3_download_daily_data(stations, limit = 500)

    print("\nPhase 1 complete.")
    print("Next: run ingestion/ingest.py to load into Postgres.")
