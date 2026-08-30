# Download the station inventory from NOAA
# Parse that fixed-width file into a dataframe
# Filter to Canadian stations with enough history
# Download one CSV per station

import time
import pandas as pd 
import requests 
from pathlib import Path 


# config 
RAW_DIR = Path("data/raw")
DAILY_DIR = RAW_DIR / "daily"
URL_BYSTATION = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
URL_DAILYSTATION = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/by_station/"
PREFIX = "CA"
MIN_YRS = 20


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
    column_names = ["station_id", "latitude", "longitude", "elevation", "province", "name", "gsn_flag", "hcn_flag", "wmo_id"]
    df = pd.read_fwf(path, colspecs=colspecs, names=column_names, header = None)    
    
    return df



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




