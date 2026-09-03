from pathlib import Path

# paths
BASE_DIR  = Path(__file__).parent 
RAW_DIR   = BASE_DIR / "data" / "raw"
DAILY_DIR = RAW_DIR / "daily"

#urls
URL_BYSTATION = "https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-stations.txt"
URL_INV  = "https://noaa-ghcn-pds.s3.amazonaws.com/ghcnd-inventory.txt"
URL_DAILYSTATION = "https://noaa-ghcn-pds.s3.amazonaws.com/csv/by_station/"

# constants
PREFIX = "CA"
MIN_YRS      = 20
ELEMENTS       = {"TMAX", "TMIN", "PRCP", "SNOW", "SNWD"}
BAD_QFLAGS     = {"D","G","I","K","L","M","N","O","R","S","T","W","X","Z"}

# database config
DB_CONFIG = {
    "host":     "localhost",
    "port":     5433,
    "dbname":   "climate_dw",
    "user":     "climate",
    "password": "climate123",
}