"""
This script sets up the necessary configurations and constants for retrieving
weather, crime, and census data. It includes API tokens, URLs for data sources, and FIPS codes.
"""

import os

# Read API tokens from environment variables
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
CENSUS_TOKEN = os.getenv("CENSUS_TOKEN")

# Data sources
CARTO_URL = "https://phl.carto.com/api/v2/sql"
CRIME_TABLE_NAME = "incidents_part1_part2"
WEATHER_STATION_ID = "GHCND:USW00013739"
WEATHER_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
CENSUS_API_URL = "https://api.census.gov/data/2023/acs/acs5"
CENSUS_SHAPE_URL = (
    "https://hub.arcgis.com/api/v3/datasets/"
    "8bc0786524a4486bb3cf0f9862ad0fbf_0"
    "/downloads/data"
    "?format=geojson&spatialRefId=4326&where=1%3D1"
)

# Geographic FIPS codes
PHILLY_COUNTY_FIPS = "101"
PA_STATE_FIPS = "42"
