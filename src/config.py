"""
This script sets up the necessary configurations and constants for retrieving
weather, crime, and census data. It includes API tokens, URLs for data sources, FIPS codes, the
probability threshold for clustering, and the hyperparameter search space, among other things.
"""

import os

from hyperopt import hp

# Read API tokens from environment variables
NOAA_TOKEN = os.getenv("NOAA_TOKEN")
CENSUS_TOKEN = os.getenv("CENSUS_TOKEN")
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY")
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET")
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN")

# Number of years of data to collect
# NOTE: This is set to a low number to keep the data size manageable and to ensure stability in streamlit
YEARS_TO_COLLECT = 1

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

# Folder path to store files in Dropbox
FOLDER_PATH = "/crime_nexus"

# Hyperopt space
SEARCH_SPACE = {
    "n_neighbors": hp.quniform("n_neighbors", 15, 150, 10),
    "min_dist": hp.uniform("min_dist", 0.0, 0.5),
    "n_components": hp.quniform("n_components", 5, 50, 1),
    "min_cluster_size": hp.quniform("min_cluster_size", 3000, 5000, 500),
    "min_samples": hp.quniform("min_samples", 10, 200, 5),
}

# How many days of data to keep at once
RUN_RETENTION_DAYS = 7

# Set the number of evaluations the TPE algorithm will perform
# NOTE: Ideally, this would be set higher, but each run takes some time, and GitHub
# Actions (on the free tier) has a cap of 6 hours for the entire script, so I set this number to be
# lower to ensure the whole daily pipeline will run in <6 hours. This can be increased if I did have
# a paid tier.
NUM_EXPERIMENT_EVALS = 50

# Number of top High Quality clusters to keep
HQ_CLUSTER_LIMIT = 10

# Random seed to get the psuedo random number generator
RANDOM_SEED = 42

# Philadelphia county boundary GeoJSON
BOUNDARY = "https://raw.githubusercontent.com/blackmad/neighborhoods/master/philadelphia.geojson"
