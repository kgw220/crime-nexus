"""
This script takes the most recent crime data, merged data (without cluster labels), and labeled
merged data (with cluster labels) sourced from the daily pipeline,  and makes a streamlit
application to visualize the data with maps. This is based off the `cluster_hotspot_mapping.ipynb`
notebook, but with cleaner code. This also includes extra functionality, such as being able to
subset data to particular attributes during hotspot analysis, and allowing the user in the streamlit
application to download specific clusters of data for further analysis.
"""

import concurrent.futures
import folium
import geopandas as gpd
import matplotlib.colors
import matplotlib.pyplot as plt
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import sys
import traceback

from streamlit_utils import download_and_save_map

REPO_OWNER = "kgw220"
REPO_NAME = "crime-nexus"
ARTIFACT_NAME = "map-html-artifact"
BRANCH = "main"
MAP_FILE_PATH = "map_cache.html"

GIT_TOKEN = st.secrets["GITHUB_TOKEN"]
# --------------------------------------------------------------------------------------------------


st.title("TEST")

downloaded_path = download_and_save_map(
    owner=REPO_OWNER,
    repo=REPO_NAME,
    artifact_name=ARTIFACT_NAME,
    branch=BRANCH,
    token=GIT_TOKEN,
    path=MAP_FILE_PATH,
)

# Now, read the file from the local path if it exists
if downloaded_path and os.path.exists(downloaded_path):
    with open(downloaded_path, "r", encoding="utf-8") as f:
        map_html_string = f.read()

    st.header("Displaying Map")
    components.html(map_html_string, height=800, scrolling=True)
else:
    st.warning("Map file could not be loaded. Please check the logs.")
