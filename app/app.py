"""
This script takes the most recent crime data, merged data (without cluster labels), and labeled
merged data (with cluster labels) sourced from the daily pipeline,  and makes a streamlit
application to visualize the data with maps. This is based off the `cluster_hotspot_mapping.ipynb`
notebook, but with cleaner code. This also includes extra functionality, such as being able to
subset data to particular attributes during hotspot analysis, and allowing the user in the streamlit
application to download specific clusters of data for further analysis.
"""

import matplotlib.colors
import matplotlib.pyplot as plt
import os
import pandas as pd
import streamlit as st
import sys

# Setup file path to root directory of repository
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from src.data_utils import (
    get_latest_github_artifact_data,
    plot_recent_crimes,
    plot_cluster_outlines,
    plot_hotspot_analysis,
)
from src.config import GITHUB_REPO, WORKFLOW_FILE_NAME, ARTIFACT_NAME, GITHUB_TOKEN

st.title("Streamlit test")

# Load the data from the artifact from the most recent run
crime_df, labeled_df, merged_df = get_latest_github_artifact_data(
    repo_name=GITHUB_REPO,
    workflow_filename=WORKFLOW_FILE_NAME,
    artifact_name=ARTIFACT_NAME,
    github_token=GITHUB_TOKEN,
)
print(crime_df)
