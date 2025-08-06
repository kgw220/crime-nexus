"""
This script takes the most recent crime data, merged data (without cluster labels), and labeled
merged data (with cluster labels) sourced from the daily pipeline,  and makes a streamlit
application to visualize the data with maps. This is based off the `cluster_hotspot_mapping.ipynb`
notebook, but with cleaner code. This also includes extra functionality, such as being able to
subset data to particular attributes during hotspot analysis, and allowing the user in the streamlit
application to download specific clusters of data for further analysis.
"""

import folium
import geopandas as gpd
import matplotlib.colors
import matplotlib.pyplot as plt
import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from streamlit_folium import st_folium

from streamlit_utils import (
    load_data_from_directory,
    plot_cluster_outlines,
    plot_hotspot_analysis,
    plot_recent_crimes,
    add_legend,
)

# Philadelphia county boundary GeoJSON
BOUNDARY = "https://raw.githubusercontent.com/blackmad/neighborhoods/master/philadelphia.geojson"

# Distance threshold for clustering crimes into outlines (in feet)
DISTANCE_THRESHOLD = 1000

st.set_page_config(layout="wide")

# --------------------------------------------------------------------------------------------------

# Add a sidebar for filtering and information
with st.sidebar:
    st.title("Crime Nexus")
    st.markdown("Use this app to visualize recent crime data and hotspot clusters in Philadelphia.")
    st.markdown("---")
    progress_bar = st.progress(0, text="Loading data")

script_dir = os.path.dirname(__file__)
data_directory_path = os.path.join(script_dir, "..", "data")

# Load the data
with st.spinner("Loading data for the crime map..."):
    crime_df, labeled_merged_df, merged_df = load_data_from_directory(data_directory_path)

progress_bar.progress(20, text="Setting up map structure...")

# Extract crime type from the OHE'd columns
crime_type_cols = [col for col in crime_df.columns if col.startswith("crime_")]
for col in crime_type_cols:
    crime_df[col] = pd.to_numeric(crime_df[col], errors="coerce").fillna(0)
crime_df["crime_type"] = crime_df[crime_type_cols].idxmax(axis=1)

# Define a color map for each crime type
unique_types = crime_df["crime_type"].unique()
cmap_types = plt.get_cmap("tab20", len(unique_types))
color_map_types = {
    crime: matplotlib.colors.rgb2hex(cmap_types(i)) for i, crime in enumerate(unique_types)
}

# Create a dynamic color map for the clustered crimes
unique_clusters = sorted(labeled_merged_df["cluster_label"].unique())
cmap_clusters = plt.get_cmap("jet", len(unique_clusters))
color_map_clusters = {
    cluster: matplotlib.colors.rgb2hex(cmap_clusters(i))
    for i, cluster in enumerate(unique_clusters)
}

# Create a mapping for alphabetical cluster labels
cluster_nums = sorted(unique_clusters)
alpha_labels = {num: chr(65 + i) for i, num in enumerate(cluster_nums)}
# Apply the new labels to the dataframe
labeled_merged_df["cluster_alpha_label"] = labeled_merged_df["cluster_label"].map(alpha_labels)

progress_bar.progress(40, text="Initializing map...")

# Load Philadelphia boundary GeoJSON
philly_gdf = gpd.read_file(BOUNDARY)
min_lon, min_lat, max_lon, max_lat = philly_gdf.total_bounds
map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

# Initialize Folium map centered at mean lat/lon of crime
m_crime = folium.Map(
    location=[crime_df["lat"].mean(), crime_df["lon"].mean()],
    zoom_start=12,
    max_bounds=map_bounds,
    min_zoom=12,
    width="100%",
    height="100%",
)
# Add the Philadelphia boundary outline to the map
folium.GeoJson(
    philly_gdf[["geometry"]],
    style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0.0},
    name="Philadelphia Boundary",
).add_to(m_crime)

progress_bar.progress(55, text="Adding crime data to map...")

# Add recent crime, cluster outline, and hotspot layers to the map
m_crime = plot_recent_crimes(m_crime, crime_df, color_map_types)

progress_bar.progress(70, text="Adding crime clusters to map...")

m_crime = plot_cluster_outlines(
    m_crime, labeled_merged_df, color_map_clusters, alpha_labels, DISTANCE_THRESHOLD
)

progress_bar.progress(85, text="Adding crime hotspots to map...")

m_crime = plot_hotspot_analysis(m_crime, crime_df, philly_gdf)

progress_bar.progress(90, text="Adding final touches to map...")

# Add a control for controlling the layers
folium.LayerControl().add_to(m_crime)

# Add legend for the crime types and cluster labels
m_crime = add_legend(m_crime, color_map_types, color_map_clusters, alpha_labels)

progress_bar.progress(100, text="Map is ready!")

# Render the Folium map to an HTML string
map_html_string = m_crime._repr_html_()


# Display the HTML string in Streamlit
components.html(map_html_string, width=5000, height=3000)
