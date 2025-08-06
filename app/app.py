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
import sys
import traceback

from streamlit_folium import st_folium

from streamlit_utils import (
    load_data_from_directory,
    plot_recent_crimes,
    plot_cluster_outlines,
    plot_hotspot_analysis,
)

# Philadelphia county boundary GeoJSON
BOUNDARY = "https://raw.githubusercontent.com/blackmad/neighborhoods/master/philadelphia.geojson"

# Distance threshold for clustering crimes into outlines (in feet)
DISTANCE_THRESHOLD = 1000

# --------------------------------------------------------------------------------------------------

# Load the data

script_dir = os.path.dirname(__file__)
data_directory_path = os.path.join(script_dir, "..", "data")

with st.spinner("Loading data for the crime map..."):
    crime_df, labeled_merged_df, merged_df = load_data_from_directory(data_directory_path)


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

# Load Philadelphia boundary GeoJSON
philly_gdf = gpd.read_file(BOUNDARY)
min_lon, min_lat, max_lon, max_lat = philly_gdf.total_bounds
map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

# Create Folium map of crime, centered at mean lat/lon
m_crime = folium.Map(
    location=[crime_df["lat"].mean(), crime_df["lon"].mean()],
    zoom_start=12,
    max_bounds=map_bounds,
    min_zoom=12,
)
# Add the Philadelphia boundary outline to the map
folium.GeoJson(
    philly_gdf[["geometry"]],
    style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0.0},
    name="Philadelphia Boundary",
).add_to(m_crime)

print("Initialized map")
# Add recent crime, cluster outline, and hotspot layers to the map
m_crime = plot_hotspot_analysis(m_crime, merged_df, philly_gdf)

# m_crime = plot_recent_crimes(m_crime, crime_df, color_map_types)
print("Plotted recent crimes on map")
# m_crime = plot_cluster_outlines(
#     m_crime, labeled_merged_df, color_map_clusters, alpha_labels, DISTANCE_THRESHOLD
# )
print("Plotted cluster outlines on map")
# with st.spinner("Building hotspot layer..."):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future = executor.submit(plot_hotspot_analysis, m_crime, crime_df, philly_gdf)
#         try:
#             m_crime = future.result()
#         except Exception as e:
#             st.error("Error during hotspot analysis:")
#             st.text(traceback.format_exc())
# m_crime = plot_hotspot_analysis(m_crime, merged_df, philly_gdf)
print("Plotted hotspot analysis on map")

print("Added layers to map")
# Add a control for controlling the layers
folium.LayerControl().add_to(m_crime)
print("Added layer control to map")
# Setting up HTML for crime type legend
legend_html_start = """
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 250px; height: 400px; 
     border:2px solid grey; z-index:9998; font-size:14px;
     background-color:white; padding: 10px;">
     <b>Crime Type Legend</b><br>
     <div style="height: 90%; overflow-y: auto;">
     """
legend_items = ""
for crime_type, color in color_map_types.items():
    clean_name = crime_type.replace("crime_", "")
    legend_items += (
        f'&nbsp; <i class="fa fa-circle" style="color:{color}"></i> &nbsp; {clean_name}<br>'
    )
legend_html_end = "</div></div>"
full_legend_html = legend_html_start + legend_items + legend_html_end
m_crime.get_root().html.add_child(folium.Element(full_legend_html))

# Setting up HTML for cluster legend
legend_cluster_html_start = """
     <div style="position: fixed; 
     bottom: 50px; right: 50px; width: 150px; height: 225px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;">
     <b>Cluster Legend</b><br>
     <div style="height: 90%; overflow-y: auto;">
     """
legend_cluster_items = ""
for cluster_label, color in color_map_clusters.items():
    label_text = f"Cluster {alpha_labels[cluster_label]}" if cluster_label != -1 else "Noise"
    icon_shape = "tag" if cluster_label != -1 else "times"
    legend_cluster_items += (
        f'&nbsp; <i class="fa fa-{icon_shape}" style="color:{color}"></i> &nbsp; {label_text}<br>'
    )
legend_cluster_html_end = "</div></div>"
full_legend_cluster_html = (
    legend_cluster_html_start + legend_cluster_items + legend_cluster_html_end
)
m_crime.get_root().html.add_child(folium.Element(full_legend_cluster_html))

print("Added cluster legend to map")

# Render the map in Streamlit and display it
st_data = st_folium(m_crime, width=1500, height=600)
