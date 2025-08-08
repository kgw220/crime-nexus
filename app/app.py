"""
This script takes the most recent crime data, merged data (without cluster labels), and labeled
merged data (with cluster labels) sourced from the daily pipeline,  and makes a streamlit
application to visualize the data with maps. This is based off the `cluster_hotspot_mapping.ipynb`
notebook, with cleaner code. This also includes extra functionality, such as being able to
subset data to particular attributes during hotspot analysis, and allowing the user in the streamlit
application to download specific clusters of data for further analysis.
"""

import dropbox
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

from streamlit_folium import st_folium

from streamlit_utils import (
    init_dropbox_client,
    load_dropbox_datasets,
    plot_cluster_outlines,
    plot_hotspot_analysis,
    plot_recent_crimes,
    add_legend,
    reverse_ohe_and_clean,
)

# Philadelphia county boundary GeoJSON
BOUNDARY = "https://raw.githubusercontent.com/blackmad/neighborhoods/master/philadelphia.geojson"
# Distance threshold for clustering crimes into outlines (in feet)
DISTANCE_THRESHOLD = 1000
# Folder path where data files are stored in Dropbox
FOLDER_PATH = "/crime_nexus"

st.set_page_config(layout="wide")

# --------------------------------------------------------------------------------------------------

# Move data loading outside of the tab blocks to prevent re-running on every tab switch
script_dir = os.path.dirname(__file__)
data_directory_path = os.path.join(script_dir, "..", "data")

# with st.spinner("Loading data for the crime map..."):
#     crime_df, labeled_merged_df, merged_df, hotspot_grid = load_data_from_directory(
#         data_directory_path
#     )

# Initialize the Dropbox client
dbx = init_dropbox_client()

with st.spinner("Loading data for the crime map..."):
    crime_df, hotspot_grid, merged_df, labeled_merged_df = load_dropbox_datasets(dbx, FOLDER_PATH)

# Add a sidebar for filtering and information
with st.sidebar:
    st.title("Crime Nexus")
    st.markdown(
        "This app visualizes recent crime data, clusters, and hotspots in Philadelphia. "
        "Use the Layer Control at the top right corner of the map to toggle layers on and off for"
        " better visibility on how crime is distributed in the city. Clusters highlight patterns "
        " with crime (with consideration with census and weather data), and hotspots highlight "
        "areas that are statistically significant to have more crime. Data updates daily by 6PM "
        "EST."
    )
    st.markdown("---")
    st.markdown(
        f"➼ Recent crime on the map shows crimes for "
        f"**{crime_df['dispatch_date'].min().strftime('%Y-%m-%d')}** \n\n"
        f"➼ Cluster/Hotspot data on the map shows data from "
        f"**{labeled_merged_df['dispatch_date'].min().strftime('%Y-%m-%d')} to "
        f"{labeled_merged_df['dispatch_date'].max().strftime('%Y-%m-%d')}**"
    )
    st.markdown("---")

# --- Main App UI ---
tab1, tab2 = st.tabs(["Map Viewer", "Data Downloader"])

# Create a mapping for alphabetical cluster labels in the global scope
unique_clusters = sorted(labeled_merged_df["cluster_label"].unique())
cluster_nums = sorted(unique_clusters)
alpha_labels = {num: chr(65 + i) for i, num in enumerate(cluster_nums)}
labeled_merged_df["cluster_alpha_label"] = labeled_merged_df["cluster_label"].map(alpha_labels)

# Extract crime type from the OHE'd columns in the global scope
crime_type_cols = [col for col in crime_df.columns if col.startswith("crime_")]
for col in crime_type_cols:
    crime_df[col] = pd.to_numeric(crime_df[col], errors="coerce").fillna(0)
crime_df["crime_type"] = crime_df[crime_type_cols].idxmax(axis=1)


@st.cache_resource(show_spinner=False)
def create_and_render_map(crime_df, _labeled_merged_df, _hotspot_grid):
    # --- Map rendering logic ---

    # Define a color map for each crime type
    unique_types = crime_df["crime_type"].unique()
    cmap_types = plt.get_cmap("tab20", len(unique_types))
    color_map_types = {
        crime: matplotlib.colors.rgb2hex(cmap_types(i)) for i, crime in enumerate(unique_types)
    }

    # Create a dynamic color map for the clustered crimes
    unique_clusters = sorted(_labeled_merged_df["cluster_label"].unique())
    cmap_clusters = plt.get_cmap("jet", len(unique_clusters))
    color_map_clusters = {
        cluster: matplotlib.colors.rgb2hex(cmap_clusters(i))
        for i, cluster in enumerate(unique_clusters)
    }

    # Create a mapping for alphabetical cluster labels
    cluster_nums = sorted(unique_clusters)
    alpha_labels = {num: chr(65 + i) for i, num in enumerate(cluster_nums)}

    # Load Philadelphia boundary GeoJSON
    philly_gdf = gpd.read_file(BOUNDARY)
    min_lon, min_lat, max_lon, max_lat = philly_gdf.total_bounds
    map_bounds = [[min_lat, min_lon], [max_lat, max_lon]]

    # Initialize Folium map centered at mean lat/lon of crime
    m_crime = folium.Map(
        location=[crime_df["lat"].mean(), crime_df["lon"].mean()],
        zoom_start=11,
        max_bounds=map_bounds,
        min_zoom=11,
        width="100%",
        height="100%",
    )
    # Add the Philadelphia boundary outline to the map
    folium.GeoJson(
        philly_gdf[["geometry"]],
        style_function=lambda x: {"color": "black", "weight": 2, "fillOpacity": 0.0},
        name="Philadelphia Boundary",
    ).add_to(m_crime)

    # Add recent crime, cluster outline, and hotspot layers to the map
    m_crime = plot_recent_crimes(m_crime, crime_df, color_map_types)
    m_crime = plot_cluster_outlines(
        m_crime, _labeled_merged_df, color_map_clusters, alpha_labels, DISTANCE_THRESHOLD
    )
    m_crime = plot_hotspot_analysis(m_crime, _hotspot_grid)

    # Add a control for controlling the layers
    folium.LayerControl().add_to(m_crime)

    # Add legend for the crime types and cluster labels
    m_crime = add_legend(m_crime, color_map_types, color_map_clusters, alpha_labels)

    # Render the Folium map to an HTML string
    map_html_string = m_crime._repr_html_()

    return map_html_string


with tab1:
    map_html_string = create_and_render_map(crime_df, labeled_merged_df, hotspot_grid)
    # Display the HTML string in Streamlit
    components.html(map_html_string, width=1600, height=700)

with tab2:
    # --- Data Downloader logic ---

    # Undo the one-hot encoding and clean the data for download
    labeled_merged_df_clean = reverse_ohe_and_clean(
        labeled_merged_df, ["crime_", "psa_", "district_"]
    )

    # Convert the datetime column to a string format for better readability
    labeled_merged_df_clean["dispatch_date"] = labeled_merged_df_clean["dispatch_date"].dt.strftime(
        "%Y/%m/%d"
    )
    labeled_merged_df_clean["dispatch_time"] = labeled_merged_df_clean["dispatch_time"].dt.strftime(
        "%H:%M:%S"
    )

    # Rename columns for better readability
    labeled_merged_df_clean.rename(
        columns={
            "district": "police_district",
            "psa": "police_service_area",
            "crime": "crime_type",
        },
        inplace=True,
    )

    # Remove unnecessary columns
    columns_to_remove = [
        "dispatch_date_dt",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
        "day_of_week_sin",
        "day_of_week_cos",
        "cluster_label",
        "geometry",
    ]
    processed_download_df = labeled_merged_df_clean.drop(columns=columns_to_remove, errors="ignore")

    st.markdown(
        "Select a cluster below to display and download its raw data. This is for those"
        " that would like to perform further analysis, as the map viewer is primarily for"
        "visualization purposes and some initial analysis."
    )

    # Get a list of unique cluster labels for the selectbox widget
    # We will use the alphabetical labels for a better user experience
    available_clusters = sorted(labeled_merged_df["cluster_alpha_label"].unique())

    # Use st.session_state to manage the selected cluster
    if "selected_cluster" not in st.session_state:
        st.session_state.selected_cluster = available_clusters[0]

    selected_cluster = st.selectbox(
        "Select a Cluster",
        options=available_clusters,
        index=available_clusters.index(st.session_state.selected_cluster),
        key="download_cluster_selector",
    )

    # Update session state on change
    st.session_state.selected_cluster = selected_cluster

    # Filter the DataFrame based on the selected cluster
    if selected_cluster:
        filtered_df = processed_download_df[
            processed_download_df["cluster_alpha_label"] == selected_cluster
        ]
        st.write(f"Displaying data for {len(filtered_df)} rows in cluster **{selected_cluster}**.")
        st.dataframe(filtered_df)

        # Create a download button for the filtered data
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv_data,
            file_name=f"clustered_crime_data_{selected_cluster}.csv",
            mime="text/csv",
        )
    else:
        st.info("Please select a cluster to display and download data.")
