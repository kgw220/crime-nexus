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

from streamlit_folium import st_folium

from streamlit_utils import (
    init_dropbox_client,
    get_dropbox_folder_metadata,
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

# Add custom CSS to style the Streamlit app; blue tabs and custom padding
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 5rem;
            padding-bottom: 3rem;
            padding-left: 5rem;
            padding-right: 5rem;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-bottom: 3px solid #00BFFF !important; 
            color: #00BFFF;
        }

        .stTabs [data-baseweb="tab-list"] button[aria-selected="false"] {
            color: #E5E5E5; 
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: #00BFFF !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------------------------

# Initialize the Dropbox client
dbx = init_dropbox_client()

# Get folder signature to check if the folder has changed
folder_meta = get_dropbox_folder_metadata(dbx, FOLDER_PATH)

if folder_meta:
    current_folder_signature = str(folder_meta)
    print(f"Current folder signature: {current_folder_signature}")

    with st.spinner("Updating the crime map with new data..."):
        crime_df, hotspot_grid, merged_df, labeled_merged_df = load_dropbox_datasets(
            dbx, current_folder_signature, FOLDER_PATH
        )
else:
    st.warning("Could not load the data. Please refresh the application and try again.")
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <h1 style="
            font-family: 'Roboto Mono', monospace;
            color: #E5E5E5;
            text-align: center;
            margin-top: -15px; /* Pulls the title up */
            margin-bottom: 0px; /* Removes space below the title */
            text-shadow: 
                0 0 5px #00BFFF, 
                0 0 10px #00BFFF, 
                0 0 20px #00BFFF, 
                0 0 40px #1E90FF, 
                0 0 80px #1E90FF;
        ">
        Crime Nexus </h1>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "This app visualizes recent crime data, clusters, and hotspots in Philadelphia. "
        "Use the Layer Control at the top right corner of the map to toggle layers on and off for"
        " better visibility on how crime is distributed in the city. Clusters highlight patterns "
        " with crime (with consideration with census and weather data), and hotspots highlight "
        "areas that are statistically significant to have more crime. ***Data updates daily by 6PM "
        "EST*.***"
    )
    st.markdown("---")
    st.markdown("📊Date Ranges for Current Data📊")
    st.markdown(
        f"➼ Recent crime data on the map reflects crimes recorded on "
        f"**{crime_df['dispatch_date'].min().strftime('%Y-%m-%d')}** \n\n"
        f"➼ Cluster & Hotspot data on the map reflects data from "
        f"**{labeled_merged_df['dispatch_date'].min().strftime('%Y-%m-%d')} to "
        f"{labeled_merged_df['dispatch_date'].max().strftime('%Y-%m-%d')}**"
    )
    st.markdown("---")
    st.markdown(
        "**Data is updated with an automated daily script. However, there may be connection"
        " issues that lead to data not being updated on a given day. In this case, please"
        " be patient and wait until the next day. In the case where data is several days "
        "outdated, please raise an issue in the project repo.*"
    )

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


# Cache map creation so it does not get recreated every time the app runs
@st.cache_data(show_spinner=False)
def create_and_render_map(crime_df, _labeled_merged_df, _hotspot_grid, _cache_key):
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


# Map viewer tab
with tab1:
    map_html_string = create_and_render_map(
        crime_df, labeled_merged_df, hotspot_grid, current_folder_signature
    )
    # Display the HTML string in Streamlit
    components.html(map_html_string, width=1600, height=700)

# Data downloaded tab
with tab2:
    # Undo the one-hot encoding and clean the data for download
    labeled_merged_df_clean = reverse_ohe_and_clean(
        labeled_merged_df, ["crime_", "psa_", "district_"], current_folder_signature
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
        filtered_df = filtered_df.reset_index(drop=True)
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
