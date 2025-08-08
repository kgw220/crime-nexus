"""
Related utilies for loading data from the pipeline, and plotting map layers for the crime-nexus
project.
"""

import io
import os
import zipfile

import dropbox
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import streamlit as st

from branca.element import Element
from folium.plugins import MarkerCluster
from pysal.explore import esda
from pysal.lib import weights
from shapely.geometry import MultiPoint, Polygon
from sklearn.cluster import DBSCAN
from typing import List, Tuple, Union


# Define functions regarding retrieving the data ---------------------------------------------------


# @st.cache_data
# def load_pickle_by_prefix(folder: str, prefix: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
#     """
#     Finds and loads a pickle file from a folder based on a filename prefix.

#     Parameters:
#     ----------
#     folder: str
#         The directory to search for the file
#     prefix: str
#         The starting string of the filename to find

#     Returns:
#     -------
#     Union[pd.DataFrame, gpd.GeoDataFrame]:
#         The loaded data from the pickle file, which can be either a pandas or geopandas DataFrame

#     Raises:
#     ------
#     FileNotFoundError:
#         If no file with the specified prefix is found
#     """
#     matches = [f for f in os.listdir(folder) if f.startswith(prefix)]
#     if not matches:
#         raise FileNotFoundError(f"No file with prefix '{prefix}' found in {folder}")
#     file_path = os.path.join(folder, matches[0])

#     print(f"Loading file: {file_path}")
#     df = pd.read_pickle(file_path)

#     return df


# @st.cache_data(show_spinner=False)
# def load_data_from_directory(
#     data_dir: str = "data",
# ) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
#     """
#     Loads data files directly from a specified local directory.

#     This function assumes the necessary .pkl files are present in the
#     `data_dir` within the project. It loads the crime data, labeled
#     merged data, and merged data based on filename prefixes.

#     Parameters:
#     ----------
#     data_dir: str
#         The local directory where the data files are stored.

#     Returns:
#     -------
#     Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
#         A tuple containing the three loaded dataframes.
#     """
#     print("\nLoading dataframes from local directory...")
#     crime_df = load_pickle_by_prefix(data_dir, "crime_data")
#     labeled_df = load_pickle_by_prefix(data_dir, "labeled_merged_data")
#     merged_df = load_pickle_by_prefix(data_dir, "merged_data")
#     hotspot_df = load_pickle_by_prefix(data_dir, "hotspot_grid")

#     return crime_df, labeled_df, merged_df, hotspot_df


@st.cache_resource
def init_dropbox_client():
    """
    Initializes the Dropbox client, cached to run only once.
    """
    print("--- Initializing Dropbox Client ---")
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=st.secrets["DROPBOX_REFRESH_TOKEN"],
        app_key=st.secrets["DROPBOX_APP_KEY"],
        app_secret=st.secrets["DROPBOX_APP_SECRET"],
    )
    return dbx


@st.cache_data()
def _load_dataset_dropbox(
    _dropbox_client: dropbox.Dropbox, folder_path: str, filenames: List[str], prefix: str
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Finds, downloads, and loads a single CSV file from Dropbox into a DataFrame.

    This helper function searches a provided list of filenames for one starting with a
    specific prefix. It then constructs the full file path, downloads it from
    Dropbox, and parses the CSV content into a Pandas DataFrame.

    Parameters:
    ----------
    _dropbox_client: dropbox.Dropbox
        An authenticated Dropbox API client.
    folder_path: str
        The path to the parent folder in Dropbox where the file is located.
    filenames: List[str]
        A list of filenames available in the specified Dropbox folder.
    prefix: str
        The prefix of the filename to search for (e.g., "crime_").

    Returns:
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]
        A DataFrame containing the data

    Raises:
    ------
    FileNotFoundError
        If no file in `filenames` starts with the given `prefix`
    IOError
        If the file download from Dropbox fails or if the content cannot be
        parsed into a DataFrame
    """
    # Find the first file that matches the specified prefix
    matched_files = [f for f in filenames if f.startswith(prefix)]

    if not matched_files:
        raise FileNotFoundError(f"No file found in '{folder_path}' with prefix '{prefix}'")

    # Construct the full path for the file to be downloaded
    file_to_download = f"{folder_path}/{matched_files[0]}"

    # Download the file's binary content and read it into a Pandas DataFrame
    try:
        _, response = _dropbox_client.files_download(file_to_download)
        # Use io.BytesIO to treat the binary content as a file-like object
        return pd.read_pickle(io.BytesIO(response.content))

    except dropbox.exceptions.ApiError as err:
        raise IOError(
            f"Failed to download '{file_to_download}' from Dropbox. Error: {err}"
        ) from err


@st.cache_data(show_spinner=False, ttl="30m")
def load_dropbox_datasets(
    _dropbox_client: dropbox.Dropbox, folder_path: str = "/crime_nexus"
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Load specific datasets from a Dropbox folder into four Pandas DataFrames or Geopandas
    GeoDataFrames.

    This function searches a given Dropbox folder for four CSV files, each identified
    by a unique prefix, and returns them as separate DataFrames. The required
    file prefixes are:
        - "crime_"
        - "hotspot_grid"
        - "merged_"
        - "labeled_merged_"

    Parameters:
    ----------
    dropbox_client: dropbox.Dropbox
        An authenticated Dropbox client/session
    folder_path : str
        Path to the folder in Dropbox where the files are stored

    Returns:
    -------
    Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        A tuple containing four DataFrames in the following order:
        1. Crime dataset
        2. Hotspot grid dataset
        3. Merged dataset
        4. Labeled merged dataset

    Raises:
    ------
    FileNotFoundError
        If the specified folder doesn't exist or if any of the required files
        with the specified prefixes are missing
    IOError
        If a file fails to download or be read by Pandas
    """
    # First, get a list of all file names in the target folder
    try:
        result = _dropbox_client.files_list_folder(folder_path)
        filenames = [
            entry.name for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)
        ]
    except dropbox.exceptions.ApiError as err:
        raise FileNotFoundError(
            f"Could not access Dropbox folder '{folder_path}'. Error: {err}"
        ) from err

    # Load each dataset
    crime_df = _load_dataset_dropbox(_dropbox_client, folder_path, filenames, "crime_")
    hotspot_grid_df = _load_dataset_dropbox(_dropbox_client, folder_path, filenames, "hotspot_grid")
    merged_df = _load_dataset_dropbox(_dropbox_client, folder_path, filenames, "merged_")
    labeled_merged_df = _load_dataset_dropbox(
        _dropbox_client, folder_path, filenames, "labeled_merged_"
    )

    return crime_df, hotspot_grid_df, merged_df, labeled_merged_df


# Define functions regarding mapping the data ------------------------------------------------------


def plot_recent_crimes(
    m: folium.Map, recent_crime: pd.DataFrame, color_map_types: dict
) -> folium.Map:
    """
    Plots recent crime incidents on the map as circles, colored by crime type

    Parameters:
    -----------
    m: folium.Map
        The Folium map object to add the layers to
    recent_crime: pd.DataFrame
        DataFrame containing recent crime data with a 'crime_type' column
    color_map_types: dict
        A dictionary mapping crime types to hex colors

    Returns:
    --------
    folium.Map
        The updated Folium map object with the recent crime layers
    """
    # Create layers for recent crimes (as circles)
    crime_agg_view = MarkerCluster(name="Recent Crimes (Aggregated Counts)").add_to(m)
    crime_det_view = folium.FeatureGroup(name="Recent Crimes By Type").add_to(m)

    # Add recent crime markers
    for _, row in recent_crime.iterrows():
        popup_text = row["crime_type"].replace("crime_", "")
        marker_args = {
            "location": [row["lat"], row["lon"]],
            "radius": 5,
            "color": color_map_types[row["crime_type"]],
            "fill": True,
            "fill_color": color_map_types[row["crime_type"]],
            "fill_opacity": 0.7,
            "popup": popup_text,
        }
        folium.CircleMarker(**marker_args).add_to(crime_agg_view)
        folium.CircleMarker(**marker_args).add_to(crime_det_view)

    return m


def plot_cluster_outlines(
    m: folium.Map,
    df_clustered: pd.DataFrame,
    color_map_clusters: dict,
    alpha_labels: dict,
    dbscan_threshold: int,
) -> folium.Map:
    """
    Plots intelligent cluster outlines and representative icons on the map with DBSCAN

    Parameters:
    -----------
    m: folium.Map
        The Folium map object to add the layers to
    df_clustered: pd.DataFrame
        DataFrame with clustered crime data, including 'cluster_label' and summary stats
    color_map_clusters: dict
        A dictionary mapping cluster labels to hex colors
    alpha_labels: dict
        A dictionary mapping numeric cluster labels to alphabetical labels
    dbscan_threshold: int
        An integer representing how many feet apart crime has to be, to be clustered together.
        Larger values mean capturing broader patterns, while smaller values mean capturing more
        local patterns

    Returns:
    --------
    folium.Map
        The updated Folium map object with the cluster outline layers
    """
    # Create new layers for the cluster outlines and icons
    cluster_outlines = folium.FeatureGroup(name="Cluster Outlines").add_to(m)
    cluster_icons = folium.FeatureGroup(name="Cluster Icons").add_to(m)

    # Create subclusters with the clustered data. This is so I can visualize the clusters of crime by
    # outlying distinct regions where crimes cluster together, instead of plotting each individuala
    # crime which would be a lot harder to view and cause lag on the map when rendering

    # Convert df_clustered to a GeoDataFrame and project it for distance calculations
    clustered_gdf = gpd.GeoDataFrame(
        df_clustered,
        geometry=gpd.points_from_xy(df_clustered.lon, df_clustered.lat),
        crs="EPSG:4326",  # Assume standard lat/lon
    ).to_crs(
        "EPSG:2272"
    )  # Project to a system that uses feet for accurate calculations

    # Group by the primary cluster label to process each cluster
    for primary_cluster_label, group in clustered_gdf.groupby("cluster_label"):

        # Extract coordinates for DBSCAN
        coords = np.array(list(zip(group.geometry.x, group.geometry.y)))

        # Need at least 3 points to form a hull
        if len(coords) < 3:
            continue

        # Run DBSCAN to find spatial sub-clusters
        db = DBSCAN(eps=dbscan_threshold, min_samples=3).fit(coords)
        group = group.copy()
        group["sub_cluster"] = db.labels_

        # Now, create an outline for each spatial sub-cluster
        for sub_cluster_label, sub_group in group.groupby("sub_cluster"):
            # Skip noise points from the sub-clustering and groups too small to form a shape
            if sub_cluster_label == -1 or len(sub_group) < 3:
                continue

            # Create a single MultiPoint object from all points in the sub-cluster
            multi_point = MultiPoint(sub_group["geometry"].tolist())

            # Calculate the convex hull (the outline)
            hull = multi_point.convex_hull

            # Calculate the center for the icon (still in projected CRS)
            centroid_proj = hull.centroid

            # Find the most common crime type in this sub-group
            sub_group_crime_cols = [col for col in sub_group.columns if col.startswith("crime_")]
            # Get the full column name of the most frequent crime
            dominant_crime_col = sub_group[sub_group_crime_cols].sum().idxmax()
            most_common_crime = dominant_crime_col.replace("crime_", "")

            # Label the crime if it is extremely dominant or not
            dominant_crime_count = sub_group[dominant_crime_col].sum()
            dominant_crime_pct = dominant_crime_count / len(sub_group)
            dominance_indicator = "✓" if dominant_crime_pct >= 0.5 else "⚠️"

            # Calculate average median income, poverty rate, and population density
            avg_median_income = sub_group["income_median"].mean()
            avg_poverty_rate = sub_group["poverty_rate"].mean()
            avg_pop_density = sub_group["pop_density_sq_km"].mean()

            # Get color and info for this cluster (using the primary label for color)
            icon_color_hex = color_map_clusters.get(primary_cluster_label, "#000000")

            # Define HTML for detailed popup text with the statistics
            popup_text = f"""
            <b>Cluster {alpha_labels[primary_cluster_label]} (Sub-Group)</b><br>
            Count: {len(sub_group)}<br>
            <hr style='margin: 2px;'>
            <b>Dominant Crime:</b> {most_common_crime} ({dominant_crime_pct:.0%}) {dominance_indicator}<br>
            <b>Avg. Median Income:</b> ${avg_median_income:,.0f}<br>
            <b>Avg. Poverty Rate:</b> {avg_poverty_rate:.1%}<br>
            <b>Avg. Pop. Density:</b> {avg_pop_density:,.0f}/km²
            """

            # Convert hull back to lat/lon for plotting on Folium map
            hull_gdf = gpd.GeoDataFrame([1], geometry=[hull], crs="EPSG:2272").to_crs("EPSG:4326")

            # Add the outline to its layer
            folium.GeoJson(
                hull_gdf.geometry.to_json(),
                style_function=lambda x, color=icon_color_hex: {
                    "fillColor": color,
                    "color": color,
                    "weight": 2,
                    "fillOpacity": 0.2,
                },
            ).add_to(cluster_outlines)

            # Convert centroid to lat/lon for the marker
            centroid_gdf = gpd.GeoDataFrame([1], geometry=[centroid_proj], crs="EPSG:2272").to_crs(
                "EPSG:4326"
            )
            centroid_latlon = [
                centroid_gdf.geometry.y.iloc[0],
                centroid_gdf.geometry.x.iloc[0],
            ]

            # Define the HTML for the cluster outline icon
            icon_html = f'<div style="text-align: center; color: {icon_color_hex};"><i class="fa fa-tag fa-2x"></i></div>'

            # Add the representative icon to each layer
            folium.Marker(
                location=centroid_latlon,
                icon=folium.DivIcon(html=icon_html, icon_size=(24, 24), icon_anchor=(12, 12)),
                popup=popup_text,
            ).add_to(cluster_icons)

    return m


def plot_hotspot_analysis(m: folium.Map, hotspot_grid: gpd.GeoDataFrame) -> folium.Map:
    """
    Performs a hotspot analysis on the given data and adds it as a choropleth layer

    Parameters:
    -----------
    m: folium.Map
        The Folium map object to add the layers to
    hotspot_grid: gpd.GeoDataFrame
        A GeoDataFrame containing the grid cells with their z-scores

    Returns:
    --------
    folium.Map
        The updated Folium map object with the hotspot layer
    """
    # Create a Choropleth layer for the hotspots; The reversed red/blue colormap is used, so
    # hotspots (identified with darker red) are shown with a higher z score, and coldspots
    # (identified with blue) are shown with a negative/near zero z score.
    chlorpleth = folium.Choropleth(
        geo_data=hotspot_grid.to_crs("EPSG:4326"),
        name="Hotspots",
        data=hotspot_grid,
        columns=["index", "z_score"],
        key_on="feature.id",
        fill_color="RdBu_r",
        fill_opacity=0.6,
        line_opacity=0.2,
        legend_name="Hotspot Intensity",
        highlight=True,
    ).add_to(m)

    # Add CSS for formatting the chlorpleth legend, so it stands out better with the background of
    # the map
    legend_css = """
    <style>
        .legend {
            background-color: rgba(255, 255, 255, 0.7);
            padding: 6px 12px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            line-height: 1.4;
            box-shadow: 0 0 4px rgba(0,0,0,0.4);
            border: 1px solid rgba(0,0,0,0.2);
        }

        .legend text, .legend-title {
            text-shadow:
                0 0 2px #fff,
                -1px -1px 0 #fff,
                1px -1px 0 #fff,
                -1px 1px 0 #fff,
                1px 1px 0 #fff;
        }
    </style>
    """
    m.get_root().html.add_child(Element(legend_css))

    return m


def add_legend(
    m: folium.Map, color_map_types: dict, color_map_clusters: dict, alpha_labels: dict
) -> folium.Map:
    """
    Adds a legend to the Folium map for better visualization of the data layers.

    Parameters:
    -----------
    m: folium.Map
        The Folium map object to which the legend will be added
    color_map_types: dict
        A dictionary mapping crime types to hex colors
    color_map_clusters: dict
        A dictionary mapping cluster labels to hex colors
    alpha_labels: dict
        A dictionary mapping numeric cluster labels to alphabetical labels

    Returns:
    --------
    folium.Map
        The updated Folium map object with the legend added
    """
    # Add adaptive CSS for the legend to make it responsive based on screen size
    adaptive_css = """
    <style>
        #crime-legend { width: 250px; height: 400px; font-size: 14px; }
        #cluster-legend { width: 150px; height: 225px; font-size: 14px; }
        @media (max-width: 1200px) {
            #crime-legend { width: 180px; height: 250px; font-size: 12px; }
            #cluster-legend { width: 130px; height: 200px; font-size: 12px; }
        }
    </style>
    """
    m.get_root().header.add_child(folium.Element(adaptive_css))

    toggle_js = """
    <script type='text/javascript'>
      function toggleLegends() {
        var crimeLegend = document.getElementById('crime-legend');
        var clusterLegend = document.getElementById('cluster-legend');
        if (crimeLegend.style.display === 'none' || crimeLegend.style.display === '') {
          crimeLegend.style.display = 'block';
          clusterLegend.style.display = 'block';
        } else {
          crimeLegend.style.display = 'none';
          clusterLegend.style.display = 'none';
        }
      }
    </script>"""

    toggle_button_html = """
    <div style="position: fixed; top: 20px; right: 20px; z-index:10000;">
      <button onclick="toggleLegends()" style="background-color: white; border: 2px solid grey; 
      padding: 5px 10px; font-size: 14px; cursor: pointer;">Toggle Legends</button>
    </div>"""

    # Add the JavaScript and the button to the map
    m.get_root().html.add_child(folium.Element(toggle_js))
    m.get_root().html.add_child(folium.Element(toggle_button_html))

    # Setup HTML for crime type legend
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
    m.get_root().html.add_child(folium.Element(full_legend_html))

    # Set up HTML for cluster legend
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
        legend_cluster_items += f'&nbsp; <i class="fa fa-{icon_shape}" style="color:{color}"></i> &nbsp; {label_text}<br>'
    legend_cluster_html_end = "</div></div>"
    full_legend_cluster_html = (
        legend_cluster_html_start + legend_cluster_items + legend_cluster_html_end
    )
    m.get_root().html.add_child(folium.Element(full_legend_cluster_html))

    return m


# Define functions for other tasks -----------------------------------------------------------------


@st.cache_data
def reverse_ohe_and_clean(_df, ohe_prefixes):
    """
    Reverses one-hot-encoding for specified columns, adds human-readable columns,
    and cleans up the DataFrame for display.

    Parameters:
    ----------
    df: pd.DataFrame
        The DataFrame containing one-hot-encoded columns

    ohe_prefixes: list
        A list of prefixes for the one-hot-encoded columns to reverse
    """
    processed_df = _df.copy()

    for prefix in ohe_prefixes:
        # Identify OHE columns based on the current prefix
        ohe_cols = [col for col in processed_df.columns if col.startswith(prefix)]

        if ohe_cols:
            # Create a new column with the original, human-readable name
            new_col_name = prefix.replace("_", "")
            processed_df[new_col_name] = (
                processed_df[ohe_cols].idxmax(axis=1).str.replace(prefix, "")
            )

            # Drop the original OHE columns for a cleaner output
            processed_df = processed_df.drop(columns=ohe_cols, errors="ignore")

    return processed_df
