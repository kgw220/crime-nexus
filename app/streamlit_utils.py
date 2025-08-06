"""
Related utilies for loading data from the pipeline, and plotting map layers for the crime-nexus
project.
"""

import io
import os
import zipfile

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
from typing import Tuple, Union


@st.cache_data
def load_pickle_by_prefix(folder: str, prefix: str) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """
    Finds and loads a pickle file from a folder based on a filename prefix.

    Parameters:
    ----------
    folder: str
        The directory to search for the file
    prefix: str
        The starting string of the filename to find

    Returns:
    -------
    Union[pd.DataFrame, gpd.GeoDataFrame]:
        The loaded data from the pickle file, which can be either a pandas or geopandas DataFrame

    Raises:
    ------
    FileNotFoundError:
        If no file with the specified prefix is found
    """
    matches = [f for f in os.listdir(folder) if f.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"No file with prefix '{prefix}' found in {folder}")
    file_path = os.path.join(folder, matches[0])

    print(f"Loading file: {file_path}")
    df = pd.read_pickle(file_path)

    return df


@st.cache_data(show_spinner=False)
def load_data_from_directory(
    data_dir: str = "data",
) -> Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """
    Loads data files directly from a specified local directory.

    This function assumes the necessary .pkl files are present in the
    `data_dir` within the project. It loads the crime data, labeled
    merged data, and merged data based on filename prefixes.

    Parameters:
    ----------
    data_dir: str
        The local directory where the data files are stored.

    Returns:
    -------
    Tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
        A tuple containing the three loaded dataframes.
    """
    print("\nLoading dataframes from local directory...")
    crime_df = load_pickle_by_prefix(data_dir, "crime_data")
    labeled_df = load_pickle_by_prefix(data_dir, "labeled_merged_data")
    merged_df = load_pickle_by_prefix(data_dir, "merged_data")

    return crime_df, labeled_df, merged_df


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
    crime_agg_view = MarkerCluster(name="Recent Crimes (Aggregated)").add_to(m)
    crime_det_view = folium.FeatureGroup(name="Recent Crimes (Detailed)").add_to(m)

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


def plot_hotspot_analysis(
    m: folium.Map, df_merged_crime: pd.DataFrame, philly_gdf: gpd.GeoDataFrame
) -> folium.Map:
    """
    Performs a hotspot analysis on the given data and adds it as a choropleth layer

    Parameters:
    -----------
    m: folium.Map
        The Folium map object to add the layers to
    df_merged_crime: pd.DataFrame
        DataFrame with merged crime data
    philly_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the Philadelphia boundary

    Returns:
    --------
    folium.Map
        The updated Folium map object with the hotspot layer
    """
    # Convert df_clustered to a GeoDataFrame and project it for distance calculations
    print("Converting dataframe")
    clustered_gdf = gpd.GeoDataFrame(
        df_merged_crime,
        geometry=gpd.points_from_xy(df_merged_crime.lon, df_merged_crime.lat),
        crs="EPSG:4326",
    ).to_crs("EPSG:2272")

    print("Making grid for hotspot analysis")
    # Create grid based on the entire Philadelphia boundary for full coverage
    philly_gdf_proj = philly_gdf.to_crs("EPSG:2272")
    xmin, ymin, xmax, ymax = philly_gdf_proj.total_bounds
    cell_size = 2500  # Grid cell size in feet; can be set higher/lower
    grid_cells = []
    x = xmin
    while x < xmax:
        y = ymin
        while y < ymax:
            grid_cells.append(
                Polygon(
                    [
                        (x, y),
                        (x + cell_size, y),
                        (x + cell_size, y + cell_size),
                        (x, y + cell_size),
                    ]
                )
            )
            y += cell_size
        x += cell_size
    hotspot_grid = gpd.GeoDataFrame(grid_cells, columns=["geometry"], crs="EPSG:2272")
    # Ensure hotspot_grid has 'n_crimes' column with 0 for empty cells
    hotspot_grid["n_crimes"].fillna(0, inplace=True)

    print("Aggregating crime in each grid cell")

    # Count points from df_clustered in each grid cell
    joined = gpd.sjoin(clustered_gdf, hotspot_grid, how="inner", predicate="within")
    crime_counts = joined.groupby("index_right").size().rename("n_crimes")
    hotspot_grid = hotspot_grid.merge(crime_counts, left_index=True, right_index=True, how="left")
    hotspot_grid["n_crimes"].fillna(0, inplace=True)
    # Create a separate grid for the analysis containing only cells with crime
    # analysis_grid = hotspot_grid[hotspot_grid["n_crimes"] > 0].copy()

    print("Doing calculations for hotspot analysis")
    # Calculate the Gi* statistic (z-scores) only on cells with data
    w = weights.Queen.from_dataframe(analysis_grid)

    print("Calculating G* local statistic")
    print("Variance of n_crimes:", analysis_grid["n_crimes"].var())
    g_local = esda.G_Local(analysis_grid["n_crimes"].values, w)

    print("Adding z-scores to the grid")

    analysis_grid["z_score"] = g_local.Zs

    print("Mergeing z-scores back into the grid")

    # Merge the z-scores back into the full grid for complete visualization
    hotspot_grid = hotspot_grid.merge(
        analysis_grid[["z_score"]], left_index=True, right_index=True, how="left"
    )
    # Fill cells with no z-score (0 crimes or islands) with a neutral value of 0
    hotspot_grid["z_score"].fillna(0, inplace=True)

    # Trim the grid to the Philadelphia boundary
    hotspot_grid_trimmed = gpd.overlay(hotspot_grid, philly_gdf_proj, how="intersection")

    # Reset the index to create a column that can be used as a key
    hotspot_grid_for_plot = hotspot_grid_trimmed.reset_index()

    # Select only the necessary columns to prevent JSON serialization errors
    hotspot_data_for_viz = hotspot_grid_for_plot[["index", "z_score", "geometry"]]

    # Create a Choropleth layer for the hotspots; The reversed red/blue colormap is used, so
    # hotspots (identified with darker red) are shown with a higher z score, and coldspots
    # (identified with blue) are shown with a negative/near zero z score.
    print("Adding hotspot layer to map")
    chlorpleth = folium.Choropleth(
        geo_data=hotspot_data_for_viz.to_crs("EPSG:4326"),
        name="Hotspots",
        data=hotspot_data_for_viz,
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
