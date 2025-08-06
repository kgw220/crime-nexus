"""
This script combines the experimental notebooks to create a formal daily pipeline that retrieves,
processes, and clusters crime, weather, and census data for the city of Philadelphia, PA. Refer to
the experimental notebooks for more details on how the code is setup!
"""

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import os
import pandas as pd
import uuid

from datetime import datetime, timedelta


from data_utils import (
    fetch_crime_data,
    clean_crime_data,
    fetch_weather_data,
    clean_weather_data,
    fetch_census_data,
    clean_census_data,
    get_census_tracts,
    merge_crime_census,
    calculate_population_density,
    prepare_experiment,
    cleanup_old_runs,
    run_tpe_search,
    get_best_run_parameters,
    run_final_pipeline,
    plot_recent_crimes,
    plot_cluster_outlines,
    plot_hotspot_analysis,
)

# Import configuration constants and API tokens, as defined in config.py in the same directory
from config import (
    NOAA_TOKEN,
    CENSUS_TOKEN,
    CARTO_URL,
    CRIME_TABLE_NAME,
    WEATHER_STATION_ID,
    WEATHER_URL,
    CENSUS_API_URL,
    PHILLY_COUNTY_FIPS,
    PA_STATE_FIPS,
    CENSUS_SHAPE_URL,
    RUN_RETENTION_DAYS,
    NUM_EXPERIMENT_EVALS,
    SEARCH_SPACE,
    HQ_CLUSTER_LIMIT,
    RANDOM_SEED,
    BOUNDARY,
    DISTANCE_THRESHOLD,
)


def main():
    """
    Main function to run the entire data pipeline from fetching to clustering.
    """
    # Define date range to collect data (using a 3-year rolling window up to yesterday)
    END_DATE = (datetime.now() - timedelta(days=1)).date()
    START_DATE = END_DATE - timedelta(days=365)
    START_STR = START_DATE.strftime("%Y-%m-%d")
    END_STR = END_DATE.strftime("%Y-%m-%d")

    # Part 1: Data Retrieval and Merging -----------------------------------------------------------

    print("📌📌📌📌📌📌📌📌📌📌Starting data retrieval and merging!📌📌📌📌📌📌📌📌📌📌")
    # Fetch and clean the data from the various sources
    crime_df = fetch_crime_data(
        table=CRIME_TABLE_NAME,
        start_date=START_STR,
        end_date=END_STR,
        carto_url=CARTO_URL,
        max_retries=5,
    )
    assert (
        not crime_df.empty
    ), "🚨CRITICAL: Crime data fetch returned an empty DataFrame. Halting execution.🚨"

    crime_df = clean_crime_data(crime_df)

    # Initialize an empty list to hold the weather DataFrames for each year
    weather_frames = []
    # Calculate the years within the 1-year date range
    start_year = START_DATE.year
    end_year = END_DATE.year

    # Loop through each year in the range and fetch the weather data
    for year in range(start_year, end_year + 1):
        # Define the start and end for this specific year's API call
        year_start_str = f"{year}-01-01"
        year_end_str = f"{year}-12-31"
        print(f"\n<<<<< Downloading weather for year {year} >>>>>")
        df_year = fetch_weather_data(
            station_id=WEATHER_STATION_ID,
            start_date=year_start_str,
            end_date=year_end_str,
            token=NOAA_TOKEN,
            weather_url=WEATHER_URL,
            max_retries=5,
        )

        if not df_year.empty:
            weather_frames.append(df_year)

    # Combine the weather data from each year, into a single DataFrame
    if weather_frames:
        # Combine the data from all successful yearly calls
        weather_df = pd.concat(weather_frames, ignore_index=True)
        # Filter the combined data to the precise 3-year rolling window
        weather_df["date_dt_obj"] = pd.to_datetime(weather_df["date_dt"])
        weather_df = weather_df[
            (weather_df["date_dt_obj"] >= pd.to_datetime(START_STR))
            & (weather_df["date_dt_obj"] <= pd.to_datetime(END_STR))
        ].drop(columns=["date_dt_obj"])
        print(f"\n<<<<< Successfully combined weather data for {len(weather_df)} days. >>>>>")
    else:
        print("\n <<<<< No weather data could be downloaded. >>>>>")
        weather_df = pd.DataFrame()

    assert (
        not weather_df.empty
    ), "🚨CRITICAL: Weather data fetch returned an empty DataFrame. Halting execution.🚨"

    weather_df = clean_weather_data(weather_df)

    census_df = fetch_census_data(
        state_fips=PA_STATE_FIPS,
        county_fips=PHILLY_COUNTY_FIPS,
        token=CENSUS_TOKEN,
        census_api_url=CENSUS_API_URL,
        max_retries=5,
    )
    assert (
        not census_df.empty
    ), "CRITICAL: Census data fetch returned an empty DataFrame. Halting execution.🚨"

    census_df = clean_census_data(census_df)

    # To properly map census data, I need to determine which tract each crime is in.
    # This uses a geojson file that outlines each census tract in Philadelphia.
    gdf_tracts = get_census_tracts(CENSUS_SHAPE_URL, max_retries=5)
    assert (
        not gdf_tracts.empty
    ), "🚨Census tract data fetch returned an empty DataFrame. Halting execution.🚨"

    # Perform spatial join to map crimes to census tracts
    final_crime_data = merge_crime_census(crime_df, gdf_tracts)

    # Merge all the data together
    print("<<<<< Merging crime, weather, and census data >>>>>")
    merged_df = pd.merge(
        final_crime_data,
        weather_df,
        left_on="dispatch_date_dt",
        right_on="date_dt",
        how="left",
    )
    merged_df = pd.merge(
        merged_df, census_df, left_on="tract_id", right_on="tract_fips", how="left"
    )
    merged_df = merged_df.drop(columns=["date_dt", "tract_fips"], errors="ignore")

    # As one final step, add the population density column
    merged_df = calculate_population_density(merged_df)

    # Drop any remaining missing values; these are rare cases not worth imputing
    final_merged_df = merged_df.dropna()

    # Print some information about the final merged DataFrame and check for nulls
    min_date = final_merged_df["dispatch_date"].min()
    max_date = final_merged_df["dispatch_date"].max()
    print(f"\n<<<<< 📝Final merged DataFrame contains data from {min_date} to {max_date}📝 >>>>>")
    print(final_merged_df.info())
    assert final_merged_df.isnull().sum().sum() == 0, "🚨DataFrame contains null values.🚨"

    # Save just crime data from the latest date (END_DATE);
    # NOTE: Ideally, I would use all the merged data, but there seems to be a ~3 day delay with the
    # NOAA weather API (I receieve data from 7/24/25 on 7/27/25). This is not too important anyways,
    # since this is used as a second layer over the final clusters.
    yesterday_crime = crime_df[crime_df["dispatch_date"].dt.date == END_DATE]

    print(f"<<<<< 📝Yesterday's recorded crimes:📝 >>>>>")
    print(f"There were {len(yesterday_crime)} recorded crimes!")
    print(yesterday_crime.head())
    print(yesterday_crime.info())

    # Define the base directory (project root) and the data directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Construct the full output path and save the file
    recent_crime_output_path = os.path.join(data_dir, f"crime_data_{END_STR}.pkl")
    yesterday_crime.to_pickle(recent_crime_output_path)
    print(f"💾Yesterday's crime data saved to {recent_crime_output_path}💾")

    # Save the merged data from entire 3-year rolling window (for hotspot analysis)
    merged_output_path = os.path.join(data_dir, f"merged_data_{START_STR}_to_{END_STR}.pkl")
    final_merged_df.to_pickle(merged_output_path)
    print(f"💾Merged crime data saved to {merged_output_path}💾")

    # Part 2: Clustering----------------------------------------------------------------------------

    print("📌📌📌📌📌📌📌📌📌📌Starting clustering part of pipeline!📌📌📌📌📌📌📌📌📌📌")

    # Prepare the MLFlow experiment and get the name
    exp_name = prepare_experiment(RUN_RETENTION_DAYS)

    # Create a unique ID for this specific pipeline execution; This will be used to identify which
    # runs were ran the day the script was ran
    pipeline_run_id = str(uuid.uuid4())
    print(f"The unique pipeline ID for today is {pipeline_run_id}")

    # Run the TPE hyperparameter search
    run_tpe_search(
        df=final_merged_df,
        max_evals=NUM_EXPERIMENT_EVALS,
        seed=RANDOM_SEED,
        pipeline_run_id=pipeline_run_id,
        search_space=SEARCH_SPACE,
    )

    # Get the best parameters from the experiment
    experiment = mlflow.get_experiment_by_name(exp_name)
    best_params = get_best_run_parameters(experiment.experiment_id, pipeline_run_id)

    # Run the final pipeline with the best parameters, and save the file
    df_final = run_final_pipeline(
        df=final_merged_df,
        best_params=best_params,
        seed=RANDOM_SEED,
        max_clusters=HQ_CLUSTER_LIMIT,
    )
    labeled_merged_output_path = os.path.join(
        data_dir, f"labeled_merged_data_{START_STR}_to_{END_STR}.pkl"
    )
    df_final.to_pickle(labeled_merged_output_path)
    print(f"----------Final clustered data saved to {labeled_merged_output_path}----------")

    # Part 3: Mapping ------------------------------------------------------------------------------

    print("📌📌📌📌📌📌📌📌📌📌Starting mapping part of pipeline!📌📌📌📌📌📌📌📌📌📌")

    print("\n<<<<< 🗺️Creating Folium map of crime data🗺️ >>>>>")
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
    unique_clusters = sorted(df_final["cluster_label"].unique())
    cmap_clusters = plt.get_cmap("jet", len(unique_clusters))
    color_map_clusters = {
        cluster: matplotlib.colors.rgb2hex(cmap_clusters(i))
        for i, cluster in enumerate(unique_clusters)
    }

    # Create a mapping for alphabetical cluster labels
    cluster_nums = sorted(unique_clusters)
    alpha_labels = {num: chr(65 + i) for i, num in enumerate(cluster_nums)}
    # Apply the new labels to the dataframe
    df_final["cluster_alpha_label"] = df_final["cluster_label"].map(alpha_labels)

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

    print("\n<<<<< 🗺️Adding layers to map🗺️ >>>>>")
    # Add recent crime, cluster outline, and hotspot layers to the map
    m_crime = plot_recent_crimes(m_crime, crime_df, color_map_types)
    m_crime = plot_cluster_outlines(
        m_crime, df_final, color_map_clusters, alpha_labels, DISTANCE_THRESHOLD
    )
    m_crime = plot_hotspot_analysis(m_crime, final_merged_df, philly_gdf)

    # Add a control for controlling the layers
    folium.LayerControl().add_to(m_crime)

    print("\n<<<<< 🗺️Setting up legend HTML🗺️ >>>>>")
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
        legend_cluster_items += f'&nbsp; <i class="fa fa-{icon_shape}" style="color:{color}"></i> &nbsp; {label_text}<br>'
    legend_cluster_html_end = "</div></div>"
    full_legend_cluster_html = (
        legend_cluster_html_start + legend_cluster_items + legend_cluster_html_end
    )
    m_crime.get_root().html.add_child(folium.Element(full_legend_cluster_html))

    print("\n<<<<< 🗺️Saving map HTML🗺️ >>>>>")
    # Save final map as html file
    map_output_path = os.path.join(data_dir, f"map.html")
    m_crime.save(map_output_path)
    print(f"----------Final map saved to {map_output_path}----------")


if __name__ == "__main__":
    main()
