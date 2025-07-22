"""
This script combines the experimental notebooks to create a daily pipeline that
retrieves, processes, and clusters crime, weather, and census data for
the city of Philadelphia, PA. Refer to the experimental notebooks for more detail on how the code
is setup!
"""

import geopandas as gpd
import hdbscan
import mlflow
import numpy as np
import os
import pandas as pd
import random
import requests
import random
import time
import umap
import uuid

from datetime import datetime, timedelta
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

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
    PROBABILITY_THRESHOLD,
)


# Set the number of evaluations the TPE algorithm will perform
# NOTE: Ideally, this would be set higher, but each run takes several minutes to load, and GitHub
# Actions (on the free tier) has a cap of 6 hours for the entire script, so I set this number to be
# lower to ensure the whole daily pipeline will run in <6 hours. This can be increased if I did have
# a paid tier
NUM_EXPERIMENT_EVALS = 15


# Functions to fetch and clean data ----------------------------------------------------------------


def fetch_crime_data(table: str, start_date: str, end_date: str, carto_url: str) -> pd.DataFrame:
    """
    Downloads Philadelphia crime data from OpenDataPhilly in the specified date range.

    Parameters:
    -----------
    table: str
        The name of the table in the OpenDataPhilly database.
    start_date: str
        The start date for the data in 'YYYY-MM-DD' format.
    end_date: str
        The end date for the data in 'YYYY-MM-DD' format.
    CARTO_URL: str
        The base URL for the OpenDataPhilly API.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the crime data with relevant columns.
    """
    print(f"Extracting crime data from {start_date} to {end_date}...")
    # Simple SQL query to get the data needed from the OpenDataPhilly database
    # Subsetting to the columns we need, some are left out because I think they are irrelevant
    # Variables Selected:
    # - dc_dist:         The police district where the incident occurred.
    # - psa:             The Police Service Area, a smaller geographic subdivision of a district.
    # - dispatch_date:   The date the call for service was dispatched (YYYY-MM-DD).
    # - dispatch_time:   The time the call for service was dispatched (HH:MI:SS).
    # - hour:            The hour of the day the call was dispatched (0-23).
    # - text_general_code: The text description for the type of incident (e.g., "Theft," "Assault").
    # - location_block:  The street address of the incident, anonymized to the block level.
    # - lat:             The latitude coordinate for the incident location (aliased from point_y).
    # - lon:             The longitude coordinate for the incident location (aliased from point_x).
    query = f"""
        SELECT dc_dist, psa, dispatch_date, dispatch_time, hour, text_general_code, location_block,
        point_y as lat, point_x as lon
        FROM {table}
        WHERE dispatch_date >= '{start_date}' AND dispatch_date <= '{end_date}'
    """
    # Connect to the OpenDataPhilly API and get the data
    response = requests.get(carto_url, params={"q": query})
    response.raise_for_status()
    data = response.json().get("rows", [])
    crime_df = pd.DataFrame(data)

    return pd.DataFrame(data)


def clean_crime_data(crime_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the crime data to prepare for analysis. This involves steps like one-hot encoding,
    removing NAs, renaming columns, and creating new features.
    """
    print("Cleaning crime data...")

    # Drop rows with missing latitude, longitude, or police service area (psa)
    crime_df = crime_df.dropna(subset=["lat", "lon", "psa"])

    # Rename columns for clarity
    crime_rename_map = {
        "dc_dist": "police_district",
        "psa": "police_service_area",
        "text_general_code": "crime_type",
        "location_block": "address_block",
    }
    crime_df = crime_df.rename(columns=crime_rename_map)

    # Convert police district and service area to dummy variables since they are categorical
    crime_df["police_district"] = crime_df["police_district"].astype(str)
    crime_df["police_service_area"] = crime_df["police_service_area"].astype(str)
    district_dummies = pd.get_dummies(crime_df["police_district"], prefix="district")
    psa_dummies = pd.get_dummies(crime_df["police_service_area"], prefix="psa")
    crime_df = (
        crime_df.join(district_dummies)
        .join(psa_dummies)
        .drop(columns=["police_district", "police_service_area"])
    )

    # Create dummy columns for every unique crime type
    dummies = pd.get_dummies(crime_df["crime_type"], prefix="crime")
    crime_df = crime_df.join(dummies).drop(columns=["crime_type"])

    # Convert relevant columns to datetime
    crime_df["dispatch_date_dt"] = pd.to_datetime(crime_df["dispatch_date"]).dt.date
    crime_df["dispatch_date"] = pd.to_datetime(crime_df["dispatch_date"])
    crime_df["dispatch_time"] = pd.to_datetime(
        crime_df["dispatch_time"], format="%H:%M:%S", errors="coerce"
    )

    # Extract hour from dispatch_time; while the original dataframe had an hour column, it had
    # missing values
    crime_df["hour"] = crime_df["dispatch_time"].dt.hour

    # Define hour/month/day of week cyclic features
    crime_df["hour_sin"] = np.sin(2 * np.pi * crime_df["hour"] / 24.0)
    crime_df["hour_cos"] = np.cos(2 * np.pi * crime_df["hour"] / 24.0)
    crime_df["month"] = crime_df["dispatch_date"].dt.month
    crime_df["month_sin"] = np.sin(2 * np.pi * crime_df["month"] / 12.0)
    crime_df["month_cos"] = np.cos(2 * np.pi * crime_df["month"] / 12.0)
    crime_df["day_of_week"] = crime_df["dispatch_date"].dt.dayofweek
    crime_df["day_of_week_sin"] = np.sin(2 * np.pi * crime_df["day_of_week"] / 7.0)
    crime_df["day_of_week_cos"] = np.cos(2 * np.pi * crime_df["day_of_week"] / 7.0)
    crime_df = crime_df.drop(columns=["hour", "month", "day_of_week"])

    return crime_df


def fetch_weather_data(
    station_id: str,
    start_date: str,
    end_date: str,
    token: str,
    WEATHER_URL: str,
    max_retries: int = 5,
) -> pd.DataFrame:
    """
    Downloads daily weather data from NOAA for a single station.
    This function is designed to be called for a limited date range (e.g., one year), to avoid
    overwhelming the API with a single large request.
    """
    print(f"Fetching weather from {start_date} to {end_date}...")

    headers = {"token": token}
    all_results = []
    offset = 1

    # Get weather data from NOAA API
    # Variables Selected:
    # TMAX: Maximum Temperature
    # TMIN: Minimum Temperature
    # PRCP: Precipitation
    # AWND: Average Wind Speed
    # SNOW: Snowfall
    # SNWD: Snow Depth
    # NOTE: more can be added as per the documentation:
    # https://www1.ncdc.noaa.gov/pub/data/cdo/documentation/GHCND_documentation.pdf
    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "startdate": start_date,
            "enddate": end_date,
            "units": "standard",
            "limit": 1000,
            "offset": offset,
            "datatypeid": ["TMAX", "TMIN", "PRCP", "AWND", "SNOW", "SNWD"],
        }

        # Retry logic for temporary server errors
        response = None
        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(WEATHER_URL, headers=headers, params=params, timeout=20)
                if response.status_code in [500, 502, 503, 504]:
                    raise requests.exceptions.HTTPError(f"{response.status_code} Server Error")
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                wait_time = 2**retries
                print(f"Warning: Request failed ({e}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
                retries += 1
        if not response or response.status_code != 200:
            print(f" Error: Failed to download data after {max_retries} retries.")
            return pd.DataFrame()

        results = response.json().get("results", [])
        if not results:
            break

        # Process the results and append to the list
        all_results.extend(results)
        offset += 1000
        time.sleep(0.2)

    # Format the results into a DataFrame and do some basic cleaning to help later on
    df = pd.DataFrame(all_results)
    if df.empty:
        return pd.DataFrame()
    df["date_dt"] = pd.to_datetime(df["date"]).dt.date
    weather_df = df.pivot_table(index="date_dt", columns="datatype", values="value").reset_index()

    return weather_df


def clean_weather_data(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the weather data to prepare for analysis. This involves renaming columns for now.

    Parameters:
    ----------
    weather_df: pd.DataFrame
        The DataFrame containing the raw weather data.

    Returns:
    -------
    pd.DataFrame
        A cleaned DataFrame with relevant columns and features for analysis.
    """
    # Rename columns for clarity
    weather_rename_map = {
        "AWND": "avg_wind_speed_mph",
        "PRCP": "precipitation_inches",
        "SNOW": "snowfall_inches",
        "SNWD": "snow_depth_inches",
        "TMAX": "max_temp_f",
        "TMIN": "min_temp_f",
    }
    weather_df = weather_df.rename(columns=weather_rename_map)

    return weather_df


def fetch_census_data(
    state_fips: str, county_fips: str, token: str, CENSUS_API_URL: str
) -> pd.DataFrame:
    """
    Downloads and processes ACS 5-Year data for all tracts in a county, including calculation of key
    demographic rates.

    Parameters:
    -----------
    state_fips: str
        The FIPS code for the state (e.g., "42" for Pennsylvania).
    county_fips: str
        The FIPS code for the county (e.g., "101" for Philadelphia County).
    token: str
        Your Census API token for authentication.
    CENSUS_API_URL: str
        The base URL for the Census API.

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the census data with relevant columns and calculated rates.
    """
    print(f"Downloading census data for Philadelphia County (FIPS: {state_fips}{county_fips})...")

    # Variables Selected:
    # NAME: Geographic Area Name
    # B01003_001E: Total Population
    # B19013_001E: Median Household Income
    # B17001_002E: Poverty Count (total people below poverty line)
    # B01002_001E: Median Age
    # B25002_001E: Total Housing Units
    # B25002_003E: Vacant Housing Units
    # B25003_001E: Total Occupied Housing Units
    # B25003_003E: Renter-Occupied Housing Units
    # NOTE: could get more variables, as listed here:
    # https://api.census.gov/data/2023/acs/acs5/variables.html
    variables = (
        "NAME,B01003_001E,B19013_001E,B17001_002E,"
        "B01002_001E,B25002_001E,B25002_003E,"
        "B25003_001E,B25003_003E"
    )
    params = {
        "get": variables,
        "for": "tract:*",
        "in": f"state:{state_fips} county:{county_fips}",
        "key": token,
    }

    # Make the request to the Census API and parse the response
    response = requests.get(CENSUS_API_URL, params=params)
    response.raise_for_status()
    data = response.json()
    census_df = pd.DataFrame(data[1:], columns=data[0])

    return census_df


def clean_census_data(census_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the census data to prepare for analysis. This involves renaming columns for clarity and
    creating new columns.

    Parameters:
    ----------
    census_df: pd.DataFrame
        The DataFrame containing the raw census data.

    Returns:
    -------
    pd.DataFrame
        A cleaned DataFrame with relevant columns and features for analysis.
    """
    # Rename columns for clarity on what they represent
    census_df = census_df.rename(
        columns={
            "B01003_001E": "pop_total",
            "B19013_001E": "income_median",
            "B17001_002E": "poverty_total",
            "B01002_001E": "median_age",
            "B25002_001E": "total_housing_units",
            "B25002_003E": "vacant_housing_units",
            "B25003_001E": "total_occupied_units",
            "B25003_003E": "renter_occupied_units",
            "tract": "tract_fips_short",
        }
    )

    # Create the full FIPS code for joining
    census_df["tract_fips"] = (
        census_df["state"] + census_df["county"] + census_df["tract_fips_short"]
    )

    # Convert all relevant columns to numeric, handling errors
    numeric_cols = [
        "pop_total",
        "income_median",
        "poverty_total",
        "median_age",
        "total_housing_units",
        "vacant_housing_units",
        "total_occupied_units",
        "renter_occupied_units",
    ]
    for col in numeric_cols:
        census_df[col] = pd.to_numeric(census_df[col], errors="coerce")

    # Calculate new features which are useful for the clustering task
    # Replace zeros with NaN to prevent division errors
    census_df["pop_total"] = census_df["pop_total"].replace(0, np.nan)
    census_df["total_housing_units"] = census_df["total_housing_units"].replace(0, np.nan)
    census_df["total_occupied_units"] = census_df["total_occupied_units"].replace(0, np.nan)
    census_df["poverty_rate"] = census_df["poverty_total"] / census_df["pop_total"]
    census_df["vacancy_rate"] = census_df["vacant_housing_units"] / census_df["total_housing_units"]
    census_df["renter_occupancy_rate"] = (
        census_df["renter_occupied_units"] / census_df["total_occupied_units"]
    )

    # Replace any resulting NaN values (from division by zero) with 0
    rate_cols = ["poverty_rate", "vacancy_rate", "renter_occupancy_rate"]
    census_df[rate_cols] = census_df[rate_cols].fillna(0)

    print(f"Downloaded and processed census data for {len(census_df)} tracts.")

    # Return the final, clean set of columns
    final_columns = [
        "tract_fips",
        "pop_total",
        "income_median",
        "median_age",
        "poverty_rate",
        "vacancy_rate",
        "renter_occupancy_rate",
    ]

    return census_df[final_columns]


def get_census_tracts(CENSUS_SHAPE_URL: str) -> gpd.GeoDataFrame:
    """
    Fetches census tracts from ArcGIS API and returns them as a GeoDataFrame.

    Parameters:
    -----------
    CENSUS_SHAPE_URL: str
        The URL for the ArcGIS API endpoint that provides census tract geometries.

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the census tracts with their geometries.
    """
    print("Fetching census tracts from ArcGIS API...")

    # Make the API call to get the GeoJSON data
    response = requests.get(CENSUS_SHAPE_URL)

    # This will raise an error if the request fails
    response.raise_for_status()

    # Load the GeoJSON response directly into a GeoDataFrame
    gdf_tracts = gpd.GeoDataFrame.from_features(response.json()["features"])

    # Set the coordinate reference system, which is standard for web data
    gdf_tracts = gdf_tracts.set_crs("EPSG:4326")

    return gdf_tracts


def merge_crime_census(crime_df: pd.DataFrame, census_tracts: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Merges crime data with census data based on spatial join with census tracts.

    Parameters:
    ----------
    crime_df: pd.DataFrame
        The DataFrame containing the cleaned crime data.
    census_df: pd.DataFrame
        The DataFrame containing the cleaned census data.
    census_tracts: gpd.GeoDataFrame
        The GeoDataFrame containing the census tracts geometries.

    Returns:
    -------
    pd.DataFrame
        A merged DataFrame containing crime and census data.
    """
    # Convert the crime DataFrame to a GeoDataFrame
    print("Converting crime data to GeoDataFrame...")
    crime_gdf = gpd.GeoDataFrame(
        crime_df,
        geometry=gpd.points_from_xy(crime_df["lon"], crime_df["lat"]),
        crs="EPSG:4326",  # Ensure CRS matches the census tract data
    )

    # Perform the spatial join
    print("Performing spatial join to map crimes to census tracts...")
    # Note: The 'predicate' argument was renamed to 'op' in recent geopandas versions
    try:
        # For newer GeoPandas versions
        final_crime_data = gpd.sjoin(crime_gdf, census_tracts, how="inner", op="within")
    except TypeError:
        # For older GeoPandas versions
        final_crime_data = gpd.sjoin(crime_gdf, census_tracts, how="inner", predicate="within")

    print("\n Spatial join complete. Crime data now includes census tract info.")

    # Clean up the final DataFrame
    final_crime_data = final_crime_data.drop(
        columns=[
            "OBJECTID",
            "STATEFP10",
            "COUNTYFP10",
            "TRACTCE10",
            "NAME10",
            "NAMELSAD10",
            "MTFCC10",
            "FUNCSTAT10",
            "AWATER10",
            "INTPTLAT10",
            "INTPTLON10",
            "LOGRECNO",
            "index_right",
        ]
    )
    final_crime_map = {
        "GEOID10": "tract_id",
        "ALAND10": "land_area_sq_meters",
    }
    final_crime_data = final_crime_data.rename(columns=final_crime_map)

    return final_crime_data


def calculate_population_density(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates population density for each census tract in the merged DataFrame.

    Parameters:
    ----------
    merged_df: pd.DataFrame
        The DataFrame containing the merged crime, weather, and census data.

    Returns:
    -------
    pd.DataFrame
        The DataFrame with an additional column for population density.
    """
    print("Calculating population density...")

    # Convert land_area_sq_meters to square kilometers, replacing 0 with NaN to avoid errors
    # Divide by 1,000,000 to convert square meters to square kilometers
    merged_df["area_sq_km"] = merged_df["land_area_sq_meters"] / 1000000
    merged_df["area_sq_km"] = merged_df["area_sq_km"].replace(0, np.nan)

    # Calculate density and fill any missing results with 0
    merged_df["pop_density_sq_km"] = merged_df["pop_total"] / merged_df["area_sq_km"]
    merged_df["pop_density_sq_km"] = merged_df["pop_density_sq_km"].fillna(0)

    print("Population density calculated.")

    # Drop the land_area_sq_meters column as it's no longer needed
    merged_df = merged_df.drop(columns=["land_area_sq_meters"])

    return merged_df


# Define functions related to the clustering part of the pipeline ----------------------------------


def run_clustering_pipeline(df: pd.DataFrame, start_str: str, end_str: str) -> pd.DataFrame:
    """
    Runs the data scaling, dimensionality reduction, and clustering part of the pipeline.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the merged crime, weather, and census data.
    start_str: str
        The start date of the data collection period in 'YYYY-MM-DD' format.
    end_str: str
        The end date of the data collection period in 'YYYY-MM-DD' format.

    Returns:
    --------
    pd.DataFrame
        The DataFrame with cluster labels added, saved to a file.
    """
    print("Starting clustering part of the pipeline...")

    # The first significant step is to scale our data, as clustering is based on distance.
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # NOTE: I do not want to scale the sinusoidal features as they are already in the desired range
    cols_to_scale = [col for col in numeric_cols if "_sin" not in col and "_cos" not in col]
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Now, the data is ready for applying UMAP, a non-linear dimensionality reduction algorithm.
    # Some columns cannot be used here, so they are dropped.
    columns_to_drop = [
        "dispatch_date",
        "dispatch_time",
        "address_block",
        "dispatch_date_dt",
        "geometry",
        "tract_id",
    ]
    df_umap_ready = df.drop(columns=columns_to_drop)

    # Embed with UMAP. Sticking with static parameters for now.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    df_embedding = reducer.fit_transform(df_umap_ready).astype("float64")

    # Apply HDBSCAN to cluster the embedded data.
    clusterer = hdbscan.HDBSCAN(min_cluster_size=5000, min_samples=40, gen_min_span_tree=True)
    clusterer.fit(df_embedding)

    # Add the cluster labels back to the original dataframe.
    df["cluster_label"] = clusterer.labels_

    # Calculate and print the custom score (ideally used for MLFlow later if possible...)
    labels = clusterer.labels_
    dbcv_score = clusterer.relative_validity_
    noise_prop = np.sum(labels == -1) / len(labels)
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    custom_score = (dbcv_score - noise_prop) - (0.1 * np.log1p(num_clusters))
    print(f"Clustering complete with custom score: {custom_score:.4f}")

    # Calculate cluster quality and filter for high-quality clusters.
    # This identifies stable clusters based on HDBSCAN's probabilities.

    # Get the cluster labels and probabilities for each observation
    labels = clusterer.labels_
    probs = clusterer.probabilities_

    # Create a DataFrame with labels and probabilities
    prob_df = pd.DataFrame({"label": labels, "probability": probs})

    # Exclude noise
    prob_df = prob_df[(prob_df["label"] != -1)]

    # Compute mean probability for each cluster
    mean_probs = prob_df.groupby("label")["probability"].mean().sort_values(ascending=False)

    # Filter clusters with mean probability over the threshold
    high_quality_clusters = mean_probs[mean_probs > PROBABILITY_THRESHOLD]

    # Subset to only the high quality clusters
    df_high_quality = df[df["cluster_label"].isin(high_quality_clusters.index)].copy()
    print(f"\nFiltered data to {len(df_high_quality)} points belonging to high-quality clusters.")

    # Save the labeled data to a file in the data directory

    # Define the base directory (project root) relative to this script's location (src/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Define the full path to the data directory
    data_dir = os.path.join(base_dir, "data")
    # Create the data directory if it does not already exist
    os.makedirs(data_dir, exist_ok=True)
    # Construct the full output path for the clustered data file.
    output_path = os.path.join(data_dir, f"labeled_merged_data_{start_str}_to_{end_str}.pkl")

    df_high_quality.to_pickle(output_path)

    print(f"Clustering complete. Labeled data saved to {output_path}")


def run_tpe_search(df: pd.DataFrame, max_evals: int, pipeline_run_id: str):
    """
    Runs a hyperparameter search using Hyperopt's TPE algorithm, logging results to MLFlow.
    """
    print(f"Starting hyperparameter search for {max_evals} evaluations using TPE...")

    # Prepare data for clustering (done once)
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if "_sin" not in col and "_cos" not in col]
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    columns_to_drop = [
        "dispatch_date",
        "dispatch_time",
        "address_block",
        "dispatch_date_dt",
        "geometry",
        "tract_id",
    ]
    df_umap_ready = df_scaled.drop(columns=columns_to_drop)

    # Define the Hyperopt Search Space
    space = {
        "n_neighbors": hp.quniform("n_neighbors", 15, 150, 10),
        "min_dist": hp.uniform("min_dist", 0.0, 0.5),
        "n_components": hp.quniform("n_components", 5, 50, 1),
        "min_cluster_size": hp.quniform("min_cluster_size", 4000, 6000, 100),
        "min_samples": hp.quniform("min_samples", 10, 200, 5),
    }

    # Define the Objective Function for Hyperopt
    def objective(params):
        """
        Objective function for Hyperopt to minimize. It runs the clustering pipeline
        and returns the negative of the custom score, logging everything to MLFlow.
        """
        params["n_neighbors"] = int(params["n_neighbors"])
        params["n_components"] = int(params["n_components"])
        params["min_cluster_size"] = int(params["min_cluster_size"])
        params["min_samples"] = int(params["min_samples"])

        with mlflow.start_run():

            # Tag the run with the unique pipeline execution ID
            mlflow.set_tag("pipeline_run_id", pipeline_run_id)

            print(f"\n--- Evaluating parameters: {params} ---")
            mlflow.log_params(params)

            umap_params = {
                "n_neighbors": params["n_neighbors"],
                "min_dist": params["min_dist"],
                "n_components": params["n_components"],
                "random_state": 42,
            }
            hdbscan_params = {
                "min_cluster_size": params["min_cluster_size"],
                "min_samples": params["min_samples"],
                "gen_min_span_tree": True,
            }

            reducer = umap.UMAP(**umap_params)
            df_embedding = reducer.fit_transform(df_umap_ready).astype("float64")
            clusterer = hdbscan.HDBSCAN(**hdbscan_params)
            clusterer.fit(df_embedding)

            labels = clusterer.labels_
            dbcv_score = clusterer.relative_validity_
            noise_prop = np.sum(labels == -1) / len(labels) if len(labels) > 0 else 1.0
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            custom_score = (dbcv_score - noise_prop) - (0.1 * np.log1p(num_clusters))

            print(f"-> Trial complete. Custom Score: {custom_score:.4f}")
            mlflow.log_metric("custom_score", custom_score)
            mlflow.log_metric("dbcv_score", dbcv_score)
            mlflow.log_metric("noise_proportion", noise_prop)
            mlflow.log_metric("num_clusters", num_clusters)

            return {"loss": -custom_score, "status": STATUS_OK}

    # Run the TPE Optimization
    trials = Trials()
    rstate = np.random.default_rng(42)

    fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=max_evals,
        trials=trials,
        rstate=rstate,
    )


def get_best_run_parameters(experiment_id: str, pipeline_run_id: str) -> dict:
    """
    Queries MLFlow to find the best run from the experiment and returns its parameters.
    """
    print("\nFinding best run from today's runs...")

    # Filter runs by the unique pipeline_run_id tag
    filter_string = f"tags.pipeline_run_id = '{pipeline_run_id}'"
    best_run = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["metrics.custom_score DESC"],
        max_results=1,
    )

    if best_run.empty:
        raise Exception("No runs found in the experiment. Cannot determine best parameters.")

    best_params = best_run.filter(regex="params\..*").to_dict("records")[0]
    best_params = {key.replace("params.", ""): value for key, value in best_params.items()}

    print(f"Best parameters found: {best_params}")
    return best_params


def run_final_pipeline(df: pd.DataFrame, best_params: dict, start_str: str, end_str: str):
    """
    Runs the clustering pipeline with the best hyperparameters and saves the final output.
    """
    print("\nRunning final pipeline with best hyperparameters...")

    scaler = StandardScaler()
    scaler.set_output(transform="pandas")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cols_to_scale = [col for col in numeric_cols if "_sin" not in col and "_cos" not in col]
    df_scaled = df.copy()
    df_scaled[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    columns_to_drop = [
        "dispatch_date",
        "dispatch_time",
        "address_block",
        "dispatch_date_dt",
        "geometry",
        "tract_id",
    ]
    df_umap_ready = df_scaled.drop(columns=columns_to_drop)

    umap_params = {
        "n_neighbors": int(float(best_params["n_neighbors"])),
        "min_dist": float(best_params["min_dist"]),
        "n_components": int(float(best_params["n_components"])),
        "random_state": 42,
    }
    hdbscan_params = {
        "min_cluster_size": int(float(best_params["min_cluster_size"])),
        "min_samples": int(float(best_params["min_samples"])),
        "prediction_data": True,
    }

    reducer = umap.UMAP(**umap_params)
    df_embedding = reducer.fit_transform(df_umap_ready).astype("float64")
    clusterer = hdbscan.HDBSCAN(**hdbscan_params)
    clusterer.fit(df_embedding)

    df["cluster_label"] = clusterer.labels_

    labels = clusterer.labels_
    probs = clusterer.probabilities_
    prob_df = pd.DataFrame({"label": labels, "probability": probs})
    prob_df = prob_df[prob_df["label"] != -1]

    if not prob_df.empty:
        mean_probs = prob_df.groupby("label")["probability"].mean()
        high_quality_clusters = mean_probs[mean_probs > PROBABILITY_THRESHOLD]
        df_high_quality = df[df["cluster_label"].isin(high_quality_clusters.index)].copy()

        print(f"\nFiltered final data to {len(df_high_quality)} points in high-quality clusters.")

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        output_path = os.path.join(data_dir, f"labeled_merged_data_{start_str}_to_{end_str}.pkl")

        df_high_quality.to_pickle(output_path)
        print(f"Final clustered data saved to {output_path}")
    else:
        print("No high-quality clusters found in the final run.")


def main():
    """
    Main function to run the entire data pipeline from fetching to clustering.
    """
    # Define date range to collect data (using a 3-year rolling window up to yesterday)
    END_DATE = datetime.now() - timedelta(days=1)
    START_DATE = END_DATE - timedelta(days=3 * 365)
    START_STR = START_DATE.strftime("%Y-%m-%d")
    END_STR = END_DATE.strftime("%Y-%m-%d")
    END_STR_2_DAYS_AGO = (END_DATE - timedelta(2)).strftime("%Y-%m-%d")

    # Part 1: Data Retrieval and Merging -----------------------------------------------------------
    print("Starting data retrieval and merging...")
    # Fetch and clean the data from the various sources
    crime_df = fetch_crime_data(CRIME_TABLE_NAME, START_STR, END_STR, CARTO_URL)
    crime_df = clean_crime_data(crime_df)

    # Initialize an empty list to hold the weather DataFrames for each year
    weather_frames = []
    # Calculate the years within your 3-year date range
    start_year = START_DATE.year
    end_year = END_DATE.year

    # Loop through each year in the range and fetch the weather data
    for year in range(start_year, end_year + 1):
        # Define the start and end for this specific year's API call
        year_start_str = f"{year}-01-01"
        year_end_str = f"{year}-12-31"
        print(f"-> Downloading weather for year {year}...")
        df_year = fetch_weather_data(
            station_id=WEATHER_STATION_ID,
            start_date=year_start_str,
            end_date=year_end_str,
            token=NOAA_TOKEN,
            WEATHER_URL=WEATHER_URL,
        )

        if not df_year.empty:
            weather_frames.append(df_year)

    # If I successfully downloaded data for any year, combine them into a single DataFrame
    if weather_frames:
        # Combine the data from all successful yearly calls
        weather_df = pd.concat(weather_frames, ignore_index=True)
        # Filter the combined data to the precise 3-year rolling window
        weather_df["date_dt_obj"] = pd.to_datetime(weather_df["date_dt"])
        weather_df = weather_df[
            (weather_df["date_dt_obj"] >= pd.to_datetime(START_STR))
            & (weather_df["date_dt_obj"] <= pd.to_datetime(END_STR))
        ].drop(columns=["date_dt_obj"])
        print(f"\n Successfully combined weather data for {len(weather_df)} days.")
    else:
        print("\n No weather data could be downloaded.")
        weather_df = pd.DataFrame()

    weather_df = clean_weather_data(weather_df)

    census_df = fetch_census_data(PA_STATE_FIPS, PHILLY_COUNTY_FIPS, CENSUS_TOKEN, CENSUS_API_URL)
    census_df = clean_census_data(census_df)

    # To properly map census data, I need to determine which tract each crime is in.
    # This uses a geojson file that outlines each census tract in Philadelphia.
    gdf_tracts = get_census_tracts(CENSUS_SHAPE_URL)

    # Perform spatial join to map crimes to census tracts
    final_crime_data = merge_crime_census(crime_df, gdf_tracts)

    # Merge all the data together
    print("Merging crime, weather, and census data...")
    merged_df = pd.merge(
        final_crime_data, weather_df, left_on="dispatch_date_dt", right_on="date_dt", how="left"
    )
    merged_df = pd.merge(
        merged_df, census_df, left_on="tract_id", right_on="tract_fips", how="left"
    )
    merged_df = merged_df.drop(columns=["date_dt", "tract_fips"], errors="ignore")

    # As one final step, add the population density column
    merged_df = calculate_population_density(merged_df)

    # Drop any remaining missing values; these are rare cases not worth imputing
    final_merged_df = merged_df.dropna()
    assert final_merged_df.isna().sum().sum() == 0, "Error: Missing values were found!"

    # Print some information about the final merged DataFrame
    print(merged_df.head())
    print(merged_df.info())

    # Save the data from 2 days ago from the end date
    two_days_ago_date = (END_DATE - timedelta(days=2)).date()
    data_2_days_ago = final_merged_df[
        final_merged_df["dispatch_date_dt"] >= two_days_ago_date
    ].copy()

    print(data_2_days_ago.head())
    print(data_2_days_ago.info())

    # Define the base directory (project root) and the data directory path
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")

    # Create the directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Construct the full output path and save the file
    output_path = os.path.join(data_dir, f"merged_data_{END_STR_2_DAYS_AGO}.pkl")
    data_2_days_ago.to_pickle(output_path)
    print(f"Daily merged data saved to {output_path}")

    # Part 2: Clustering- --------------------------------------------------------------------------
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not mlflow_tracking_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable not set!")

    mlflow.set_tracking_uri(mlflow_tracking_uri)
    experiment_name = "/Users/kevingrahamwu@gmail.com/Daily_Crime_Clustering"
    mlflow.set_experiment(experiment_name)

    # reate a unique ID for this specific pipeline execution
    pipeline_run_id = str(uuid.uuid4())

    # Run the TPE hyperparameter search
    run_tpe_search(final_merged_df, NUM_EXPERIMENT_EVALS, pipeline_run_id)

    # Get the best parameters from the experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    best_params = get_best_run_parameters(experiment.experiment_id, pipeline_run_id)

    # Run the final pipeline with the best parameters
    run_final_pipeline(final_merged_df, best_params, START_STR, END_STR)
    # run_clustering_pipeline(final_merged_df, START_STR, END_STR)


if __name__ == "__main__":
    main()
