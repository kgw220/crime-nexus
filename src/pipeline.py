"""
This script combines the experimental notebooks to create a formal daily pipeline that retrieves,
processes, and clusters crime, weather, and census data for the city of Philadelphia, PA. Refer to
the experimental notebooks for more details on how the code is setup!
"""

import dropbox
import folium
import geopandas as gpd
import io
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import os
import pandas as pd
import uuid

from datetime import datetime, timedelta

from data_utils import (
    calculate_population_density,
    clean_census_data,
    clean_crime_data,
    clean_weather_data,
    delete_all_files,
    fetch_census_data,
    fetch_crime_data,
    fetch_weather_data,
    find_hotspots,
    get_best_run_parameters,
    get_census_tracts,
    list_files,
    merge_crime_census,
    prepare_experiment,
    run_final_pipeline,
    run_tpe_search,
    upload_file,
)

# Import configuration constants and API tokens, as defined in config.py in the same directory
from config import (
    BOUNDARY,
    CARTO_URL,
    CENSUS_API_URL,
    CENSUS_SHAPE_URL,
    CENSUS_TOKEN,
    CRIME_TABLE_NAME,
    DROPBOX_APP_KEY,
    DROPBOX_APP_SECRET,
    DROPBOX_REFRESH_TOKEN,
    FOLDER_PATH,
    HQ_CLUSTER_LIMIT,
    NOAA_TOKEN,
    NUM_EXPERIMENT_EVALS,
    PA_STATE_FIPS,
    PHILLY_COUNTY_FIPS,
    RANDOM_SEED,
    RUN_RETENTION_DAYS,
    SEARCH_SPACE,
    WEATHER_STATION_ID,
    WEATHER_URL,
    YEARS_TO_COLLECT,
)


def main():
    """
    Main function to run the entire data pipeline from fetching to clustering.
    """
    # Define date range to collect data (using a 1-year rolling window up to yesterday)
    END_DATE = (datetime.now() - timedelta(days=1)).date()
    START_DATE = END_DATE - timedelta(days=365 * YEARS_TO_COLLECT)
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

    # Save just crime data from the latest date
    # NOTE: I use the "max" date instead of the END_DATE string, because the script runs on UTC
    # time, and if I run it in the evening in EST/EDT time, UTC will be a day ahead, leading
    # to no data for yesterday_crime
    yesterday_date = crime_df["dispatch_date"].max()
    yesterday_crime = crime_df[crime_df["dispatch_date"]== yesterday_date]

    print(f"<<<<< 📝Yesterday's recorded crimes:📝 >>>>>")
    print(f"There were {len(yesterday_crime)} recorded crimes!")
    print(yesterday_crime.head())
    print(yesterday_crime.info())

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

    # Perform initial hotspot analysis here, since it is too computationally expensive on
    # streamlit
    philly_gdf = gpd.read_file(BOUNDARY)
    hotspot_grid = find_hotspots(final_merged_df, philly_gdf)

    # Part 3: Saving the data ----------------------------------------------------------------------

    print("📌📌📌📌📌📌📌📌📌📌Saving data!📌📌📌📌📌📌📌📌📌📌")

    # Initialize the Dropbox client with OAuth2 credentials and refresh token.
    # The SDK will auto-refresh access tokens when they expire.
    dbx = dropbox.Dropbox(
        oauth2_refresh_token=DROPBOX_REFRESH_TOKEN,
        app_key=DROPBOX_APP_KEY,
        app_secret=DROPBOX_APP_SECRET,
    )

    # Ensure folder exists
    try:
        dbx.files_get_metadata(FOLDER_PATH)
    except dropbox.exceptions.ApiError as e:
        if isinstance(e.error, dropbox.files.GetMetadataError) or "not_found" in str(e).lower():
            dbx.files_create_folder_v2(FOLDER_PATH)
            print(f"📁 Created Dropbox folder: {FOLDER_PATH}")
        else:
            raise

    # Print all the pre-existing files in the Dropbox folder
    print(f"----------Files in Dropbox folder {FOLDER_PATH} before upload:----------")
    list_files(dbx, FOLDER_PATH)
    print(f"----------Deleting in Dropbox folder {FOLDER_PATH} before uploading:----------")
    delete_all_files(dbx, FOLDER_PATH)

    # Convert final data results as bytes, and store to Dropbox
    buffer = io.BytesIO()

    yesterday_crime.to_pickle(buffer)
    buffer.seek(0)
    upload_file(
        dropbox_client=dbx,
        file_bytes=buffer.read(),
        folder_path=FOLDER_PATH,
        file_name=f"crime_data_{END_STR}.pkl",
    )
    # Reset buffer for the next upload
    buffer.seek(0)
    buffer.truncate(0)

    final_merged_df.to_pickle(buffer)
    buffer.seek(0)
    upload_file(
        dropbox_client=dbx,
        file_bytes=buffer.read(),
        folder_path=FOLDER_PATH,
        file_name=f"merged_data_{START_STR}_to_{END_STR}.pkl",
    )
    # Reset buffer for the next upload
    buffer.seek(0)
    buffer.truncate(0)

    df_final.to_pickle(buffer)
    buffer.seek(0)
    upload_file(
        dropbox_client=dbx,
        file_bytes=buffer.read(),
        folder_path=FOLDER_PATH,
        file_name=f"labeled_merged_data_{START_STR}_to_{END_STR}.pkl",
    )
    # Reset buffer for the next upload
    buffer.seek(0)
    buffer.truncate(0)

    hotspot_grid.to_pickle(buffer)
    buffer.seek(0)
    upload_file(
        dropbox_client=dbx,
        file_bytes=buffer.read(),
        folder_path=FOLDER_PATH,
        file_name=f"hotspot_grid_{START_STR}_to_{END_STR}.pkl",
    )
    print(f"----------Logged all data to Dropbox App----------")

    print(f"----------Files in Dropbox folder {FOLDER_PATH} after upload:----------")
    list_files(dbx, FOLDER_PATH)


if __name__ == "__main__":
    main()
