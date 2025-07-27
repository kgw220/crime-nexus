"""
This script combines the experimental notebooks to create a formal daily pipeline that retrieves,
processes, and clusters crime, weather, and census data for the city of Philadelphia, PA. Refer to
the experimental notebooks for more details on how the code is setup!
"""

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
    PROBABILITY_THRESHOLD,
    SEARCH_SPACE,
    HQ_CLUSTER_LIMIT,
    RANDOM_SEED,
)


def main():
    """
    Main function to run the entire data pipeline from fetching to clustering.
    """
    # Define date range to collect data (using a 3-year rolling window up to yesterday)
    END_DATE = datetime.now() - timedelta(days=1)
    START_DATE = END_DATE - timedelta(days=3 * 365)
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
    # Calculate the years within your 3-year date range
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
    yesterday_crime = crime_df[crime_df["dispatch_date"] == END_DATE]

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
    output_path = os.path.join(data_dir, f"crime_data_{END_STR}.pkl")
    yesterday_crime.to_pickle(output_path)
    print(f"💾Yesterday's crime data saved to {output_path}💾")

    # Part 2: Clustering----------------------------------------------------------------------------
    print("📌📌📌📌📌📌📌📌📌📌Starting clustering part of pipeline!📌📌📌📌📌📌📌📌📌📌")

    # Prepare the MLFlow experiment and get the name
    exp_name = prepare_experiment(RUN_RETENTION_DAYS)

    # Create a unique ID for this specific pipeline execution; This will be used to identify which
    # runs were ran the day the script was ran
    pipeline_run_id = str(uuid.uuid4())
    print(f"🪪The unique pipeline ID for today is {pipeline_run_id}🪪")

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
        prob_threshold=PROBABILITY_THRESHOLD,
    )
    labeled_output_path = os.path.join(
        data_dir, f"labeled_merged_data_{START_STR}_to_{END_STR}.pkl"
    )
    df_final.to_pickle(labeled_output_path)
    print(f"----------Final clustered data saved to {output_path}----------")


if __name__ == "__main__":
    main()
