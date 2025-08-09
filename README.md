# crime-nexus

## Background
This project analyzes and visualizes crime data, clusters, and hotspots in Philadelphia, Pennsylvania. Using a daily automated pipeline, the app processes crime, weather, and census data to provide insights into crime distribution. The visualizations are designed to help public health officials and city planners, or curious citizens, to identify areas of concern and understand the factors influencing crime patterns.

The data is sourced from various public APIs and data sources, including the City of Philadelphia's crime data, NOAA's weather data, and the U.S. Census Bureau's demographic and geographic data.

## Goal
The primary goal is to provide a comprehensive, interactive tool for visualizing crime data in Philadelphia. The app highlights crime hotspots and uses clustering to identify areas with similar characteristics based on crime, weather, and census data. This allows for more effective resource allocation and a deeper understanding of the relationships between environmental, demographic, and criminal factors.

The app also offers a data download feature, allowing users to access the raw data for further analysis. This is particularly useful for researchers, analysts, and city officials who need to work with the data outside of the application.

## Methodology
This project uses a daily pipeline to ensure the data is always up-to-date. The pipeline is split into three main parts:

### Data Retrieval and Merging
The pipeline automatically fetches raw crime, weather, and census data for Philadelphia. It then performs data cleaning, feature engineering, and merging to create a unified dataset. This process includes:

- Data Collecting: Collecting data automatically with relevant APIs, and then merging the results.
- Spatial Joins: Mapping crimes to specific census tracts.
- Data Standardization: Correcting data types and handling missing values.
- Feature Creation: Calculating new features like population density.

### Clustering/Hotspot Analysis
The merged dataset is fit with UMAP and DBSCAN to both reduce the dimensionality of the data, and then cluster the data This process involves:

- Hyperparameter Optimization: Using a TPE algorithm to find the best clustering parameters with the help of MLFlow (I connect with my personal DataBricks workspace, where the runs are actually executed and recorded).
- Clustering: Applying the best parameters to group crime data points into distinct clusters. I then subset to the clusters with the highest points of association (since I could end up displaying over 100 clusters depending on the daily data).
- Hotspot Analysis: Identifying areas with statistically significant crime activity.

### Mapping
The processed data is then used to generate a rich, interactive map using the Folium library. The map includes several layers that can be toggled on and off:
- Recent Crimes: Individual markers for the most recent crime incidents.
- Cluster Outlines: Boundaries for each crime cluster, color-coded for easy identification.
- Hotspots: Highlighted areas with high crime density.

## App Features
The app is hosted here: https://crime-nexus.streamlit.app/. I have a brief preview image of the app below:

<img width="2506" height="1013" alt="preview" src="https://github.com/user-attachments/assets/a48cc438-11ad-44ab-9787-5679fb92c4c9" />

The Streamlit app is divided into two tabs:

### Map Viewer
This tab displays the main interactive map with all the layers and legends. It provides a visual overview of crime in the city, allowing users to zoom, pan, and toggle layers for better visibility. The map is designed to be responsive and renders across the full width of the browser. I also include some summary statistics for each cluster.

### Data Downloader
This tab allows users to select a specific crime cluster and download the raw, processed data for that cluster as a CSV file. This is useful for in-depth analysis and is a key feature for enabling data-driven insights. The data is pre-processed to reverse one-hot-encoded columns, providing clean and human-readable data to undo the steps that were done during modeling. While I do include statistics for each cluster, this is on a very high level, and those interested are encouraged to take the data and do more in depth analysis. It would be impossible for me to automatically generate accurate, in depth analysis, since the data structure will change on a daily basis.

## Repo Structure
The repo is split up into folders based on each major component. In particular, the `/experimental` directory holds notebooks to walkthrough each major component, if one would like to get more details with my exact approach. Admittedly, there were a few changes I had to make throughout this projects, so the notebooks are slightly different than the actual scripts, but it should still be accurate enough for it's main purpose. The `.github/workflows` directory holds the .yml file for my GitHub Actions run. `/app` holds files for my Streamlit app. Finally, the `/src` directory holds the pipeline script, alongside a configuration and utilities file.

## Future Work
This project has several paths to enhance what it can show. However, given I am using free services (e.g., the free tier of GitHub Actions, Streamlit Community Cloud, alongside "free" versions of tokens), this leads to several limitations. For example, I could ingest data for several more years, but the NOAA has a token limit that prevents me from doing so. Not only that, but I could ingest more data into the clustering part of the pipeline, as well as the mapping. However, there is a hard time limit of 6h for the free version of GitHub Actions. Finally, Streamlit Community Cloud has resource limits, so all in all, I had to limit the amount of data to make the whole project run smoothly in each step. This is also why I have to do my hotspot analysis in the pipeline, since Streamlit will crash if I try to do the analysis in the app directory. *BUT*, this does mean that these are easy improvements if I were willing to pay for more resources.

Additionally, I did consider making a crime prediction model, which would be similar to the hotspot analysis, but also considering factors such as the number of bars in a grid cell, and so on. But, there are several ethical concerns with crime prediction. In particular, a model would just learn the pattern of where crime is reported, and not where crime is actually happening. Crime is likely underreported in some areas compared to others, and a prediction model would just learn that pattern, thereby enhancing existing discrimination. There are several resources to explore on this topic; I found this one for LA policing decisions to be a good starting point: https://vce.usc.edu/volume-5-issue-3/pitfalls-of-predictive-policing-an-ethical-analysis/.

To conclude this point, while this project is to some degree, also falling down this pitfall, I am only displaying the crimes that were recorded, not trying to predict crimes. Many of these points are with crime themselves, but the map should be left with some level of scrutiny.

The codebase is structured to be modular and extensible, serving as a solid foundation for these future enhancements, if I ever decide to go forward with these suggestions.

## (Rough) Data Dictionaries 

### Philadelphia Crime
| Column Name       | Data Type | Description                                                                                     |
|-------------------|-----------|-------------------------------------------------------------------------------------------------|
| dc_dist           | string    | The police district where the incident occurred.                                               |
| psa               | string    | The Police Service Area (PSA), a smaller geographic subdivision of a district.                 |
| dispatch_date     | string    | The date the call for service was dispatched (YYYY-MM-DD).                                      |
| dispatch_time     | string    | The time the call for service was dispatched (HH:MI:SS).                                        |
| hour              | integer   | The hour of the day the call was dispatched (0–23).                                             |
| text_general_code | string    | The text description for the type of incident (e.g., "Theft", "Assault").                       |
| location_block    | string    | The street address of the incident, anonymized to the block level.                              |
| point_y           | float     | The latitude coordinate for the incident location.                                              |
| point_x           | float     | The longitude coordinate for the incident location.                                             |

### Weather
| Column                | Data Type | Description                          |
| --------------------- | --------- | ------------------------------------ |
| date\_dt              | string    | The date of the weather observation. |
| avg\_wind\_speed\_mph | float     | Average Wind Speed (miles per hour). |
| precipitation\_inches | float     | Precipitation (inches).              |
| snowfall\_inches      | float     | Snowfall (inches).                   |
| snow\_depth\_inches   | float     | Snow Depth (inches).                 |
| max\_temp\_f          | float     | Maximum Temperature (°F).            |
| min\_temp\_f          | float     | Minimum Temperature (°F).            |

### Census 
| Column                  | Data Type | Description                                  |
| ----------------------- | --------- | -------------------------------------------- |
| tract\_fips             | string    | The FIPS code for the census tract.          |
| pop\_total              | float     | Total Population.                            |
| income\_median          | integer   | Median Household Income.                     |
| median\_age             | float     | Median Age.                                  |
| poverty\_rate           | float     | Percentage of people below the poverty line. |
| vacancy\_rate           | float     | Percentage of vacant housing units.          |
| renter\_occupancy\_rate | float     | Percentage of renter-occupied housing units. |
