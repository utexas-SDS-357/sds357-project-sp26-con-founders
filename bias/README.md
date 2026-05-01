# Bias Directory

Our analysis of potential bias in policing behavior after Vision Zero focuses on identifying changes in traffic-enforcement patterns and assessing whether these changes have disproportionately affected marginalized communities. This directory contains the data, analysis, and output relevant to this objective.

## Data Description

The following datasets are used:

### Primary Dataset

- [Stanford Open Policing Project](https://openpolicing.stanford.edu/data/): Standardized traffic stop data for cities across the United States are available from the Stanford Open Policing Project team. For this project, we use events recorded in San Francisco, where the unit of observation represents an instance in which a law enforcement officer conducted a traffic stop. The raw data contain 905,070 stops from December of 2006 to June of 2016. Relevant fields include the date, time, location, subject demographic data, characteristics, and outcome of a stop.

### Supplementary Datasets

- [Bay Area County Shapefiles](https://data.sfgov.org/Geographic-Locations-and-Boundaries/Bay-Area-County-Polygons/wamw-vt4s/about_data): These shapefiles from the San Francisco Open Data Portal contain the official geographic boundaries for all nine Bay Area counties. Since the scope of our analysis is focused on San Francisco County, the polygon coordinates for this region were used to filter extraneous observations from the policing and collision datasets.
- [San Francisco Equity Priority Communities](https://opendata-mtc.opendata.arcgis.com/datasets/MTC::equity-priority-communities-plan-bay-area-2040/about): Equity Priority Communities, also referred to as Communities of Concern, are census tracts of higher equity risk identified by the Metropolitan Transportation Commission. Factors used to classify tracts include vehicle access, racial composition, and the proportion of elderly and youth dependents, individuals with disabilities, and low-income households. Census tract shapefiles were obtained from the San Francisco Open Data Portal and contain polygon boundaries for all census tracts within San Francisco.
- [Astral Conditions](https://sffjunkie.github.io/astral/): Solar cycle data were collected using the `astral` package. These data are used to determine whether traffic stops occur during day, night, dawn, or dusk for the San Francisco area.

## Directory Structure

The directory structure is as follows:

| Directory | Subdirectory | File descriptions |
|-------------------|----------------------|-------------------------------|
| `bias` | `src`<br><br><br><br>`data`<br><br><br><br><br>`output` | **`src`**:<br>`bias_preprocessing.ipynb`: Preprocessing files used to clean traffic stop datasets<br>`bias_eda.ipynb`: Code used for exploratory data analysis of traffic stops<br>`bias_model.ipynb`: Modeling pipeline to produce output <br>**`data`**:<br>Raw data used to process collsion data and clean collision data obtained from `bias_preprocessing.ipynb` pipeline<br>**`output`**:<br>Relevant output from collision analysis pipeline (including `bias_eda.ipynb` and `bias_model.ipynb`) used in presentations/reports |
