[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

# Analyzing Vision Zero Outcomes in Collisions, Community, and Policing

## Project Overview

In 2014, San Francisco adopted Vision Zero, a 10-year plan focused on eliminating fatal and severe crashes. 
While this initiative was designed to address collisions, enforcement of its subsequent policies may have also affected daily traffic activity and oversight. 
Currently there is limited analysis and documentation available to the public regarding the reform’s impact on traffic activity. 
This project will examine policy effectiveness by comparing pre- and post implementation outcomes associated with Vision Zero in San Francisco. 
The scope of this project defines ”effectiveness” through two key metrics: 
1. impact on traffic outcomes
2. changes in policing behavior across demographic groups.

The impact of Vision Zero San Francisco on traffic outcomes is centered around examining whether the changes reduced traffic-related fatalities and injuries. 
Our accompanying analysis of potential bias in policing behavior after Vision Zero focuses on identifying changes in traffic-enforcement patterns and assessing whether these changes have disproportionately affected marginalized communities. 
A quantitative assessment of these objectives provides insight into whether Vision Zero, given the ethical implications identified in this study, is suitable for broader adoption across U.S. cities.

## Data Description

### Primary Datasets

- [Stanford Open Policing Project](https://openpolicing.stanford.edu/data/): Standardized traffic stop data for cities across the United States are available from the Stanford Open Policing Project team.
For this project, we use events recorded in San Francisco, where the unit of observation represents an instance in which a law enforcement officer conducted a traffic stop.
The raw data contain 905,070 stops from December of 2006 to June of 2016.
Relevant fields include the date, time, location, subject demographic data, characteristics, and outcome of a stop.

- [California Statewide Integrated Traffic Records System](https://www.kaggle.com/datasets/alexgude/california-traffic-collision-data-from-switrs): The SWITRS data provide standardized records for California traffic collision outcomes.
For the scope of this project, only events occurring in major California cities, including San Francisco, were used.
The raw data contain over one million geocoded collision records from January 2010 to December 2016.
Relevant fields include crash severity, environmental conditions, contributing factors, and geographic identifiers.

### Supplementary Datasets

- [Bay Area County Shapefiles](https://data.sfgov.org/Geographic-Locations-and-Boundaries/Bay-Area-County-Polygons/wamw-vt4s/about_data): These shapefiles from the San Francisco Open Data Portal contain the official geographic boundaries for all nine Bay Area counties.
Since the scope of our analysis is focused on San Francisco County, the polygon coordinates for this region were used to filter extraneous observations from the policing and collision datasets.
- [San Francisco Equity Priority Communities](https://opendata-mtc.opendata.arcgis.com/datasets/MTC::equity-priority-communities-plan-bay-area-2040/about): Equity Priority Communities, also referred to as Communities of Concern, are census tracts of higher equity risk identified by the Metropolitan Transportation Commission.
Factors used to classify tracts include vehicle access, racial composition, and the proportion of elderly and youth dependents, individuals with disabilities, and low-income households.
Census tract shapefiles were obtained from the San Francisco Open Data Portal and contain polygon boundaries for all census tracts within San Francisco.
- [NCIC Code Jurisdiction List](https://github.com/utexas-SDS-357/sds357-project-sp26-con-founders/blob/main/did/data/raw_data/NCIC%20Code%20Jurisdiction%20List_04242023%20-%20Sheet1.csv): The National Crime Information Center (NCIC) assigns 4-digit codes to identify agency jurisdictions. In the SWITRS dataset, county designations align with these NCIC identifiers, allowing for consistent geographic mapping and enrichment.
- [Astral Conditions](https://sffjunkie.github.io/astral/): Solar cycle data were collected using the `astral` package.
These data are used to determine whether traffic stops occur during day, night, dawn, or dusk for the San Francisco area.

## Installation Instructions
To clone this repository to your local machine:
1. Download [git](https://git-scm.com/).
2. Open your Terminal/Command Prompt application.
3. Change your working directory to the local folder where you want the repository to be saved (change `local_folder_path` to the actual path name).
```
cd local_folder_path
```
4. Clone the github repository.
```
git clone https://github.com/utexas-SDS-357/sds357-project-sp26-con-founders.git
```
A new directory named after the repository will be created, containing all files, history, and metadata.
By default, the main branch will be visible and contain the following files:

| Directory | Subdirectory | File descriptions |
| ----------| ------------ | ----------------- |
| `did` | `preprocessing`<br>`data`<br>`analysis`<br>`output` |- Preprocessing files used to clean traffic collision datasets<br>- Raw and corresponding clean collision data obtained from processing pipeline<br>- Code used for exploratory data analysis and modeling collision outcomes<br>- Relevant output from collision analysis pipeline, including graphics used in presentations/reports |
| `bias` | `preprocessing`<br>`data`<br>`analysis`<br>`output` |- Preprocessing files necessary to clean traffic stop datasets<br>- Raw and corresponding clean stop data obtained from processing pipeline<br>- Files used for exploratory data analysis and modeling stop outcomes<br>- Relevant output from bias analysis pipeline, including graphics used in presentations/reports |

## Usage Instructions
To reproduce any results or make updates to the pipeline, use the installation instructions above to access the relevant files. 
Code can be accessed and edited using a shell or integrated development environment (e.g. VSCode) that supports Python. 

To access large data files, ensure `lfs` is installed:

```
git lfs install
```
Pull data by calling:
```
git lfs pull
```

### Dependencies
Necessary dependencies including python packages are included in the `requirements.txt` file within the preprocessing and analysis subdirectories. 
Notebooks contain the code necessary to install these packages. 
