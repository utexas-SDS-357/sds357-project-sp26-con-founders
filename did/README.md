# Difference in Differences Directory

Our analysis of Vision Zero impact on San Francisco traffic outcomes is centered around examining whether the changes reduced traffic-related fatalities and injuries. This directory contains the data, analysis, and output relevant to this objective.

## Data Description

The following datasets are used:

### Primary Datasets

- [California Statewide Integrated Traffic Records System](https://www.kaggle.com/datasets/alexgude/california-traffic-collision-data-from-switrs): The SWITRS data provide standardized records for California traffic collision outcomes. For the scope of this project, only events occurring in major California cities, including San Francisco, were used. The raw data contain over one million geocoded collision records from January 2010 to December 2016. Relevant fields include crash severity, environmental conditions, contributing factors, and geographic identifiers.

### Supplementary Datasets

- [NCIC Code Jurisdiction List](https://github.com/utexas-SDS-357/sds357-project-sp26-con-founders/blob/main/did/data/raw_data/NCIC%20Code%20Jurisdiction%20List_04242023%20-%20Sheet1.csv): The National Crime Information Center (NCIC) assigns 4-digit codes to identify agency jurisdictions. In the SWITRS dataset, county designations align with these NCIC identifiers, allowing for consistent geographic mapping and enrichment.

## Directory Structure

The directory structure is as follows:

| Directory | Subdirectory | File descriptions |
|----|----|----|
| `did` | `src`<br><br><br><br>`data`<br><br><br><br><br>`output` | **`src`**<br>`did_preprocessing.ipynb`: Preprocessing files used to clean traffic collision datasets<br>`did_analysis.ipynb`: Code used for exploratory data analysis and modeling collision outcomes <br>**`data`**:<br>`raw_data/`: Raw data used to process collsion data<br>`clean_data/`: Clean collision data from the `did_preprocessing.ipynb` pipeline<br>**`output`**:<br>Relevant output from collision analysis pipeline (including `did_analysis.ipynb`) used in presentations/reports |
