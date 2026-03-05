[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/MFzEnxem)

## Data Pre-Processing

This branch contains all raw datasets and preprocessing materials used for cleaning, validating, and preparing data for analysis and modeling.

### Datasets
| Dataset                       | File Name                             | Description                                                                                      |
| ----------------------------- | ------------------------------------- | ------------------------------------------------------------------------------------------------ |
| EPC Map Shapefiles            | `epc_map_1216_shapefiles/`            | Shapefiles defining Environmental Justice (EPC) geographic boundaries used for spatial analysis. |
| EPC Map Data                  | `epc_map_1216.csv`                    | Tabular Environmental Justice indicator data corresponding to EPC geographic areas.              |
| Bay Area Counties             | `Bay_Area_County_Polygons.geojson`    | County boundary polygons for the Bay Area used for geographic filtering and mapping.             |
| 2010 Census Tracts            | `Census_2010_Tracts_20260219.geojson` | Census tract boundaries used for demographic and spatial aggregation.                            |
| Traffic Collisions (Original) | `collisions_raw.csv.zip`              | Raw San Francisco traffic collision records.                                                     |
| Traffic Collisions (Updated)  | `collisions_raw_updated.csv.zip`      | Master collision dataset.                                             |
| Police Stops                  | `sf_police_stops_raw.csv.zip`         | Raw San Francisco police stop records.                                                           |

### Preprocessing Scripts

`sf_analysis.ipynb`

This notebook performs initial data exploration and preprocessing steps, including loading raw datasets, generating preliminary visualizations, filtering date and time, column filtering, and feature engineering.

The notebook produces the following key working datasets:
- `collisions_clean.zip`
- `stops_clean.csv.zip`

These processed datasets form the foundation for subsequent modeling and policy analysis.
