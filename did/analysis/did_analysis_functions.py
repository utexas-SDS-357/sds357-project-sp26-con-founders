import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime as dt
import zipfile

def load_data(path, geospatial = False):
    """
    Load tabular or geospatial data from a file path into a DataFrame or GeoDataFrame.
 
    path : str
        Relative file path to the data source. Accepted formats are
        .csv, .zip (containing one .csv), and .geojson.
    geospatial : bool, optional
        If True, the loaded DataFrame is converted to a GeoDataFrame using
        automatically detected latitude and longitude columns. Default is False.
 
    Returns
    pandas.DataFrame
        DataFrame containing the loaded data, when geospatial is False.
    geopandas.GeoDataFrame
        A GeoDataFrame with a geometry column constructed from latitude and longitude
        values.
    """

    # get filename from path
    file_name = get_filename(path)
    
    # check if zipped
    if(path[-3:]=="zip"):
        with zipfile.ZipFile(path) as z:
            csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
            df = pd.read_csv(z.open(csv_file), low_memory=False)
        print(f"{file_name} file unzipped.")

    # read in regularly for non zipped files
    else:
        if("geojson" in path.rsplit('/', 1)[-1]):
            df = gpd.read_file(path)
        else:
            df = pd.read_csv(path, low_memory=False)

    if(not geospatial):
        print(f"{file_name} converted to dataframe.")
        return df
    # convert to geopandas df
    else:
        df = coerce_to_gpd(df)
        
        print(f"{file_name} geospatial data converted to geopandas dataframe.")
        return df

def get_filename(file_path):
    """
    Extract the filename from a file path string.
 
    file_path : str
        An absolute or relative file path (e.g., '../data/raw/stops.csv.zip').
 
    Returns
    str
        The filename component of the path, including extension
        (e.g., 'stops.csv.zip').
    """
    # Splits by the last occurrence of '/' and returns the last element
    return file_path.rsplit('/', 1)[-1]

def coerce_to_gpd(df):
    """
    Convert a pandas DataFrame to a GeoDataFrame by constructing point geometries
    from latitude and longitude columns.
 
    Column names are matched case-insensitively against common variants: 'latitude'
    and 'lat' for latitude, and 'longitude', 'long', and 'lng' for longitude. 

    df : pandas.DataFrame
        The input DataFrame. Must contain at least one column matching a recognized
        latitude name and one column matching a recognized longitude name.
 
    Returns
    geopandas.GeoDataFrame
        The original DataFrame with an added 'geometry' column of Point geometries
        constructed from the detected latitude and longitude columns. 
    """
    latitude_names = ['latitude', 'lat']
    longitude_names = ['longitude', 'long', 'lng']
    lat_column_name = None
    long_column_name = None
    for col in df.columns:
        if col.lower() in latitude_names:
            lat_column_name = col
        if col.lower() in longitude_names:
            long_column_name = col
        if lat_column_name and long_column_name:
            break
        
    df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[long_column_name], df[lat_column_name]),
            # set coordinate reference system for lat/long data
            crs="EPSG:4326" 
        )
    
    print("df coerced to geopandas dataframe")
    return df