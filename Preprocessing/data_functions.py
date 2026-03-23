import pandas as pd
import geopandas as gpd
import zipfile


def load_data(path, lat=None, long=None, geospatial=False):
    """
    Load CSV, GeoJSON, or zipped CSV data.

    Parameters
    ----------
    path : str
        Path to the data file.
    lat : str
        Latitude column name (required if geospatial=True).
    long : str
        Longitude column name (required if geospatial=True).
    geospatial : bool
        If True, convert lat/long columns to a GeoDataFrame.

    Returns
    -------
    pandas.DataFrame or geopandas.GeoDataFrame
    """

    # unzip data if name ends with .zip
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
            df = pd.read_csv(z.open(csv_file))
        print("Unzipped")

    else:
        if "geojson" in path.rsplit("/", 1)[-1]:
            df = gpd.read_file(path)
        else:
            df = pd.read_csv(path)

    # convert to geospatial dataframe
    if geospatial:
        df = gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df[long], df[lat]),
            crs="EPSG:4326"
        )

        print("Data CRS:", df.crs)

    return df


def filter_data_loc(df, county_df, counties_of_interest):
    """
    Filters data to relevant coordinates bounded by the additional shapefile provided.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe needing to be filtered
    county_df : geopandas.GeoDataFrame
        A geospatial dataframe containing polygons whose boundaries will be used as an extent to filter out extraneous datapoints.
    counties_of_interest : list containing strings
        Counties in the 'county' column of the county_df whose observations will be kept after filtering. 

    Returns
    -------
    geopandas.GeoDataFrame
    """
    
    # filter points to within bounaries of county polygon
    boundaries = county_df[county_df['county'].isin(counties_of_interest)]
    filtered_df = gpd.sjoin(df, boundaries, predicate="within")
     
    return filtered_df


def filter_data_time(df, dt_col, start, stop):
    """
    Filters data to relevant time period

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe needing to be filtered
    dt_col : str
        Name of the date column in df
    start : str
        Beginning date used for filtering. Dates preceeding this period will be filtered out. 
    stop : str
        Ending date used for filtering. Dates after this period will be filtered out. 

    Returns
    -------
    pandas.DataFrame
    """

    # check values are in datetime format
    df[dt_col] = pd.to_datetime(df[dt_col])

    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)
    # filter
    df = df[df[dt_col].between(start, stop)]

    return df