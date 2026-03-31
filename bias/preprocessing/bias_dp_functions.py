import pandas as pd
import geopandas as gpd
import numpy as np
import zipfile
from astral import LocationInfo
from astral.sun import sun
from datetime import date, timedelta
from zipfile import ZipFile

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


def filter_location(df, county_df, counties_of_interest):
    """
    Spatially filter a GeoDataFrame to retain only observations falling within
    specified county boundaries.
 
    df : geopandas.GeoDataFrame
        The dataset to be filtered. Must contain a geometry column of point geometries.
    county_df : geopandas.GeoDataFrame
        A GeoDataFrame containing county boundary polygons. Must include a column
        named 'county' with string county names.
    counties_of_interest : list of str
        A list of county names to retain. Rows whose 'county' value matches any
        entry in this list (case-insensitive) are kept.
 
    Returns
    geopandas.GeoDataFrame
        A filtered GeoDataFrame containing only rows from df whose geometry falls
        within the boundaries of the specified counties.
    """


    # filter points to within bounaries of county polygon
    boundaries = county_df[county_df['county'].str.lower().isin([c.lower() for c in counties_of_interest])]
    filtered_df = gpd.sjoin(df, boundaries, predicate="within")

    print("locations filtered")
    return filtered_df


def filter_time(df, dt_col, start, stop):
    """
    Filter a DataFrame to retain only rows within a specified date or datetime range.
 
    The column specified by dt_col is coerced to datetime format prior to filtering.
    The range is inclusive of both the start and stop values.
 
    df : pandas.DataFrame
        The dataset to be filtered. Must contain the column specified by dt_col.
    dt_col : str
        The name of the column containing date or datetime values to filter on.
    start : str
        The lower bound of the date range. Accepts any format parseable by
        pandas.to_datetime (e.g., '2010-01-01').
    stop : str
        The upper bound of the date range, inclusive. Accepts any format parseable
        by pandas.to_datetime.
 
    Returns
    pandas.DataFrame
        A filtered DataFrame containing only rows where dt_col falls within
        [start, stop].
    """
    
    # check values are in datetime format
    df[dt_col] = pd.to_datetime(df[dt_col])

    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)
    # filter
    df = df[df[dt_col].between(start, stop)]

    print("time filtered")
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


def simplify_col(df, mapping, col_name):
    """
    Replace a categorical column with a set of binary indicator columns.
 
    For each key-value pair in mapping, a new binary column is created and set to 1
    for rows where col_name matches the key. The original column is then dropped.
    
    df : pandas.DataFrame
        The dataset containing the column to be remapped.
    mapping : dict of {str : str}
        A dictionary mapping original category labels to new binary column names.
        For example, {'Moving Violation': 'moving'} creates a column 'moving' that
        equals 1 where col_name is 'Moving Violation'.
    col_name : str
        The name of the categorical column to be replaced.
 
    Returns
    pandas.DataFrame
        The DataFrame with col_name removed and one new binary (0/1) column added
        for each entry in mapping.
    """

    # initialize new column values
    for col in mapping.values():
        df[col] = 0
    
    # replace old column with mappings and drop
    for name, col in mapping.items():
        df.loc[df[col_name] == name, col] = 1
    df = df.drop(columns=col_name)

    print("features re-mapped")
    return df

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

def join_features(master_df, df_new, features, geopandas = False):
    """
    Append selected features from a secondary dataset onto a master DataFrame.
 
    When geopandas is True, a spatial join is performed, retaining only rows from
    master_df whose geometries fall within polygons in df_new. 
 
    master_df : pandas.DataFrame or geopandas.GeoDataFrame
        The base dataset onto which new features are joined.
    df_new : pandas.DataFrame or geopandas.GeoDataFrame
        The secondary dataset from which features are drawn. Subsetted to the
        columns specified in features before joining.
    features : list of str
        Column names to retain from df_new prior to the join. Must include
        'geometry' when geopandas is True.
    geopandas : bool, optional
        If True, performs a spatial join using gpd.sjoin() with the 'within'
        predicate, retaining only inner matches. 
 
    Returns
    pandas.DataFrame or geopandas.GeoDataFrame
        The master DataFrame with new columns from df_new appended. Return type
        matches the type of master_df.
    """

    # filter dataset to relevant features
    df_new = df_new[features]

    # for geopandas dataset, join observations within master dataset 
    if geopandas:
        master_df = coerce_to_gpd(master_df)
        # drop leftover index columns from previous joins
        master_df = master_df.drop(columns=[c for c in master_df.columns if 'index_' in c], errors='ignore')
        master_df = gpd.sjoin(master_df, df_new, how="inner", predicate="within")
    
    print("new features added")
    return master_df

def get_sun_df(start_date, end_date):
    """
    Generate a DataFrame of daily solar event times for an area over a
    specified date range.
 
    Parameters
    ----------
    start_date : datetime.date
        The first date for which solar event times are computed.
    end_date : datetime.date
        The last date for which solar event times are computed, inclusive.
 
    Returns
    -------
    pandas.DataFrame
        A DataFrame with one row per date and the following columns:
        - date : pandas.Timestamp, the calendar date
        - dawn : datetime, the time of civil dawn
        - sunrise : datetime, the time of sunrise
        - sunset : datetime, the time of sunset
        - dusk : datetime, the time of civil dusk
    """
    city = LocationInfo("San Francisco", "USA", "America/Los_Angeles", 37.7749, -122.4194)
    rows = []
    current = start_date
    while current <= end_date:
        s = sun(city.observer, date=current)
        rows.append({
            "date": pd.Timestamp(current),
            "dawn": s["dawn"],
            "sunrise": s["sunrise"],
            "sunset": s["sunset"],
            "dusk": s["dusk"]
        })
        current += pd.Timedelta(days=1)
    print("sun status data pulled")
    return pd.DataFrame(rows)

def create_datetime(df, timezone, date_col, time_col):
    """
    Construct a timezone-aware datetime column from separate date and time columns.
 
    df : pandas.DataFrame
        The dataset containing separate date and time columns.
    timezone : str
        A valid timezone string compatible with pytz or zoneinfo
        (e.g., 'America/Los_Angeles').
    date_col : str
        The name of the column containing date values.
    time_col : str
        The name of the column containing time values.
 
    Returns
    pandas.DataFrame
        The input DataFrame with an added 'datetime' column of timezone-aware
        Timestamps. Rows where 'datetime' is NaT are removed.
    """
    # create datetime string
    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
    df['datetime'] = df['datetime'].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")

    # drop nas in datetime
    df = df.dropna(subset=['datetime'])

    print("datetime column initialized")
    return df

def get_light_condition(df, sun_df, timezone):
    """
    Assign a categorical light condition to each observation based on its
    timestamp relative to daily solar event times.
 
    df : pandas.DataFrame
        The dataset to be annotated. Must contain a timezone-aware 'datetime'
        column and a 'date' column compatible with sun_df for merging.
    sun_df : pandas.DataFrame
        A DataFrame of daily solar event times, as produced by get_sun_df().
        Must contain columns: 'date', 'dawn', 'sunrise', 'sunset', and 'dusk'.
    timezone : str
        A valid timezone string used to convert solar event timestamps
        (e.g., 'America/Los_Angeles').
 
    Returns
    pandas.DataFrame
        The input DataFrame with an added 'light_condition' column containing
        ordered categorical values from the set {'night', 'dawn', 'day', 'dusk'},
        ordered as night < dawn < day < dusk. The columns 'datetime', 'sunrise',
        'sunset', 'dawn', and 'dusk' are removed from the returned DataFrame.
    """
    # merge datasets together
    df = df.merge(sun_df, on='date', how='left')
    for col in ['dawn', 'sunrise', 'sunset', 'dusk']:
        df[col] = df[col].dt.tz_convert(timezone)
        # If sun time ends up before midnight of the date, shift to correct day
        df[col] = df.apply(
            lambda row: row[col] + pd.Timedelta(days=1) if row[col].date() < row['date'].date() else row[col],
            axis=1
        )

    # assign condition based on datetime
    conditions = [
        (df['datetime'] >= df['dawn']) & (df['datetime'] < df['sunrise']),
        (df['datetime'] >= df['sunrise']) & (df['datetime'] <= df['sunset']), 
        (df['datetime'] > df['sunset']) & (df['datetime'] <= df['dusk']), 
        (df['datetime'] < df['dawn']) | (df['datetime'] > df['dusk']) 
    ]
    choices = ["dawn", "day", "dusk", "night"]
    df['light_condition'] = pd.Categorical(
        np.select(conditions, choices, default="night"),
        categories=["night", "dawn", "day", "dusk"],
        ordered=True
    )

    # drop unnecessary columns
    unnecessary_cols = ['sunrise', 'sunset', 'dawn', 'dusk', 'datetime']
    df = df.drop(columns = unnecessary_cols)

    print("added light conditions to df")
    return(df)
