import pandas as pd
import geopandas as gpd
import numpy as np
import zipfile
from astral import LocationInfo
from astral.sun import sun
from datetime import date, timedelta
from zipfile import ZipFile

def load_data(path, geospatial = False):
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


def filter_location(df, county_df, counties_of_interest):
    # df is data to filter
    # county_df is a geospatial dataframe containing polygons that define the county of interest
    # counties of interest is a list of strings with each string corresponding to a county in the county_df that is necessary to keep

    # filter points to within bounaries of county polygon
    boundaries = county_df[county_df['county'].str.lower().isin([c.lower() for c in counties_of_interest])]
    filtered_df = gpd.sjoin(df, boundaries, predicate="within")

    print("locations filtered")
    return filtered_df


def filter_time(df, dt_col, start, stop):
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
    Extracts the filename using string rsplit
    """
    # Splits by the last occurrence of '/' and returns the last element
    return file_path.rsplit('/', 1)[-1]


def simplify_col(df, mapping, col_name):

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
    
    # create datetime string
    df['datetime'] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
    df['datetime'] = df['datetime'].dt.tz_localize(timezone, ambiguous="NaT", nonexistent="shift_forward")

    # drop nas in datetime
    df = df.dropna(subset=['datetime'])

    print("datetime column initialized")
    return df

def get_light_condition(df, sun_df, timezone):

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
