import pandas as pd
import geopandas as gpd
import numpy as np
import zipfile
import os
from plotnine import *
from plotnine import guides, guide_legend
from scipy.stats import gaussian_kde
from shapely.ops import unary_union
from shapely.geometry import Polygon as SPoly
import matplotlib.pyplot as plt


EPC_COLORS = {
    "High": "#8ec3de",
    "Higher": "#3a93c3",
    "Highest": "#1065ab",
    "None": "#d6ccbf"
}

RACE_COLORS = {
    "Asian/Pacific Islander": "#00b09b",
    "White": "#f2891f",
    "Black": "#d442d1",
    "Hispanic": "#9eb000"
}


def load_data(path, geospatial=False):
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
        values, when geospatial is True.
    """
    file_name = path.rsplit('/', 1)[-1]

    if path[-3:] == "zip":
        with zipfile.ZipFile(path) as z:
            csv_file = [f for f in z.namelist() if f.endswith(".csv")][0]
            df = pd.read_csv(z.open(csv_file), low_memory=False)
        print(f"{file_name} file unzipped.")
    else:
        if "geojson" in path.rsplit('/', 1)[-1]:
            df = gpd.read_file(path)
        else:
            df = pd.read_csv(path, low_memory=False)

    print(f"{file_name} converted to dataframe.")
    return df


def add_year(df, date_col):
    """
    Parse a date column to datetime and extract a numeric year column.

    df : pandas.DataFrame
        The dataset containing the date column to be parsed.
    date_col : str
        The name of the column containing date values.

    Returns
    pandas.DataFrame
        The input DataFrame with date_col coerced to datetime and a new
        integer column 'year' appended.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year

    print("year column added")
    return df


def get_stops_per_capita(df, population, race_col="subject_race", year_col="year", scale=10000):
    """
    Compute the number of traffic stops per scaled unit of population for each
    race group and year.

    Rows whose race value is not present as a key in population are excluded
    from the output.

    df : pandas.DataFrame
        The stops dataset. Must contain the columns specified by race_col and year_col.
    population : dict of {str : int}
        A dictionary mapping race label strings to population counts.
    race_col : str, optional
        The name of the column containing race labels. Default is 'subject_race'.
    year_col : str, optional
        The name of the column containing year values. Default is 'year'.
    scale : int, optional
        The population denominator for the rate calculation. Default is 10000,
        yielding stops per 10,000 residents.

    Returns
    pandas.DataFrame
        A DataFrame grouped by year and race with columns: year, subject_race,
        stops, population, and stops_per_capita.
    """
    result = (
        df.groupby([year_col, race_col])
        .size()
        .reset_index(name="stops")
    )
    result["population"] = result[race_col].map(population)
    result = result.dropna(subset=["population"])
    result["stops_per_capita"] = result["stops"] / result["population"] * scale

    print("stops per capita calculated")
    return result


def get_outcome_rate(df, outcome_col, race_col="subject_race", year_col="year"):
    """
    Compute the proportion of stops resulting in a specified outcome for each
    race group and year.

    df : pandas.DataFrame
        The stops dataset. Must contain the columns specified by outcome_col,
        race_col, and year_col.
    outcome_col : str
        The name of the binary (0/1) column representing the outcome of interest
        (e.g., 'search_conducted', 'citation_issued', 'warning_issued').
    race_col : str, optional
        The name of the column containing race labels. Default is 'subject_race'.
    year_col : str, optional
        The name of the column containing year values. Default is 'year'.

    Returns
    pandas.DataFrame
        A DataFrame grouped by year and race with columns: year, subject_race,
        total_stops, outcome_count, and outcome_rate.
    """
    result = (
        df.groupby([year_col, race_col])
        .agg(
            total_stops=(outcome_col, "count"),
            outcome_count=(outcome_col, "sum")
        )
        .reset_index()
    )
    result["outcome_rate"] = result["outcome_count"] / result["total_stops"]

    print(f"{outcome_col} rate calculated")
    return result


def remap_race_labels(df, race_map, race_col="subject_race"):
    """
    Remap race label values in a DataFrame and remove rows with unrecognized labels.

    Rows whose race value does not appear as a key in race_map are dropped.

    df : pandas.DataFrame
        The dataset containing the race column to be remapped.
    race_map : dict of {str : str}
        A dictionary mapping raw race label strings to display strings.
    race_col : str, optional
        The name of the column containing race labels. Default is 'subject_race'.

    Returns
    pandas.DataFrame
        A copy of the input DataFrame with race_col values replaced according
        to race_map. Rows not present in race_map are removed.
    """
    df = df[df[race_col].isin(race_map.keys())].copy()
    df[race_col] = df[race_col].map(race_map)

    print("race labels remapped")
    return df


def prep_epc_polygons(epc_gdf, county_gdf, counties_of_interest):
    """
    Prepare an EPC GeoDataFrame for polygon plotting by filtering to a county,
    exploding multipolygons, and extracting coordinates into long format.

    epc_gdf : geopandas.GeoDataFrame
        Communities of concern GeoDataFrame. Must contain 'epc_class' and
        'geometry' columns.
    county_gdf : geopandas.GeoDataFrame
        County boundary GeoDataFrame. Must contain a 'county' column.
    counties_of_interest : list of str
        County names to retain (case-sensitive).

    Returns
    pandas.DataFrame
        A long-format DataFrame with columns x, y, group, and epc_class
        suitable for use with geom_polygon in plotnine.
    """
    epc = epc_gdf[['epc_class', 'geometry']].copy()
    epc['epc_class'] = epc['epc_class'].replace('NA', 'None').fillna('None')
    epc = epc.to_crs("EPSG:4326")

    boundaries = county_gdf[county_gdf['county'].isin(counties_of_interest)]
    epc = gpd.sjoin(epc, boundaries, predicate="within")
    epc = epc.explode(index_parts=False)

    epc["coords"] = epc.geometry.apply(lambda geom: list(geom.exterior.coords))
    df = epc.drop(columns="geometry").explode("coords")
    df[["x", "y"]] = pd.DataFrame(df["coords"].tolist(), index=df.index)
    df["group"] = df.index

    print("epc polygons prepared")
    return df


def prep_stops_period(stops_df, cutoff_date, races, race_map,
                      window_years=2, race_col="subject_race", date_col="date"):
    """
    Filter stops to a symmetric time window around a cutoff date, assign a
    before/after period label, and remap race labels.

    stops_df : pandas.DataFrame
        The cleaned stops dataset. Must contain date_col and race_col.
    cutoff_date : str
        The policy cutoff date string parseable by pandas.Timestamp
        (e.g., '2014-03-18').
    races : list of str
        Raw race label strings to retain before remapping.
    race_map : dict of {str : str}
        Mapping from raw to display race label strings.
    window_years : int, optional
        Number of years before and after the cutoff to include. Default is 2.
    race_col : str, optional
        Name of the race column. Default is 'subject_race'.
    date_col : str, optional
        Name of the date column. Default is 'date'.

    Returns
    pandas.DataFrame
        A filtered copy of stops_df with an ordered categorical 'period' column
        and remapped race labels.
    """
    cutoff = pd.Timestamp(cutoff_date)
    start = cutoff - pd.DateOffset(years=window_years)
    end = cutoff + pd.DateOffset(years=window_years)

    df = stops_df[stops_df[race_col].isin(races)].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df[(df[date_col] >= start) & (df[date_col] <= end)]

    before_label = f"Before {cutoff.strftime('%B %Y')}"
    after_label = f"After {cutoff.strftime('%B %Y')}"
    df["period"] = np.where(df[date_col] < cutoff, before_label, after_label)
    df["period"] = pd.Categorical(
        df["period"],
        categories=[before_label, after_label],
        ordered=True
    )
    df = remap_race_labels(df, race_map, race_col=race_col)

    print("stops period data prepared")
    return df


def get_clipped_contours(df, x_col, y_col, group_cols, boundary,
                         grid_size=200, levels=3):

    rows = []
    for keys, grp in df.groupby(group_cols):
        if len(grp) < 10:
            continue
        lngs, lats = grp[x_col].values, grp[y_col].values
        kde = gaussian_kde(np.vstack([lngs, lats]))

        xi = np.linspace(lngs.min(), lngs.max(), grid_size)
        yi = np.linspace(lats.min(), lats.max(), grid_size)
        Xi, Yi = np.meshgrid(xi, yi)
        Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)

        fig_tmp, ax_tmp = plt.subplots()
        cs = ax_tmp.contour(Xi, Yi, Zi, levels=levels)
        all_segs = cs.allsegs
        plt.close(fig_tmp)

        race, period = keys

        for level_idx, segs in enumerate(all_segs):
            for i, seg in enumerate(segs):
                if len(seg) < 3:
                    continue
                poly = SPoly(seg)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                clipped = poly.intersection(boundary)
                if clipped.is_empty:
                    continue
                geoms = list(clipped.geoms) if hasattr(clipped, "geoms") else [clipped]
                for j, geom in enumerate(geoms):
                    if geom.geom_type not in ("Polygon", "LineString"):
                        continue
                    coords = (
                        np.array(geom.exterior.coords)
                        if geom.geom_type == "Polygon"
                        else np.array(geom.coords)
                    )
                    for cx, cy in coords:
                        rows.append({
                            "lng": cx,
                            "lat": cy,
                            "subject_race": race,
                            "period": period,
                            "level_idx": level_idx,
                            "contour_group": f"{race}_{period}_{level_idx}_{i}_{j}",
                            "Stop Density": "Stop Concentration"
                        })

    result = pd.DataFrame(rows)
    # keep innermost (min) level only
    min_level = result["level_idx"].min()
    result = result[result["level_idx"] == min_level]

    print("contours computed")
    return result


def plot_epc_map(epc_polygon_df):
    """
    Plot a choropleth map of San Francisco communities of concern by EPC class.

    epc_polygon_df : pandas.DataFrame
        The output of prep_epc_polygons(). Must contain columns x, y, group,
        and epc_class.

    Returns
    plotnine.ggplot
        A ggplot object rendering a filled polygon map colored by EPC class.
    """
    return (
        ggplot(epc_polygon_df, aes(x="x", y="y", group="group", fill="epc_class"))
        + geom_polygon(color="black", size=0.1)
        + scale_fill_manual(name="Concern Level", values=EPC_COLORS)
        + labs(title="\nSan Francisco Communities of Concern")
        + coord_equal()
        + theme_void()
        + theme(
            plot_background=element_rect(fill="#FEF8F0"),
            panel_background=element_rect(fill="#FEF8F0"),
            legend_background=element_rect(fill="#FEF8F0", color="#FEF8F0"),
            figure_size=(10, 5),
            plot_title=element_text(size=20, weight="bold", ha="right"),
            legend_title=element_text(size=14),
            legend_text=element_text(size=14)
        )
    )


def plot_density_map(epc_polygon_df, contour_df):
    """
    Plot a faceted map of stop density contours overlaid on EPC class polygons,
    faceted by race and period.

    epc_polygon_df : pandas.DataFrame
        The output of prep_epc_polygons(). Must contain columns x, y, group,
        and epc_class.
    contour_df : pandas.DataFrame
        The output of get_clipped_contours(). Must contain columns lng, lat,
        contour_group, subject_race, period, and Stop Density.

    Returns
    plotnine.ggplot
        A ggplot object rendering a faceted density map with EPC polygon base layer.
    """
    return (
        ggplot()
        + geom_polygon(
            epc_polygon_df,
            aes(x="x", y="y", group="group", fill="epc_class"),
            color="black",
            size=0.1
        )
        + geom_path(
            contour_df,
            aes(x="lng", y="lat", group="contour_group", linetype="Stop Density"),
            color="#b31529",
            size=0.8
        )
        + facet_grid("subject_race ~ period")
        + scale_fill_manual(name="Concern Level", values=EPC_COLORS)
        + scale_linetype_manual(name="", values={"Stop Concentration": "solid"})
        + guides(linetype=guide_legend(override_aes={"color": "#b31529", "size": 1.5}))
        + coord_equal()
        + theme_void()
        + theme(
            figure_size=(16, 10),
            plot_title=element_text(size=20, weight="bold", ha="center"),
            legend_title=element_text(size=20),
            legend_text=element_text(size=20),
            strip_text=element_text(size=25, weight="bold"),
            plot_background=element_blank(),
            panel_background=element_blank(),
            legend_background=element_blank()
        )
    )


def plot_stops_proportion(stops_df, cutoff_date, date_col="date", epc_col="epc_class"):
    """
    Plot the proportion of stops over time by EPC class with a vertical line
    marking the policy cutoff date.

    stops_df : pandas.DataFrame
        The cleaned stops dataset. Must contain date_col and epc_col.
    cutoff_date : str
        The policy cutoff date string parseable by pandas.Timestamp.
    date_col : str, optional
        Name of the date column. Default is 'date'.
    epc_col : str, optional
        Name of the EPC class column. Default is 'epc_class'.

    Returns
    plotnine.ggplot
        A ggplot object rendering a faceted line chart of stop proportions over time.
    """
    df = stops_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    stop_counts = (
        df.groupby([epc_col, "month"])
        .size()
        .reset_index(name="n_stops")
    )
    stop_counts["prop_within_class"] = (
        stop_counts["n_stops"] /
        stop_counts.groupby(epc_col)["n_stops"].transform("sum")
    )

    return (
        ggplot(stop_counts, aes(x="month", y="prop_within_class"))
        + geom_line()
        + geom_vline(
            xintercept=pd.Timestamp(cutoff_date),
            linetype="dashed",
            color="blue"
        )
        + scale_x_date(date_breaks="2 years", date_labels="%Y", name="Year")
        + scale_y_continuous(name="Proportion of Stops")
        + facet_grid(f"~{epc_col}")
        + theme_linedraw()
        + theme(figure_size=(10, 5))
    )


def plot_stops_by_hour(stops_df, race_col="subject_race", epc_col="epc_class",
                       time_col="time"):
    """
    Plot the proportion of stops by race across hours of the day, faceted by
    whether the stop occurred within or outside a community of concern.

    stops_df : pandas.DataFrame
        The cleaned stops dataset. Must contain time_col, race_col, and epc_col.
    race_col : str, optional
        Name of the race column. Default is 'subject_race'.
    epc_col : str, optional
        Name of the EPC class column. Default is 'epc_class'.
    time_col : str, optional
        Name of the time column. Default is 'time'.

    Returns
    plotnine.ggplot
        A ggplot object rendering a faceted line chart of stop proportions by
        hour and race.
    """
    df = stops_df.copy()
    df["hour_rounded"] = pd.to_datetime(df[time_col]).dt.round('h').dt.hour
    df["is_epc"] = (df[epc_col] != "Non-EPC").map({
        True: "In Community of Concern",
        False: "Outside Community of Concern"
    })

    stop_counts = (
        df.groupby(["is_epc", "hour_rounded", race_col])
        .size()
        .reset_index(name="n_stops")
    )
    stop_counts = stop_counts[stop_counts[race_col] != "other"]
    stop_counts["proportion"] = stop_counts.groupby(["is_epc", race_col])["n_stops"].transform(lambda x: x / x.sum())
    stop_counts[race_col] = stop_counts[race_col].str.title()

    return (
        ggplot(stop_counts, aes(x="hour_rounded", y="proportion", color=race_col))
        + geom_line(size=2)
        + scale_x_continuous(
            limits=[0, 23],
            breaks=[0, 6, 12, 18],
            expand=(0, 0),
            name="Time of Day (Hour)"
        )
        + scale_y_continuous(name="Proportion of Stops by Race")
        + scale_color_manual(name="Subject Race", values=RACE_COLORS)
        + facet_grid("~is_epc")
        + theme_bw()
        + theme(figure_size=(12, 6))
    )


def plot_stops_per_capita(df, year_col="year", rate_col="stops_per_capita",
                          race_col="subject_race", scale=10000):
    """
    Plot traffic stops per capita by race group over time as a line chart.

    df : pandas.DataFrame
        The output of get_stops_per_capita(), with remapped race labels applied.
    year_col : str, optional
        Name of the year column. Default is 'year'.
    rate_col : str, optional
        Name of the per capita rate column. Default is 'stops_per_capita'.
    race_col : str, optional
        Name of the race column. Default is 'subject_race'.
    scale : int, optional
        Population denominator used in rate calculation, displayed in the
        y-axis label. Default is 10000.

    Returns
    plotnine.ggplot
        A ggplot object rendering a line chart of stops per capita by race over time.
    """
    return (
        ggplot(df, aes(x=year_col, y=rate_col, color=race_col))
        + geom_line(size=1.5)
        + scale_color_manual(name="Subject Race", values=RACE_COLORS)
        + labs(
            x="Year",
            y=f"Stops per {scale:,} Residents",
            title="Traffic Stops per Capita by Race"
        )
        + theme_bw()
        + theme(
            figure_size=(12, 6),
            legend_position="right",
            plot_title=element_text(size=20),
            axis_title=element_text(size=15),
            axis_text=element_text(size=15),
            legend_title=element_text(size=15),
            legend_text=element_text(size=15)
        )
    )


def plot_outcome_rate(df, title, y_label, year_col="year", rate_col="outcome_rate",
                      race_col="subject_race"):
    """
    Plot a stop outcome rate by race group over time as a line chart with points.

    df : pandas.DataFrame
        The output of get_outcome_rate(), with remapped race labels applied.
    title : str
        The title of the plot.
    y_label : str
        The y-axis label describing the outcome rate being plotted.
    year_col : str, optional
        Name of the year column. Default is 'year'.
    rate_col : str, optional
        Name of the outcome rate column. Default is 'outcome_rate'.
    race_col : str, optional
        Name of the race column. Default is 'subject_race'.

    Returns
    plotnine.ggplot
        A ggplot object rendering a line and point chart of the outcome rate
        by race over time.
    """
    return (
        ggplot(df, aes(x=year_col, y=rate_col, color=race_col))
        + geom_line(size=1.5)
        + geom_point(size=2)
        + scale_color_manual(name="Subject Race", values=RACE_COLORS)
        + labs(x="Year", y=y_label, title=title)
        + theme_minimal()
        + theme(
            figure_size=(12, 6),
            legend_position="right",
            plot_title=element_text(size=20),
            axis_title=element_text(size=15),
            axis_text=element_text(size=15),
            legend_title=element_text(size=15),
            legend_text=element_text(size=15)
        )
    )


def save_plot(plot, filename, output_dir="../output", width=12, height=6, dpi=150):
    """
    Save a plotnine ggplot object to a file in the specified output directory.

    plot : plotnine.ggplot
        The plot object to be saved.
    filename : str
        The output filename including extension (e.g., 'stops_per_capita.png').
    output_dir : str, optional
        The directory to save the file to. Default is '../output'.
    width : int, optional
        Plot width in inches. Default is 12.
    height : int, optional
        Plot height in inches. Default is 6.
    dpi : int, optional
        Resolution in dots per inch. Default is 150.

    Returns
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    plot.save(path, width=width, height=height, dpi=dpi)
    print(f"plot saved to {path}")
