from typing import Dict, List, TypedDict, Iterable, Tuple
from datetime import date as date_t
from geopandas import GeoDataFrame
from .config import *
from tqdm import tqdm
from . import access
import pandas as pd
import osmnx as ox
import numpy as np
import dateutil
import warnings


def calculate_box(
    lat: float,
    lng: float,
    size_km: float,
    date: date_t,
    size_days: int
) -> access.Box:
    """Calculate the bounds of a geographical region and time period.

    The GPS coordinates of the centre of the range are given, and a
    "radius" in kilometres is given. The GPS coordinates of the square
    around that central point are calculated. The side length of the
    square will be *double* the value of size_km.
    """
    # Approximation based on https://en.wikipedia.org/wiki/Latitude#Meridian_distance_on_the_ellipsoid
    size_lat_deg = size_km / 110.574
    size_lng_deg = size_km / (111.320 * np.cos(np.deg2rad(lat)))

    return access.Box(
        lat,
        lng,
        lat - size_lat_deg,
        lat + size_lat_deg,
        lng - size_lng_deg,
        lng + size_lng_deg,
        size_km,
        date,
        date + dateutil.relativedelta.relativedelta(days=-size_days),
        date + dateutil.relativedelta.relativedelta(days=size_days),
        size_days
    )


def dilate_box(
    box: access.Box,
    dilation_km: float,
) -> access.Box:
    """Make the region spanned by a box larger by some number of kilometres on each side."""
    return calculate_box(
        box.lat,
        box.lng,
        box.size_km+dilation_km,
        box.date,
        box.size_days
    )


def get_random_coords_in_dataset(n: int) -> pd.DataFrame:
    """Get a random lattitude and longitude in the dataset."""
    with access.make_connection() as conn:
        results = access.query(
            conn, "SELECT lattitude, longitude FROM `postcode_data` ORDER BY RAND() LIMIT %s;", n)
        return pd.DataFrame(
            results,
            columns=["Lattitude", "Longitude"]
        )


def count_pois(
    pois: Dict[date_t, GeoDataFrame],
    box: access.Box,
    tags: Dict[str, bool]
) -> Dict[date_t, Dict[str, int]]:
    """Count the OSM places of interest within a given region and time period which have each tag.

    We are given a dict mapping dates to GeoDataFrames containing POIs on that date.
    From this we calculate for each date, how many POIs are there of each type
    on that date, within a given region.

    Note that the given region may be smaller than the one used to calculate
    the GeoDataFrames. For example, the GeoDataFrames might show all POIs in
    a city, but here we only want to count the ones near a specific property.
    This is done to minimise the number of slow calls to the OSM API.
    """
    result: Dict[date_t, Dict[str, int]] = {}
    for date in pois:
        if date < box.start_date or date > box.end_date:
            continue
        pois_date = pois[date]

        # Since centroids don't account for an elliptical earth,
        # it's discourages to use them with GPS coordinates, and
        # emits warnings when you do. However, we're calculating
        # centroids for polygons on the scale of buildings, not
        # e.g., countries, so innaccuracies will be small and
        # can be ignored.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lat_series = pois_date["geometry"].centroid.y
            lng_series = pois_date["geometry"].centroid.x

        pois_date_area = pois_date[(box.min_lat <= lat_series) & (lat_series <= box.max_lat) & (
            box.min_lng <= lng_series) & (lng_series <= box.max_lng)]
        result_for_date: Dict[str, int] = {}
        for tag in tags:
            if not tags[tag]:
                continue
            result_for_date[tag] = pois_date_area[tag].notna(
            ).sum() if tag in pois_date_area else 0
        result[date] = result_for_date
    return result


def dist_km(
    lat1: float,
    lng1: float,
    lat2: float,
    lng2: float
) -> float:
    """Calculate the distance in kilometres between two pairs of GPS coordinates."""
    # Approximation from https://www.omnicalculator.com/other/latitude-longitude-distance#obtaining-the-distance-between-two-points-on-earth-distance-between-coordinates

    R = 6371

    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)
    lng1 = np.deg2rad(lng1)
    lng2 = np.deg2rad(lng2)

    return 2 * R * np.arcsin(
        np.sqrt(
            np.sin((lat2-lat1)/2) ** 2 +
            (
                np.cos(lat1) *
                np.cos(lat2) *
                np.sin((lng2 - lng1)/2)**2
            )
        )
    )


def add_poi_data(
    df: pd.DataFrame,
    pois: Dict[date_t, GeoDataFrame],
    tags: Dict[str, bool],
    distance_around_property_km: float,
    verbose: bool = True
) -> pd.DataFrame:
    """Takes an existing DataFrame of properties, and appends the counts of each type of POI near that property."""
    result = df.copy()
    all_poi_counts: Dict[str, List[int]] = {}
    for tag in tags:
        all_poi_counts[tag] = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False, disable=not verbose):
        lat = float(row["Lattitude"])
        lng = float(row["Longitude"])
        date = row["Date"]

        box_around_property = calculate_box(
            lat,
            lng,
            distance_around_property_km,
            date,
            0)
        poi_counts = count_pois(pois, box_around_property, tags)

        for tag in tags:
            all_poi_counts[tag].append(poi_counts[date][tag])

    for tag in tags:
        result[tag] = all_poi_counts[tag]

    return result


def get_closest_city_to(
    lat: float,
    lng: float,
    df: pd.DataFrame
) -> str:
    """Get the name of the closest city to a pair of GPS coordinates."""
    df = df.copy()
    df["Change in Lattitude"] = pd.to_numeric(
        df["Lattitude"], downcast="float") - lat
    df["Change in Longitude"] = pd.to_numeric(
        df["Longitude"], downcast="float") - lng
    df["Sqr Distance"] = (pd.to_numeric(df["Change in Lattitude"], downcast="float") ** 2) + \
        (pd.to_numeric(df["Change in Longitude"], downcast="float") ** 2)
    closest_idx = df["Sqr Distance"].argmin()
    return df["City"].iloc[closest_idx]


def distance_km_to_city_center(
    lat: float,
    lng: float,
    city: str
) -> float:
    """Calculate the distance between a given coordinate and the centre of a given city."""
    # Since centroids don't account for an elliptical earth,
    # it's discourages to use them with GPS coordinates, and
    # emits warnings when you do. However, the exact geometric
    # centre of the outline of a city is always going to be an
    # approximation of the city centre anyway, so curvature
    # errors can be ignored.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        city_center = ox.geocode_to_gdf(city, which_result=1).centroid
    return dist_km(lat, lng, city_center.y[0], city_center.x[0])


def add_distance_to_city_data(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """Takes an existing DataFrame of properties, and appends the distance from each property to its city centre."""
    result = df.copy()
    distance_to_city_center: List[float] = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False, disable=not verbose):
        lat = float(row["Lattitude"])
        lng = float(row["Longitude"])
        city = row["City"]

        distance_to_city_center.append(
            distance_km_to_city_center(lat, lng, city)
        )

    result["Distance to City Center"] = distance_to_city_center

    return result


def data(
    property_box: access.Box,
    tags: Dict[str, bool],
    distance_around_property_km: float,
    verbose: bool = True
) -> pd.DataFrame:
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    poi_box = dilate_box(property_box, distance_around_property_km)
    df, pois = access.data(property_box, poi_box, tags, verbose=verbose)
    df = add_poi_data(
        df, pois, tags, distance_around_property_km, verbose=verbose)
    df = add_distance_to_city_data(df, verbose=verbose)
    return df


def calculate_days_since_start_of_dataset(
    start_date: date_t,
    dates: Iterable[date_t]
) -> List[int]:
    """For a list of dates, calculate the integer number of days between each the start of our dataset."""
    result: List[int] = []
    for date in dates:
        result.append((date - start_date).days)
    return result


def calculate_one_hot_from_property_type(property_types: Iterable[str]) -> Tuple[List[int]]:
    """Given a list of property types ("F", "S", "D", "T", or "O") return a tuple of one-hot vectors encoding them."""
    flats: List[int] = []
    semi_detached: List[int] = []
    detached: List[int] = []
    terraced: List[int] = []
    other: List[int] = []

    for t in property_types:
        flats.append(1 if t == "F" else 0)
        semi_detached.append(1 if t == "S" else 0)
        detached.append(1 if t == "D" else 0)
        terraced.append(1 if t == "T" else 0)
        other.append(1 if t == "O" else 0)

    return (flats, semi_detached, detached, terraced, other)


def make_design(
    lat: Iterable[float],
    lng: Iterable[float],
    date: Iterable[date_t],
    property_type: Iterable[str],
    tag_counts: Iterable[Iterable[int]],
    distance_to_city_center: Iterable[float],
    start_date: date_t = date_t(1996, 1, 1),
) -> np.ndarray:
    """Given some datapoints, create a design matrix ready for fitting a GLM."""
    x = np.stack((
        lat,
        lng,
        calculate_days_since_start_of_dataset(start_date, date),
        *calculate_one_hot_from_property_type(property_type),
        *np.array(tag_counts).T,
        distance_to_city_center,
        np.ones_like(lat)
    )).astype(float).T
    return x


class LabelledDataset(TypedDict):
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray


def labelled(
    df: pd.DataFrame,
    tags: Dict[str, bool],
    train_prop: float = 0.8,
    start_date: date_t = date_t(1996, 1, 1),
    seed: Optional[int] = None,
) -> LabelledDataset:
    """Provide a labelled set of data ready for supervised learning."""
    x_train_val = (
        np.array(df["Lattitude"]),
        np.array(df["Longitude"]),
        np.array(df["Date"]),
        np.array(df["Type"]),
        np.array(list(np.array(df[tag]) for tag in tags)).T,
        np.array(df["Distance to City Center"])
    )

    y_train_val = np.array(df["Price"], dtype=float)

    train_val_num = len(x_train_val[0])

    train_num = int(train_val_num * train_prop)
    train_val_idxs = np.arange(0, train_val_num)

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(train_val_idxs)

    train_idxs = train_val_idxs[:train_num]
    val_idxs = train_val_idxs[train_num:]

    x_train = tuple(x_[train_idxs] for x_ in x_train_val)
    x_val = tuple(x_[val_idxs] for x_ in x_train_val)

    y_train = y_train_val[train_idxs]
    y_val = y_train_val[val_idxs]

    design_train = make_design(*x_train, start_date=start_date)
    design_val = make_design(*x_val, start_date=start_date)

    return {
        "x_train": design_train,
        "y_train": y_train,
        "x_val": design_val,
        "y_val": y_val
    }
