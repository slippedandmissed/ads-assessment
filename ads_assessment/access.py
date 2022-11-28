from typing import Optional, List, Any, Tuple, Dict
from pymysql.connections import Connection
from datetime import date as date_t
from geopandas import GeoDataFrame
from dataclasses import dataclass
from tqdm import tqdm
from . import config
import pandas as pd
import osmnx as ox
import dateutil
import requests
import pymysql
import os

ox.settings.timeout = 1000


def make_connection(
    override_database_name: Optional[str] = None,
    local_infile: bool = False
) -> Connection:
    """Connect to the MySQL database using the credentials from .config"""
    return pymysql.connect(
        user=config.creds["username"],
        password=config.creds["password"],
        host=config.creds["host"],
        database=config.creds["database_name"] if override_database_name is None else override_database_name,
        local_infile=local_infile,
    )


def execute(
    conn: Connection,
    sql: str,
    *args: List[Any]
) -> None:
    """Execute a SQL command, discarding the output."""
    with conn.cursor() as cur:
        cur.execute(sql, args=args)
    conn.commit()


def query(
    conn: Connection,
    sql: str,
    *args: List[Any]
) -> Tuple[Tuple[Any]]:
    """Execute a SQL command, returning the resulting records."""
    with conn.cursor() as cur:
        cur.execute(sql, *args)
        return cur.fetchall()


def upload_file(
    conn: Connection,
    table: str,
    path: str
) -> None:
    """Upload a local CSV file into a table."""
    # This is a potential SQL injection surface
    # but there doesn't seem to be an easy way
    # around it, as PyMySQL always escapes parameters
    # with single quotes (') rather than backticks (`)
    # which causes a syntax error when applied to
    # table names
    execute(
        conn,
        f"""LOAD DATA LOCAL INFILE %s
INTO TABLE `{table}`
FIELDS TERMINATED BY ','
OPTIONALLY ENCLOSED BY '"'
LINES STARTING BY ''
TERMINATED BY '\n';""",
        path
    )


def download_file(
    url: str,
    output_path: str,
    verbose: bool = True,
    force: bool = False
) -> None:
    """Download a file from a URL and store it locally."""
    if not force and os.path.exists(output_path):
        if verbose:
            print(
                f"Skipping download of {output_path} because it already exists"
            )
        return
    response = requests.get(url)
    with open(output_path, "wb") as fp:
        fp.write(response.content.strip(b"\n"))
    if verbose:
        print(f"Downloaded {output_path}")


def upload_pp_data(
    conn: Connection,
    year: int,
    part: Optional[int] = None,
    local_path_template: str = "./pp_data/y%d%s.csv",
    url_template: str = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-%d%s.csv"
) -> None:
    """Fetch and upload ProPublica property price data for a given year (and optionally, part) to the database."""
    part_str = "" if part is None else f"-part{part}"
    url = url_template % (year, part_str)
    local_path = local_path_template % (year, part_str)
    download_file(url, local_path)
    upload_file(conn, "pp_data", local_path)


# From Task D description:
#
#   we're going to use the prices for a particular region
#   in a given time period. This means we can select that
#   region and time period and build the joined data only
#   from the relevent rows from the two tables.
#
# This class encodes a region and a time period
# The helper function to calculate a Box can be
# found in .address
@dataclass
class Box:
    lat: float
    lng: float
    min_lat: float
    max_lat: float
    min_lng: float
    max_lng: float
    size_km: float
    date: date_t
    start_date: date_t
    end_date: date_t
    size_days: int


def join_pp_and_postcode_data(
    conn: Connection,
    box: Box
) -> None:
    """Perform the inner join between the property price data and the postcode data for a given region and time period."""

    # Create table
    execute(conn, "DROP TABLE IF EXISTS `prices_coordinates_data`;")
    execute(conn, """CREATE TABLE IF NOT EXISTS `prices_coordinates_data` (
  `price` int(10) unsigned NOT NULL,
  `date_of_transfer` date NOT NULL,
  `postcode` varchar(8) COLLATE utf8_bin NOT NULL,
  `property_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `new_build_flag` varchar(1) COLLATE utf8_bin NOT NULL,
  `tenure_type` varchar(1) COLLATE utf8_bin NOT NULL,
  `locality` tinytext COLLATE utf8_bin NOT NULL,
  `town_city` tinytext COLLATE utf8_bin NOT NULL,
  `district` tinytext COLLATE utf8_bin NOT NULL,
  `county` tinytext COLLATE utf8_bin NOT NULL,
  `country` enum('England', 'Wales', 'Scotland', 'Northern Ireland', 'Channel Islands', 'Isle of Man') NOT NULL,
  `lattitude` decimal(11,8) NOT NULL,
  `longitude` decimal(10,8) NOT NULL,
  `db_id` bigint(20) unsigned NOT NULL
) DEFAULT CHARSET=utf8 COLLATE=utf8_bin AUTO_INCREMENT=1 ;""")
    execute(conn, "ALTER TABLE `prices_coordinates_data` ADD PRIMARY KEY (`db_id`);")
    execute(conn, "ALTER TABLE `prices_coordinates_data` MODIFY `db_id` bigint(20) unsigned NOT NULL AUTO_INCREMENT,AUTO_INCREMENT=1;")

    # Perform join
    execute(
        conn,
        """INSERT INTO `prices_coordinates_data`
SELECT `price`, `date_of_transfer`, `pp_data`.`postcode`, `property_type`, `new_build_flag`, `tenure_type`, `locality`, `town_city`, `district`, `county`, `country`, `lattitude`, `longitude`, 0
FROM `pp_data`
INNER JOIN `postcode_data`
ON `pp_data`.`postcode`=`postcode_data`.`postcode`
WHERE `lattitude` > %s
AND `lattitude` < %s
AND `longitude` > %s
AND `longitude` < %s
AND `date_of_transfer` > %s
AND `date_of_transfer` < %s""",
        box.min_lat,
        box.max_lat,
        box.min_lng,
        box.max_lng,
        box.start_date,
        box.end_date
    )


def fetch_pp_and_postcode_data(
    conn: Connection
) -> pd.DataFrame:
    """Fetch the joined property price and postcode data from the database."""
    records = query(
        conn,
        "SELECT lattitude, longitude, date_of_transfer, property_type, town_city, price FROM prices_coordinates_data"
    )
    return pd.DataFrame(
        records,
        columns=["Lattitude", "Longitude", "Date", "Type", "City", "Price"]
    )


def fetch_pois(
    box: Box,
    tags: Dict[str, bool],
    verbose: bool = True
) -> Dict[date_t, GeoDataFrame]:
    """
    Fetch OSM places of interest in a given region and time period.

    Note that this can be quite a slow process because osmnx only
    lets you query one day at a time, so if the time period is
    multiple days wide, then we need to perform multiple osmnx
    queries. The result returned is a dict mapping each date in
    the time period to the osmnx results.
    """
    result: Dict[date_t, GeoDataFrame] = {}
    date = box.start_date
    with tqdm(total=box.size_days*2+1, leave=False, disable=not verbose) as pbar:
        while date <= box.end_date:
            overpass_settings = ox.settings.overpass_settings
            if date is not None:
                ox.settings.overpass_settings = overpass_settings + \
                    f'[date:"{date.strftime("%Y-%m-%dT%H:%M:%SZ")}"]'
            result_for_date = ox.geometries_from_bbox(
                box.max_lat, box.min_lat, box.min_lng, box.max_lng, tags)
            ox.settings.overpass_settings = overpass_settings
            result[date] = result_for_date
            date += dateutil.relativedelta.relativedelta(days=1)
            pbar.update()
    return result


def data(
    property_box: Box,
    poi_box: Box,
    tags: Dict[str, bool],
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[date_t, GeoDataFrame]]:
    """
    Join and fetch the property price and postcode data, and OSM places of interest, for a given region and time period.

    Note that the region for the property data might be different from that for the POI
    data. This is because we want to compare the POIs to the property, but if the property
    is right on the edge of the a region, we still want POIs on all sides of it. For this
    reason we can expect the POI region to slightly larger than the property region.
    """
    with make_connection() as conn:
        if verbose:
            print("Joining property price and postcode data...")
        join_pp_and_postcode_data(conn, property_box)
        if verbose:
            print("Joined.\nFetching joined data...")
        pp_and_postcode_data = fetch_pp_and_postcode_data(conn)
        if verbose:
            print("Fetched.")
    if verbose:
        print("Fetching POIs...")
    pois = fetch_pois(poi_box, tags, verbose=verbose)
    if verbose:
        print("Fetched.")
    return (pp_and_postcode_data, pois)
