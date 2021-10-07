# pylint: disable=unused-argument
"""Provide utility functions"""
import os
from io import BytesIO
from urllib.parse import parse_qs

import numpy as np
import rasterio
import requests  # type: ignore
from mercantile import Tile, bounds
from PIL import Image, ImageColor
from rasterio.crs import CRS
from rasterio.warp import transform_bounds
from rio_tiler.io import COGReader

WGS84_CRS = CRS.from_epsg(4326)


class SafeDict(dict):
    """dictionary for replacing missing url properties"""

    def __missing__(self, key):
        """replace missing url properties"""
        return "{" + key + "}"


def url(tile: Tile, imagery):
    """Return a tile url provided an imagery template and a tile"""
    return imagery.format(x=tile.x, y=tile.y, z=tile.z)


def class_match(ml_type, label, i):
    """Determine if a label matches a given class index"""
    if ml_type == "classification":
        return label[i] > 0
    elif ml_type == "object-detection":
        return len(list(filter(lambda bb: bb[4] == i, label)))
    elif ml_type == "segmentation":
        return np.count_nonzero(label == i)
    return None


def download_tile_tms(tile: Tile, imagery):
    """Download a satellite image tile from a tms endpoint"""

    if os.environ.get("ACCESS_TOKEN"):
        token = os.environ.get("ACCESS_TOKEN")
        imagery = imagery.format_map(SafeDict(ACCESS_TOKEN=token))

    r = requests.get(url(tile, imagery))

    return np.array(Image.open(BytesIO(r.content)))


def get_tile_tif(tile, imagery):
    """
    Read a GeoTIFF with a window corresponding to a TMS tile
    """
    with COGReader(imagery) as image:
        img = image.tile(*tile)

    return np.moveaxis(img.data, 0, 2)


def get_tile_wms(tile, imagery):
    """
    Read a WMS endpoint with query parameters corresponding to a TMS tile

    Converts the tile boundaries to the spatial/coordinate reference system
    (SRS or CRS) specified by the WMS query parameter.
    """
    # retrieve the necessary parameters from the query string
    query_dict = parse_qs(imagery.lower())
    wms_version = query_dict.get("version")[0]
    if wms_version == "1.3.0":
        wms_srs = query_dict.get("crs")[0]
    else:
        wms_srs = query_dict.get("srs")[0]

    # find our tile bounding box
    bound = bounds(*[int(t) for t in tile])
    xmin, ymin, xmax, ymax = transform_bounds(
        WGS84_CRS, CRS.from_string(wms_srs), *bound, densify_pts=21
    )

    # project the tile bounding box from lat/lng to WMS SRS
    bbox = (
        [ymin, xmin, ymax, xmax] if wms_version == "1.3.0" else [xmin, ymin, xmax, ymax]
    )

    # request the image with the transformed bounding box and save
    wms_url = imagery.replace("{bbox}", ",".join([str(b) for b in bbox]))
    r = requests.get(wms_url)

    return np.array(Image.open(BytesIO(r.content)))


def is_tif(imagery):
    """Determine if an imagery path leads to a valid tif"""
    valid_drivers = ["GTiff", "VRT"]
    try:
        with rasterio.open(imagery) as test_ds:
            if test_ds.meta["driver"] not in valid_drivers:
                # rasterio can open path, but it is not a tif
                valid_tif = False
            else:
                valid_tif = True
    except rasterio.errors.RasterioIOError:
        # rasterio cannot open the path. this is the case for a
        # tile service
        valid_tif = False

    return valid_tif


def is_wms(imagery):
    """Determine if an imagery path is a WMS endpoint"""
    return "{bbox}" in imagery


def get_image_function(imagery):
    """Return the correct image downloading function based on the imagery string"""
    if is_tif(imagery):
        return get_tile_tif
    if is_wms(imagery):
        return get_tile_wms
    return download_tile_tms


# Taken from https://github.com/CartoDB/CartoColor/blob/master/cartocolor.js#L1633-L1733
colors = ["#DDCC77", "#CC6677", "#117733", "#332288", "#AA4499", "#88CCEE"]


def class_color(c):
    """Return 3-element tuple containing rgb values for a given class"""
    if c == 0:
        return (0, 0, 0)  # background class
    return ImageColor.getrgb(colors[c % len(colors)])
