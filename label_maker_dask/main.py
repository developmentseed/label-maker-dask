"""Create machine learning training data from satellite imagery and OpenStreetMap"""
import base64
from io import BytesIO
from typing import Any, Dict, List

import dask
import mapbox_vector_tile
import numpy as np
import requests  # type: ignore
from mercantile import Tile, tiles
from PIL import Image
from rasterio.features import rasterize
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, mapping, shape

from label_maker_dask.filter import create_filter
from label_maker_dask.utils import class_color, get_image_function


# delayed functions
# not methods to avoid serialization and large object issues
@dask.delayed
def tile_to_label(tile: Tile, ml_type: str, classes: Dict, label_source: str):
    """
    Parameters
    ------------
    tile: mercantile.Tile
        tile index
    ml_type: str
    classes: Dict
    label_source: str
    Returns
    ---------
    label: tuple
        The first element is a mercantile tile. The second element is a numpy array
        representing the label of the tile
    """

    url = label_source.format(x=tile.x, y=tile.y, z=tile.z)
    r = requests.get(url)
    r.raise_for_status()

    tile_data = mapbox_vector_tile.decode(r.content)
    try:
        features = tile_data["osm"]["features"]
        clip_mask = Polygon(((0, 0), (0, 255), (255, 255), (255, 0), (0, 0)))
        geos = []
        for feat in features:
            for i, cl in enumerate(classes):
                ff = create_filter(cl.get("filter"))
                if ff(feat):
                    feat["geometry"]["coordinates"] = _convert_coordinates(
                        feat["geometry"]["coordinates"]
                    )
                    geo = shape(feat["geometry"])
                    try:
                        geo = geo.intersection(clip_mask)
                    except TopologicalError as e:
                        print(e, "skipping")
                        break
                    if cl.get("buffer"):
                        geo = geo.buffer(cl.get("buffer"), 4)
                    if not geo.is_empty:
                        geos.append((mapping(geo), i + 1))

        label = rasterize(geos, out_shape=(256, 256))
    except (KeyError, ValueError):
        print(f"failed reading QA tile: {url}")
        label = np.zeros((256, 256))

    return (tile, label)


@dask.delayed
def get_image(tup, imagery):
    """fetch images"""
    tile, label = tup
    image_function = get_image_function(imagery)
    return Result(tile, label, image_function(tile, imagery))


class Result:
    """TODO: params, return; class for label results"""

    def __init__(self, tile: Tile, label: np.array, image: np.array):
        """initialize new result"""
        self.tile = tile
        self.label = label
        self.image = image

    def show_label(self):
        """show label"""
        visible_label = np.array(
            [class_color(label_class) for label_class in np.nditer(self.label)]
        ).reshape(256, 256, 3)

        return Image.fromarray(visible_label.astype(np.uint8))

    def show_image(self):
        """show image"""
        return Image.fromarray(np.moveaxis(self.image, 0, 2).astype(np.uint8))

    def _repr_html_(self):
        """show custom HTML card"""
        labelio = BytesIO()
        self.show_label().save(labelio, format="JPEG")
        label_str = base64.b64encode(labelio.getvalue()).decode("utf-8")

        imageio = BytesIO()
        self.show_image().save(imageio, format="JPEG")
        image_str = base64.b64encode(imageio.getvalue()).decode("utf-8")

        elem = f"<div style='border-radius:5px;background-color:#eee;padding:2em;'><span>{self.tile}</span><img style='display:inline-block;vertical-align:middle;margin-left:1em;' src='data:image/jpeg;base64,{label_str}'/><img style='display:inline-block;vertical-align:middle;margin-left:1em;' src='data:image/jpeg;base64,{image_str}'/></div>"

        return elem


class LabelMakerJob:
    """TODO: params, return; class for creating label maker dask job"""

    def __init__(
        self,
        zoom: int = None,
        bounds: List[float] = None,
        classes: List[Dict[Any, Any]] = None,
        imagery: str = None,
        label_source: str = None,
        ml_type: str = None,
    ):
        """initialize new label maker job for dask"""
        self.zoom = zoom
        self.bounds = bounds
        self.classes = classes
        self.imagery = imagery
        self.label_source = label_source
        self.ml_type = ml_type
        self.results = None

    def build_job(self):
        """create task list for labels and images"""
        self.tile_list = list(tiles(*self.bounds, [self.zoom]))
        label_tups = [
            tile_to_label(tile, self.ml_type, self.classes, self.label_source)
            for tile in self.tile_list
        ]
        self.tasks = [get_image(tup, self.imagery) for tup in label_tups]
        print("Sample graph")
        return dask.visualize(self.tasks[:3])

    def n_tiles(self):
        """return the number of tiles for a built job"""
        try:
            return len(self.tile_list)
        except AttributeError:
            print("Call build_job first to construct a tile list")
            return None

    def execute_job(self):
        """compute the labels and images"""
        self.results = dask.compute(*self.tasks)


def _convert_coordinates(coords):
    # for points, return the coordinates converted
    if isinstance(coords[0], int):
        return list(map(_pixel_bounds_convert, enumerate(coords)))
    # for other geometries, recurse
    return list(map(_convert_coordinates, coords))


def _pixel_bbox(bb):
    """Convert a bounding box in 0-4096 to pixel coordinates"""
    # this will have coordinates in xmin, ymin, xmax, ymax order
    # because we flip the yaxis, we also need to reorder
    converted = list(
        map(_pixel_bounds_convert, enumerate([bb[0], bb[3], bb[2], bb[1]]))
    )
    return _buffer_bbox(converted)


def _buffer_bbox(bb, buffer=4):
    """Buffer a bounding box in pixel coordinates"""
    return list(
        map(_clamp, [bb[0] - buffer, bb[1] - buffer, bb[2] + buffer, bb[3] + buffer])
    )


def _clamp(coordinate):
    """restrict a single coordinate to 0-255"""
    return max(0, min(255, coordinate))


def _pixel_bounds_convert(x):
    """Convert a single 0-4096 coordinate to a pixel coordinate"""
    (i, b) = x
    # input bounds are in the range 0-4096 by default: https://github.com/tilezen/mapbox-vector-tile
    # we want them to match our fixed imagery size of 256
    pixel = round(b * 255.0 / 4096)  # convert to tile pixels
    return pixel if (i % 2 == 0) else 255 - pixel  # flip the y axis
