"""Create machine learning training data from satellite imagery and OpenStreetMap"""
from typing import Any, Dict, List

import dask
import mapbox_vector_tile
import numpy as np
import requests  # type: ignore
from affine import Affine
from mercantile import Tile, tiles, ul
from PIL import Image
from rasterio.features import rasterize
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, mapping, shape

from label_maker_dask.filter import create_filter
from label_maker_dask.utils import (
    EXTENT,
    class_color,
    degrees_per_pixel,
    get_image_function,
    project_feat,
)


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
        features = [
            project_feat(feat, tile.x, tile.y, tile.z, EXTENT)
            for feat in tile_data["osm"]["features"]
            if "Multi" not in feat["geometry"]["type"]
        ]
    except KeyError:
        print(f"failed reading QA tile: {url}")
        features = []

    clip_mask = Polygon(((0, 0), (0, 255), (255, 255), (255, 0), (0, 0)))
    geos = []
    for feat in features:
        for i, cl in enumerate(classes):
            ff = create_filter(cl.get("filter"))
            if ff(feat):
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

    w, n = ul(tile.x, tile.y, tile.z)
    resolution = degrees_per_pixel(tile.z, 0)
    transform = Affine(resolution, 0, w, 0, -resolution, n)

    label = rasterize(geos, out_shape=(256, 256), transform=transform)

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

    def show(self):
        """show result"""
        visible_label = np.array(
            [class_color(label_class) for label_class in np.nditer(self.label)]
        ).reshape(256, 256, 3)
        return (
            self.tile,
            Image.fromarray(visible_label.astype(np.uint8)),
            Image.fromarray(self.image.astype(np.uint8)),
        )


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
