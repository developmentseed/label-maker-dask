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

from label_maker_dask.label import get_label
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
        label = get_label(features, classes, ml_type)
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
        return Image.fromarray(self.image.astype(np.uint8))

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
