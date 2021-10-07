"""Create machine learning training data from satellite imagery and OpenStreetMap"""
from typing import Any, Dict, List

import dask
import mapbox_vector_tile
import requests  # type: ignore
from mercantile import Tile, tiles

from label_maker_dask.label import get_label
from label_maker_dask.result import (
    ClassificationResult,
    ObjectDetectionResult,
    SegmentationResult,
)
from label_maker_dask.utils import get_image_function


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
    label = get_label(tile_data, classes, ml_type)

    return (tile, label)


@dask.delayed
def get_image(tup, imagery, ml_type, classes):
    """fetch images"""
    tile, label = tup
    image_function = get_image_function(imagery)

    if ml_type == "classification":
        ResultClass = ClassificationResult
    elif ml_type == "object-detection":
        ResultClass = ObjectDetectionResult
    elif ml_type == "segmentation":
        ResultClass = SegmentationResult

    return ResultClass(tile, label, classes, image_function(tile, imagery))


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
        self.tasks = [
            get_image(tup, self.imagery, self.ml_type, self.classes)
            for tup in label_tups
        ]
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
