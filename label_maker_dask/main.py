"""Create machine learning training data from satellite imagery and OpenStreetMap"""
import json
from typing import Any, Dict, List

import dask
import fiona
import numpy as np
from mercantile import bounds, tiles

from label_maker_dask.filter import create_filter
from label_maker_dask.utils import get_image_function


class LabelMakerJob:
    """TODO: class for creating label maker dask job"""

    def __init__(
        self,
        zoom: int = None,
        bounds: List[float] = None,
        classes: List[Dict[Any, Any]] = None,
        imagery: str = None,
        label_source: str = None,
        ml_type: str = None,
        cluster=None,
    ):
        """initialize new label maker job for dask"""
        self.zoom = zoom
        self.bounds = bounds
        self.classes = classes
        self.imagery = imagery
        self.label_source = label_source
        self.ml_type = ml_type
        self.cluster = cluster
        self.results = None

    def build_job(self):
        """create task list for labels and images"""
        tile_list = tiles(self.bounds, [self.zoom])
        label_tups = [self.label(tile) for tile in tile_list]
        self.tasks = [self.get_image(tup) for tup in label_tups]
        print("Sample graph")
        dask.visualize(self.results[:3])

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

    @dask.delayed
    def label(self, tile):
        """
        Parameters
        ------------
        tile: mercantile.Tile
            tile index
        label_job: dict
            Job definition
        Returns
        ---------
        label: tuple
            The first element is a mercantile tile. The second element is a numpy array
            representing the label of the tile
        """
        ml_type = self.ml_type  # noqa: F841
        classes = self.classes
        label_source = self.label_source

        tile_bounds = bounds(tile)
        features = []

        with fiona.open(label_source, "r") as src:
            for f in src.filter(
                bbox=(
                    tile_bounds.west,
                    tile_bounds.south,
                    tile_bounds.east,
                    tile_bounds.north,
                )
            ):
                f["properties"] = json.loads(f["properties"]["json"])
                features.append(f)

        # if ml_type == 'classification':
        class_counts = np.zeros(len(classes) + 1, dtype=np.int32)
        for i, cl in enumerate(classes):
            ff = create_filter(cl.get("filter"))
            class_counts[i + 1] = int(bool([f for f in features if ff(f)]))
        # if there are no classes, activate the background
        if np.sum(class_counts) == 0:
            class_counts[0] = 1

        return (tile, class_counts)

    @dask.delayed
    def get_image(self, tup):
        """fetch images"""
        tile, label = tup
        image_function = get_image_function(self.imagery)
        return image_function(tile, self.imagery)
