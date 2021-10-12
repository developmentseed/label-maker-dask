"""labeling functions from: https://github.com/developmentseed/label-maker/blob/master/label_maker/label.py"""
import numpy as np
from rasterio.features import rasterize
from shapely.errors import TopologicalError
from shapely.geometry import Polygon, mapping, shape

from label_maker_dask.filter import create_filter


def get_label(tile_data, classes, ml_type):
    """return label for a set of features, classes, and ml_type"""
    try:
        features = tile_data["osm"]["features"]
        clip_mask = Polygon(((0, 0), (0, 255), (255, 255), (255, 0), (0, 0)))
        if ml_type == "classification":
            class_counts = np.zeros(len(classes) + 1, dtype=np.int)
            for i, cl in enumerate(classes):
                ff = create_filter(cl.get("filter"))
                class_counts[i + 1] = int(bool([f for f in features if ff(f)]))
            # if there are no classes, activate the background
            if np.sum(class_counts) == 0:
                class_counts[0] = 1
            return class_counts
        elif ml_type == "object-detection":
            bboxes = _create_empty_label(ml_type, classes)
            for feat in features:
                for i, cl in enumerate(classes):
                    ff = create_filter(cl.get("filter"))
                    if ff(feat):
                        geo = shape(feat["geometry"])
                        if cl.get("buffer"):
                            geo = geo.buffer(cl.get("buffer"), 4)
                        bb = _pixel_bbox(geo.bounds) + [i + 1]
                        bboxes = np.append(bboxes, np.array([bb]), axis=0)
            return bboxes
        elif ml_type == "segmentation":
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
                        except TopologicalError:
                            break
                        if cl.get("buffer"):
                            geo = geo.buffer(cl.get("buffer"), 4)
                        if not geo.is_empty:
                            geos.append((mapping(geo), i + 1))
            return rasterize(geos, out_shape=(256, 256))
    except (KeyError, ValueError):
        print("failed writing label for QA tile")
        return _create_empty_label(ml_type, classes)


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


def _create_empty_label(ml_type, classes):
    """Create an empty label for each ml type"""
    if ml_type == "classification":
        label = np.zeros(len(classes) + 1, dtype=np.int)
        label[0] = 1
        return label
    elif ml_type == "object-detection":
        return np.empty((0, 5), dtype=np.int)
    elif ml_type == "segmentation":
        return np.zeros((256, 256), dtype=np.int)
    return None
