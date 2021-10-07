"""result models for display in IPython environments"""
import base64
from io import BytesIO
from typing import Dict, List

import numpy as np
from mercantile import Tile
from PIL import Image, ImageDraw

from label_maker_dask.utils import class_color

style_helper = "display:inline-block;vertical-align:middle;margin-left:1em;"


class ClassificationResult:
    """TODO: params, return; class for classification label results"""

    def __init__(
        self, tile: Tile, label: np.array, classes: List[Dict], image: np.array
    ):
        """initialize new result"""
        self.tile = tile
        self.label = label
        self.image = image
        self.classes = classes

    def show_label(self):
        """show label"""
        names = ["Background"] + [cl["name"] for cl in self.classes]
        label_dict = dict(zip(names, self.label.astype(bool)))
        rows = [f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in label_dict.items()]
        return f"<table style={style_helper}><tr><th>Class</th><th>Value</th></tr>{''.join(rows)}</table>"

    def show_image(self):
        """show image"""
        return Image.fromarray(self.image.astype(np.uint8))

    def _repr_html_(self):
        """show custom HTML card"""
        imageio = BytesIO()
        self.show_image().save(imageio, format="JPEG")
        image_str = base64.b64encode(imageio.getvalue()).decode("utf-8")

        elem = f"<div style='border-radius:5px;background-color:#eee;padding:2em;'><span>{self.tile}</span>{self.show_label()}<img style={style_helper} src='data:image/jpeg;base64,{image_str}'/></div>"

        return elem


class ObjectDetectionResult:
    """TODO: params, return; class for object detection label results"""

    def __init__(
        self, tile: Tile, label: np.array, classes: List[Dict], image: np.array
    ):
        """initialize new result"""
        self.tile = tile
        self.label = label
        self.image = image
        self.classes = classes

    def draw_label(self, img):
        """draw label on an image"""
        draw = ImageDraw.Draw(img)
        for box in self.label:
            draw.rectangle(
                ((box[0], box[1]), (box[2], box[3])), outline=class_color(box[4])
            )
        return img

    def show_label(self):
        """show label"""
        img = Image.new("RGB", (256, 256))
        label = self.draw_label(img)
        return label

    def show_image(self):
        """show image"""
        return Image.fromarray(self.image.astype(np.uint8))

    def _repr_html_(self):
        """show custom HTML card"""
        combinedio = BytesIO()
        image = self.show_image()
        self.draw_label(image).save(combinedio, format="JPEG")
        combined_str = base64.b64encode(combinedio.getvalue()).decode("utf-8")

        elem = f"<div style='border-radius:5px;background-color:#eee;padding:2em;'><span>{self.tile}</span><img style={style_helper} src='data:image/jpeg;base64,{combined_str}'/></div>"

        return elem


class SegmentationResult:
    """TODO: params, return; class for segmentation label results"""

    def __init__(
        self, tile: Tile, label: np.array, classes: List[Dict], image: np.array
    ):
        """initialize new result"""
        self.tile = tile
        self.label = label
        self.image = image
        self.classes = classes

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

        elem = f"<div style='border-radius:5px;background-color:#eee;padding:2em;'><span>{self.tile}</span><img style={style_helper} src='data:image/jpeg;base64,{label_str}'/><img style={style_helper} src='data:image/jpeg;base64,{image_str}'/></div>"

        return elem
