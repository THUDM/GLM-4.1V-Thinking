# -*- coding: utf-8 -*-


import base64
import io

from PIL import Image


def encode_image(image_file: str, prefix: bool = False) -> str:
    """
    Encode an image to base64 string.

    Args:
        image_file (str): Path to the image file
        prefix (bool): Whether to add data URL prefix

    Returns:
        str: Base64 encoded image string
    """
    img = Image.open(image_file)
    img_io = io.BytesIO()
    img.convert("RGB").save(img_io, format="JPEG")
    if prefix:
        return "data:image/jpeg;base64," + base64.b64encode(img_io.getvalue()).decode("utf-8")
    return base64.b64encode(img_io.getvalue()).decode("utf-8")
