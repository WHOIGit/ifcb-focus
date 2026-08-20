"""
Image preprocessing for IFCB focus detection.

Provides the standard preprocessing pipeline for IFCB ROI images: float
normalization, adaptive histogram equalization, and resizing.
"""

import numpy as np
from skimage.util import img_as_float
from skimage.exposure import equalize_adapthist
from skimage.transform import resize


def preprocess_image(image, target_size=(512, 512), dtype='float32'):
    """
    Preprocess an image for feature extraction.

    Applies float normalization, adaptive histogram equalization, and resizing.
    Accepts a ``PIL.Image`` or a ``numpy.ndarray`` (2D grayscale or 3D RGB);
    any non-array input is normalized to a numpy array first. This is the
    adapter boundary between ``ifcbkit`` (which yields PIL images) and the
    numpy/skimage feature pipeline.

    Args:
        image: Input image (PIL.Image or ndarray).
        target_size (tuple): Target (height, width). Default (512, 512).
        dtype (str): Output dtype ('float32', 'float64', 'float16').

    Returns:
        numpy.ndarray: Preprocessed image with the specified dtype.
    """
    if dtype not in ['float32', 'float64', 'float16']:
        raise ValueError(f"Unsupported dtype: {dtype}. Use 'float32', 'float64', or 'float16'.")

    # Adapter boundary: normalize PIL images (and arrays) to numpy.
    image = np.asarray(image)

    # Convert to float and normalize to [0, 1]
    image = img_as_float(image)

    # Apply adaptive histogram equalization
    image = equalize_adapthist(image)

    # Resize to target dimensions
    image = resize(image, target_size, anti_aliasing=True)

    return image.astype(
        np.float32 if dtype == 'float32'
        else np.float64 if dtype == 'float64'
        else np.float16
    )
