"""
Image preprocessing utilities for IFCB focus detection.
"""

import numpy as np
from skimage import exposure
from skimage.util import img_as_float
from skimage.transform import resize


def preprocess_image(image, target_size=None, enhance_contrast=True, clip_limit=0.03):
    """
    Preprocess the image: normalize, enhance contrast, and optionally resize.

    This function consolidates preprocessing logic from both feature_extraction.py
    and train.py to provide a unified interface with backward compatibility.

    Parameters:
        image (ndarray): Input image (2D grayscale or 3D RGB).
        target_size (tuple, optional): Optional (height, width) to resize the image.
            If None (default), no resizing is performed.
        enhance_contrast (bool): Whether to apply adaptive histogram equalization.
            Default is True.
        clip_limit (float): Clipping limit for CLAHE (Contrast Limited Adaptive
            Histogram Equalization). Default is 0.03.

    Returns:
        ndarray: Preprocessed 2D float32 image normalized to [0, 1].

    Examples:
        >>> # Simple preprocessing (matches feature_extraction.py behavior)
        >>> processed = preprocess_image(image)

        >>> # With resizing (matches train.py behavior)
        >>> processed = preprocess_image(image, target_size=(256, 256))

        >>> # Without contrast enhancement
        >>> processed = preprocess_image(image, enhance_contrast=False)
    """
    # Normalize to [0, 1] float
    image = img_as_float(image)

    # Contrast enhancement
    if enhance_contrast:
        image = exposure.equalize_adapthist(image, clip_limit=clip_limit)

    # Optional resizing
    if target_size is not None:
        image = resize(image, target_size, anti_aliasing=True, preserve_range=True)

    return image.astype(np.float32)
