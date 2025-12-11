"""
Image augmentation utilities for IFCB focus detection.

This module provides functions for generating augmented training data.
"""

import os
import numpy as np
from skimage.filters import gaussian
from skimage.io import imread, imsave

from ifcb_focus.training.utils import list_images


def blur_image(image, sigma=1):
    """
    Apply Gaussian blur to an image.

    Args:
        image (ndarray): Input image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        ndarray: Blurred image.

    Example:
        >>> from ifcb_focus.utils import blur_image
        >>> import numpy as np
        >>> image = np.random.rand(100, 100)
        >>> blurred = blur_image(image, sigma=2.0)
    """
    return gaussian(image, sigma=sigma, preserve_range=True)


def augment_dataset(good_dir, blurred_dir, sigma_range=(2.0, 4.0)):
    """
    Create augmented dataset by blurring good images.

    Args:
        good_dir (str): Directory containing good (in-focus) images.
        blurred_dir (str): Directory where blurred images will be saved.
        sigma_range (tuple): (min_sigma, max_sigma) for random blur strength.
            Default is (2.0, 4.0).

    Example:
        >>> from ifcb_focus.utils import augment_dataset
        >>> augment_dataset('/path/to/good', '/path/to/blurred')
    """
    os.makedirs(blurred_dir, exist_ok=True)

    for image_path in list_images(good_dir):
        image = imread(image_path)
        sigma = np.random.uniform(*sigma_range)  # Random sigma for each image
        blurred_image = blur_image(image, sigma=sigma)
        blurred_image_path = os.path.join(blurred_dir, os.path.basename(image_path))
        # save as png
        imsave(blurred_image_path, blurred_image.astype(np.uint8))

    print(f"Augmented {len(list_images(good_dir))} images saved to {blurred_dir}")
