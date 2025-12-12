"""
Training utilities for IFCB focus detection.

This module provides utility functions for loading images and computing features
during model training.
"""

import os
from tqdm import tqdm
from skimage.io import imread

from ifcb_focus.features import extract_features


def list_images(folder):
    """
    List all PNG images in a folder.

    Args:
        folder (str): Path to folder containing images.

    Returns:
        list: Sorted list of image file paths.
    """
    return sorted([os.path.join(folder, f) for f in sorted(os.listdir(folder)) if f.endswith('.png')])


def load_images(image_paths):
    """
    Generator that loads images from a list of paths.

    Args:
        image_paths (list): List of image file paths.

    Yields:
        tuple: (image_array, image_id) for each image.

    Example:
        >>> paths = list_images('/path/to/images')
        >>> for image, image_id in load_images(paths):
        ...     print(f'Processing {image_id}')
    """
    for path in tqdm(image_paths, desc="Loading images"):
        image = imread(path)
        image_id = os.path.splitext(os.path.basename(path))[0]
        yield image, image_id


def skip_image(image):
    """
    Determine if an image should be skipped based on its dimensions.

    Skips images smaller than 50x50 pixels.

    Args:
        image (ndarray): Input image.

    Returns:
        bool: True if the image should be skipped, False otherwise.
    """
    return image.shape[0] < 50 or image.shape[1] < 50


def compute_features(image, feature_subset=None):
    """
    Compute features for an image.

    This is a thin wrapper around extract_features for backwards compatibility
    with the training pipeline.

    Args:
        image (ndarray): Preprocessed 2D float32 image.
        feature_subset (list, optional): List of feature names to compute.
            If None, uses the default slim set of 7 features:
            ['tenengrad', 'dog_var', 'edge_density', 'local_std', 'mad',
             'sum_modified_laplacian', 'brenner_gradient']

    Returns:
        dict: Dictionary of computed features.
    """
    if feature_subset is None:
        # Default slim set of features for speed (from original train.py)
        feature_subset = [
            'tenengrad', 'dog_var', 'edge_density', 'local_std', 'mad',
            'sum_modified_laplacian', 'brenner_gradient'
        ]

    return extract_features(image, feature_subset)
