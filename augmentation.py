import os
import numpy as np
from skimage.filters import gaussian
from skimage.io import imread, imsave
from train import list_images

def blur_image(image, sigma=1):
    """
    Apply Gaussian blur to the image.

    Parameters:
        image (ndarray): Input image.
        sigma (float): Standard deviation for Gaussian kernel.

    Returns:
        ndarray: Blurred image.
    """
    return gaussian(image, sigma=sigma, preserve_range=True)

BASE_DIR = os.environ.get('IFCB_DATA_DIR', './data')

good_dir = os.path.join(BASE_DIR, 'good_to_blur')
blurred_dir = os.path.join(BASE_DIR, 'blurred')

for image_path in list_images(good_dir):
    image = imread(image_path)
    sigma = np.random.uniform(2.0, 4.0)  # Random sigma for each image
    blurred_image = blur_image(image, sigma=sigma)
    blurred_image_path = os.path.join(blurred_dir, os.path.basename(image_path))
    # save as png
    imsave(blurred_image_path, blurred_image.astype(np.uint8))
