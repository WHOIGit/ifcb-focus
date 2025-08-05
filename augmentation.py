import os
import argparse
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


def augment_images(good_dir, blurred_dir):
    """Apply Gaussian blur augmentation to images in a directory.
    
    Processes all images in the input directory by applying random Gaussian blur
    with sigma values between 2.0 and 4.0, then saves the augmented images to
    the output directory.
    
    Args:
        good_dir (str): Path to directory containing source images to augment.
        blurred_dir (str): Path to directory where blurred images will be saved.
    """
    for image_path in list_images(good_dir):
        image = imread(image_path)
        sigma = np.random.uniform(2.0, 4.0)  # Random sigma for each image
        blurred_image = blur_image(image, sigma=sigma)
        blurred_image_path = os.path.join(blurred_dir, os.path.basename(image_path))
        # save as png
        imsave(blurred_image_path, blurred_image.astype(np.uint8))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment images by applying Gaussian blur')
    parser.add_argument('good_dir', help='Directory containing images to blur')
    parser.add_argument('blurred_dir', help='Directory to save blurred images')
    
    args = parser.parse_args()
    augment_images(args.good_dir, args.blurred_dir)
