"""
Feature extraction functions for IFCB focus detection.

This module contains all image sharpness and texture feature computation functions.
"""

import numpy as np
import pywt
from skimage.filters import sobel
from skimage import feature
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.ndimage import convolve, gaussian_filter, sobel as ndi_sobel, laplace
from scipy.stats import kurtosis

from .constants import FEATURE_NAMES, FEATURE_SPEED_RANKING


def sum_modified_laplacian(image):
    """
    Compute the Sum Modified Laplacian (SML) sharpness.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Sum of modified Laplacian values.
    """
    kernel_h = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
    kernel_v = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
    lap_h = convolve(image, kernel_h, mode='reflect')
    lap_v = convolve(image, kernel_v, mode='reflect')
    return np.sum(np.abs(lap_h) + np.abs(lap_v))


def tenengrad_sharpness(image):
    """
    Compute Tenengrad sharpness using Sobel gradient magnitude.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Mean squared Sobel gradient magnitude.
    """
    sobel_grad = sobel(image)
    return np.mean(sobel_grad ** 2)


def brenner_gradient(image):
    """
    Brenner's gradient — measures vertical detail.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Sum of squared differences between pixels 2 columns apart.
    """
    return np.sum((image[:, :-2] - image[:, 2:]) ** 2)


def dog_variance(image, sigma1=1, sigma2=2):
    """
    Difference of Gaussians (DoG) variance.

    Args:
        image (ndarray): Preprocessed 2D float image.
        sigma1 (float): Standard deviation for first Gaussian blur.
        sigma2 (float): Standard deviation for second Gaussian blur.

    Returns:
        float: Variance of the difference between two Gaussian-blurred images.
    """
    blur1 = gaussian_filter(image, sigma1)
    blur2 = gaussian_filter(image, sigma2)
    dog = blur1 - blur2
    return np.var(dog)


def edge_density(image, sigma=1.0):
    """
    Edge density via Canny edge detector.

    Args:
        image (ndarray): Preprocessed 2D float image.
        sigma (float): Standard deviation for Gaussian filter in Canny detector.

    Returns:
        float: Ratio of edge pixels to total pixels.
    """
    edges = feature.canny(image, sigma=sigma)
    return np.sum(edges) / edges.size


def local_std(image, window=5):
    """
    Local standard deviation in a sliding window.

    Args:
        image (ndarray): Preprocessed 2D float image.
        window (int): Size of the sliding window.

    Returns:
        float: Mean of local standard deviations.
    """
    shape = (image.shape[0] - window + 1, image.shape[1] - window + 1, window, window)
    strides = image.strides + image.strides
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    local_stds = windows.std(axis=(-1, -2))
    return np.mean(local_stds)


def mad(image):
    """
    Median Absolute Deviation.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Median of absolute deviations from the median.
    """
    median = np.median(image)
    return np.median(np.abs(image - median))


def fft_high_freq_ratio(image):
    """
    Ratio of high-frequency content in FFT.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Ratio of high-frequency magnitude to total magnitude.
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    center = np.array(magnitude.shape) // 2
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    radius = min(center) // 2
    mask = (x - center[1])**2 + (y - center[0])**2 > radius**2
    return np.sum(magnitude[mask]) / np.sum(magnitude)


def wavelet_energy(image, wavelet='haar'):
    """
    Wavelet decomposition energy.

    Args:
        image (ndarray): Preprocessed 2D float image.
        wavelet (str): Wavelet type for decomposition.

    Returns:
        float: Sum of absolute wavelet coefficients.
    """
    coeffs = pywt.dwt2(image, wavelet)
    _, (cH, cV, cD) = coeffs
    return np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))


def glcm_contrast(image):
    """
    Gray-Level Co-occurrence Matrix (GLCM) contrast.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: GLCM contrast value.
    """
    image_u8 = (image * 255).astype(np.uint8)
    glcm = graycomatrix(image_u8, [1], [0], symmetric=True, normed=True)
    return graycoprops(glcm, 'contrast')[0, 0]


def entropy_feature(image):
    """
    Shannon entropy of image.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Shannon entropy value.
    """
    return shannon_entropy(image)


def laplacian_variance(image):
    """
    Variance of Laplacian operator.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Variance of Laplacian-filtered image.
    """
    return np.var(laplace(image))


def directional_gradients(image):
    """
    Compute directional gradients (horizontal and vertical).

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        tuple: (x_gradient_variance, y_gradient_variance)
    """
    dx = ndi_sobel(image, axis=1)
    dy = ndi_sobel(image, axis=0)
    return np.mean(dx ** 2), np.mean(dy ** 2)


def image_kurtosis(image):
    """
    Kurtosis of image pixel distribution.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Kurtosis value (fourth moment).
    """
    return kurtosis(image.ravel())


def quantile_diff(image):
    """
    Difference between 95th and 80th percentiles.

    Args:
        image (ndarray): Preprocessed 2D float image.

    Returns:
        float: Percentile difference.
    """
    return np.percentile(image, 95) - np.percentile(image, 80)


def extract_features(image, feature_subset=None):
    """
    Compute a comprehensive set of sharpness and texture features.

    Args:
        image (ndarray): Preprocessed 2D float32 image.
        feature_subset (list, optional): List of feature names to compute.
            If None, all features from feature_names() are computed.

    Returns:
        dict: Dictionary of computed features with feature names as keys.

    Example:
        >>> from ifcb_focus.features import preprocess_image, extract_features
        >>> image = ...  # Load your image
        >>> processed = preprocess_image(image)
        >>> features = extract_features(processed)
        >>> print(features['laplacian_var'])

        >>> # Extract only specific features
        >>> slim_features = extract_features(processed, ['laplacian_var', 'dog_var'])
    """
    if feature_subset is None:
        feature_subset = feature_names()

    features = {}

    available_features = {
        'tenengrad': tenengrad_sharpness,
        'dog_var': dog_variance,
        'edge_density': edge_density,
        'local_std': local_std,
        'mad': mad,
        'sum_modified_laplacian': sum_modified_laplacian,
        'brenner_gradient': brenner_gradient,
        'fft_high_freq_ratio': fft_high_freq_ratio,
        'wavelet_energy': wavelet_energy,
        'glcm_contrast': glcm_contrast,
        'entropy': entropy_feature,
        'laplacian_var': laplacian_variance,
        'sobel_x_var': lambda img: directional_gradients(img)[0],
        'sobel_y_var': lambda img: directional_gradients(img)[1],
        'kurtosis': image_kurtosis,
        'quantile_diff': quantile_diff,
    }

    for feature_name in feature_subset:
        if feature_name in available_features:
            features[feature_name] = available_features[feature_name](image)

    return features


def feature_names():
    """
    Return the names of all available features in the order they are defined.

    Returns:
        list: List of feature names.
    """
    return FEATURE_NAMES.copy()


def rank_feature_speed():
    """
    Return features ranked by computation speed (fastest first).

    Note: This ranking should ideally be determined empirically based on
    actual benchmarks for your specific hardware and image sizes.

    Returns:
        list: List of feature names ordered from fastest to slowest.
    """
    return FEATURE_SPEED_RANKING.copy()
