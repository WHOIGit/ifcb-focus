from skimage.filters import sobel
from skimage import feature, exposure
from skimage.util import img_as_float
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from scipy.ndimage import convolve, gaussian_filter, sobel as ndi_sobel, laplace
from scipy.stats import kurtosis
import numpy as np
import pywt

def preprocess_image(image):
    # use img_as_float to convert to float32
    image = img_as_float(image)
    # enhance contrast
    image = exposure.equalize_adapthist(image, clip_limit=0.03)
    return image

def sum_modified_laplacian(image):
    kernel_h = np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]])
    kernel_v = np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]])
    lap_h = convolve(image, kernel_h, mode='reflect')
    lap_v = convolve(image, kernel_v, mode='reflect')
    return np.sum(np.abs(lap_h) + np.abs(lap_v))

def tenengrad_sharpness(image):
    sobel_grad = sobel(image)
    return np.mean(sobel_grad ** 2)

def brenner_gradient(image):
    return np.sum((image[:, :-2] - image[:, 2:]) ** 2)

def dog_variance(image, sigma1=1, sigma2=2):
    blur1 = gaussian_filter(image, sigma1)
    blur2 = gaussian_filter(image, sigma2)
    dog = blur1 - blur2
    return np.var(dog)

def edge_density(image, sigma=1.0):
    edges = feature.canny(image, sigma=sigma)
    return np.sum(edges) / edges.size

def local_std(image, window=5):
    shape = (image.shape[0] - window + 1, image.shape[1] - window + 1, window, window)
    strides = image.strides + image.strides
    windows = np.lib.stride_tricks.as_strided(image, shape=shape, strides=strides)
    local_stds = windows.std(axis=(-1, -2))
    return np.mean(local_stds)

def mad(image):
    median = np.median(image)
    return np.median(np.abs(image - median))

def fft_high_freq_ratio(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    center = np.array(magnitude.shape) // 2
    y, x = np.ogrid[:image.shape[0], :image.shape[1]]
    radius = min(center) // 2
    mask = (x - center[1])**2 + (y - center[0])**2 > radius**2
    return np.sum(magnitude[mask]) / np.sum(magnitude)

def wavelet_energy(image, wavelet='haar'):
    coeffs = pywt.dwt2(image, wavelet)
    _, (cH, cV, cD) = coeffs
    return np.sum(np.abs(cH)) + np.sum(np.abs(cV)) + np.sum(np.abs(cD))

def glcm_contrast(image):
    image_u8 = (image * 255).astype(np.uint8)
    glcm = graycomatrix(image_u8, [1], [0], symmetric=True, normed=True)
    return graycoprops(glcm, 'contrast')[0, 0]

def entropy_feature(image):
    return shannon_entropy(image)

def laplacian_variance(image):
    return np.var(laplace(image))

def directional_gradients(image):
    dx = ndi_sobel(image, axis=1)
    dy = ndi_sobel(image, axis=0)
    return np.mean(dx ** 2), np.mean(dy ** 2)

def image_kurtosis(image):
    return kurtosis(image.ravel())

def quantile_diff(image):
    return np.percentile(image, 95) - np.percentile(image, 80)

def extract_features(image, feature_subset=None):
    """
    Compute a comprehensive set of sharpness and texture features.

    Parameters:
        image (ndarray): Preprocessed 2D float32 image.

    Returns:
        dict: Dictionary of computed features.
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
    Return the names of the features in the order they are computed.
    """
    return [
        'tenengrad', 'dog_var', 'edge_density', 'local_std', 'mad',
        'sum_modified_laplacian', 'brenner_gradient', 'fft_high_freq_ratio',
        'wavelet_energy', 'glcm_contrast', 'entropy', 'laplacian_var',
        'sobel_x_var', 'sobel_y_var', 'kurtosis', 'quantile_diff'
    ]

def rank_feature_speed():
    # FIXME: determine this empirically
    # return features in order of speed, fastest first
    return [
        'mad', 'quantile_diff', 'kurtosis', 'tenengrad', 'laplacian_var',
        'brenner_gradient', 'dog_var', 'sum_modified_laplacian', 'local_std',
        'sobel_x_var', 'sobel_y_var', 'entropy', 'edge_density', 'glcm_contrast',
        'wavelet_energy', 'fft_high_freq_ratio'
    ]
