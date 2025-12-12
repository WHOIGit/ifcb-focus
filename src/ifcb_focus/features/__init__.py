"""
Feature extraction module for IFCB focus detection.

This module provides image preprocessing and feature extraction capabilities
for detecting image focus quality in IFCB data.
"""

from .preprocessing import preprocess_image
from .extraction import (
    extract_features,
    feature_names,
    rank_feature_speed,
    # Individual feature functions
    sum_modified_laplacian,
    tenengrad_sharpness,
    brenner_gradient,
    dog_variance,
    edge_density,
    local_std,
    mad,
    fft_high_freq_ratio,
    wavelet_energy,
    glcm_contrast,
    entropy_feature,
    laplacian_variance,
    directional_gradients,
    image_kurtosis,
    quantile_diff,
)
from .constants import (
    STUDENT_MODEL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SPEED_RANKING,
)

__all__ = [
    # Preprocessing
    'preprocess_image',
    # Main feature extraction
    'extract_features',
    'feature_names',
    'rank_feature_speed',
    # Constants
    'STUDENT_MODEL_FEATURES',
    'FEATURE_NAMES',
    'FEATURE_SPEED_RANKING',
    # Individual feature functions
    'sum_modified_laplacian',
    'tenengrad_sharpness',
    'brenner_gradient',
    'dog_variance',
    'edge_density',
    'local_std',
    'mad',
    'fft_high_freq_ratio',
    'wavelet_energy',
    'glcm_contrast',
    'entropy_feature',
    'laplacian_variance',
    'directional_gradients',
    'image_kurtosis',
    'quantile_diff',
]
