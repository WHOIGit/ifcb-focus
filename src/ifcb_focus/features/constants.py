"""
Constants for feature extraction in IFCB focus detection.
"""

# Slim feature set for fast inference (used by student model)
STUDENT_MODEL_FEATURES = ['laplacian_var', 'dog_var', 'kurtosis', 'mad']

# All available feature names in order
FEATURE_NAMES = [
    'tenengrad', 'dog_var', 'edge_density', 'local_std', 'mad',
    'sum_modified_laplacian', 'brenner_gradient', 'fft_high_freq_ratio',
    'wavelet_energy', 'glcm_contrast', 'entropy', 'laplacian_var',
    'sobel_x_var', 'sobel_y_var', 'kurtosis', 'quantile_diff'
]

# Feature speed ranking (fastest to slowest)
# Note: This ranking should be determined empirically
FEATURE_SPEED_RANKING = [
    'mad', 'quantile_diff', 'kurtosis', 'tenengrad', 'laplacian_var',
    'brenner_gradient', 'dog_var', 'sum_modified_laplacian', 'local_std',
    'sobel_x_var', 'sobel_y_var', 'entropy', 'edge_density', 'glcm_contrast',
    'wavelet_energy', 'fft_high_freq_ratio'
]
