"""
IFCB Focus Detection

A machine learning framework for detecting blurry images from IFCB instruments.

This package provides tools for:
- Feature extraction from IFCB images
- Training teacher and student models for focus detection
- Scoring individual bins or batches of bins
- Data preparation workflows

Example:
    >>> import ifcb_focus
    >>> from ifcb import DataDirectory
    >>> import joblib
    >>>
    >>> # Extract features from an image
    >>> image = ...  # Load your image
    >>> processed = ifcb_focus.preprocess_image(image)
    >>> features = ifcb_focus.extract_features(processed)
    >>>
    >>> # Score a bin
    >>> model = joblib.load('model.pkl')
    >>> dd = DataDirectory('/path/to/data')
    >>> bin_data = dd['bin_id']
    >>> score = ifcb_focus.score_bin(bin_data, model)
"""

__version__ = "0.1.0"

# Import public API from submodules
from .features import (
    extract_features,
    preprocess_image,
    feature_names,
    STUDENT_MODEL_FEATURES,
)
from .scoring import (
    score_bin,
    extract_slim_features,
    score_remote_bin,
    score_directory,
)
from .workflows import (
    feature_extraction,
    split_true_labels,
)
from .training import (
    train_base_model,
    train_random_forest,
    train_student_model,
)

__all__ = [
    # Version
    '__version__',
    # Features
    'extract_features',
    'preprocess_image',
    'feature_names',
    'STUDENT_MODEL_FEATURES',
    # Scoring
    'score_bin',
    'extract_slim_features',
    'score_remote_bin',
    'score_directory',
    # Workflows
    'feature_extraction',
    'split_true_labels',
    # Training
    'train_base_model',
    'train_random_forest',
    'train_student_model',
]
