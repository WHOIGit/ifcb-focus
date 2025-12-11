"""
Training module for IFCB focus detection.

This module provides functions for training teacher and student models
for IFCB focus quality detection.
"""

from .base import train_base_model
from .teacher import (
    train_random_forest,
    validate_model,
    features_and_labels,
    assemble_features,
)
from .student import (
    train_student_model,
    search_for_features,
    feature_ranking,
)
from .utils import (
    list_images,
    load_images,
    skip_image,
    compute_features,
)

__all__ = [
    # Base training
    'train_base_model',
    # Teacher model
    'train_random_forest',
    'validate_model',
    'features_and_labels',
    'assemble_features',
    # Student model
    'train_student_model',
    'search_for_features',
    'feature_ranking',
    # Utilities
    'list_images',
    'load_images',
    'skip_image',
    'compute_features',
]
