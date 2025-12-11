"""
Utility module for IFCB focus detection.

This module provides utility functions for data augmentation, downloading,
and analysis.
"""

from .augmentation import blur_image, augment_dataset
from .data import download_bin, download_validation_dataset
from .analysis import (
    KnockoutVotingEnsemble,
    get_strongest_correlations,
    create_knockout_ensemble,
    interpret_bias,
    analyze_pseudo_labels,
    evaluate_model_on_pseudo_labels,
)

__all__ = [
    # Augmentation
    'blur_image',
    'augment_dataset',
    # Data downloading
    'download_bin',
    'download_validation_dataset',
    # Analysis
    'KnockoutVotingEnsemble',
    'get_strongest_correlations',
    'create_knockout_ensemble',
    'interpret_bias',
    'analyze_pseudo_labels',
    'evaluate_model_on_pseudo_labels',
]
