"""
Scoring module for IFCB focus detection.

This module provides functions for scoring IFCB bins using trained models.
Includes functions for both student models (regressors) and teacher models (classifiers).
"""

from .core import (
    score_bin,
    extract_slim_features,
    score_remote_bin,
    score_directory,
)
from .teacher import (
    score_bin_with_teacher,
    score_remote_bin_with_teacher,
)

__all__ = [
    # Student model scoring functions (main)
    'score_bin',
    'extract_slim_features',
    'score_remote_bin',
    'score_directory',
    # Teacher model scoring functions
    'score_bin_with_teacher',
    'score_remote_bin_with_teacher',
]
