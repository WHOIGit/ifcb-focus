"""
Workflow module for IFCB focus detection.

This module provides high-level workflow functions for data preparation.
"""

from .core import feature_extraction, split_true_labels

__all__ = ['feature_extraction', 'split_true_labels']
