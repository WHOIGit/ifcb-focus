"""
Teacher model scoring functions for IFCB focus detection.

This module contains scoring functions for use with teacher models (classifiers),
which use predict_proba() and confidence thresholding. This differs from the
student model scoring approach (in core.py) which uses regressor outputs.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from ifcb_focus.features import preprocess_image
from ifcb_focus.training.utils import compute_features, skip_image


def score_bin_with_teacher(b, model, top_n=100, progress_bar=False):
    """
    Score a bin using a teacher model (classifier).

    This method:
    1. Uses predict_proba() from a classifier model (e.g., teacher model)
    2. Filters predictions by confidence threshold (0.66)
    3. Returns both score and average confidence

    Args:
        b: IFCB bin object (from pyifcb library).
        model: Trained classifier (teacher model) with predict_proba() method.
        top_n (int): Number of largest ROIs to analyze. Default is 100.
        progress_bar (bool): Whether to show progress bar. Default is False.

    Returns:
        tuple: (score, sample_confidence)
            - score: Mean probability of "good" class for high-confidence predictions
            - sample_confidence: Mean confidence of high-confidence predictions

    Example:
        >>> from ifcb import DataDirectory
        >>> import joblib
        >>> from ifcb_focus.scoring import score_bin_with_teacher
        >>>
        >>> teacher_model = joblib.load('teacher_model.pkl')
        >>> dd = DataDirectory('/path/to/data')
        >>> bin_data = dd['bin_id']
        >>> score, confidence = score_bin_with_teacher(bin_data, teacher_model)
        >>> print(f'Score: {score:.4f}, Confidence: {confidence:.4f}')

    Note:
        For student models (regressors), use score_bin() instead.
    """
    feature_names = None
    features_list = []
    s = b.schema
    width_col = s.ROI_WIDTH
    height_col = s.ROI_HEIGHT
    roi_metadata = b.adc

    # Order by area (width * height) descending
    roi_metadata['roi_number'] = roi_metadata.index
    roi_metadata = roi_metadata[[width_col, height_col, 'roi_number']].copy()
    roi_metadata.loc[:, 'area'] = roi_metadata[width_col] * roi_metadata[height_col]
    roi_metadata = roi_metadata.sort_values(by='area', ascending=False).head(top_n)

    for roi_number in tqdm(roi_metadata['roi_number'], disable=not progress_bar):
        if skip_image(b.images[roi_number]):
            continue
        img = preprocess_image(b.images[roi_number])
        features = compute_features(img)
        if feature_names is None:
            feature_names = list(features.keys())
        features = list(features.values())
        features = np.array(features)
        features_list.append(features)

    X = pd.DataFrame(np.array(features_list), columns=feature_names)
    y_probabilities = model.predict_proba(X)
    y = y_probabilities[:, 1]  # Probability of being a 'good' ROI
    confidence = np.max(y_probabilities, axis=1)  # Confidence of the prediction
    confidence_threshold = 0.66
    highconf = np.where(confidence > confidence_threshold)[0]

    score = np.mean(y[highconf]) if len(highconf) > 0 else 0.0
    sample_confidence = np.mean(confidence[highconf]) if len(highconf) > 0 else 0.0

    return score, sample_confidence


def score_remote_bin_with_teacher(host, pid, model, top_n=100, progress_bar=False):
    """
    Score a remote bin using a teacher model (classifier).

    Args:
        host (str): Hostname of the IFCB data server.
        pid (str): Bin ID.
        model: Trained classifier (teacher model) with predict_proba() method.
        top_n (int): Number of largest ROIs to analyze. Default is 100.
        progress_bar (bool): Whether to show progress bar. Default is False.

    Returns:
        tuple: (score, sample_confidence)

    Example:
        >>> import joblib
        >>> from ifcb_focus.scoring import score_remote_bin_with_teacher
        >>> teacher_model = joblib.load('teacher_model.pkl')
        >>> score, conf = score_remote_bin_with_teacher('ifcb-data.whoi.edu', 'D20230101T120000_IFCB123', teacher_model)
    """
    from ifcb import open_url

    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin_with_teacher(b, model, top_n=top_n, progress_bar=progress_bar)
