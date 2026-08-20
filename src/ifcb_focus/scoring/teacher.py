"""
Teacher model scoring functions for IFCB focus detection.

Uses ``predict_proba()`` from classifier (teacher) models with confidence
thresholding. This differs from student-model scoring (in ``core.py``), which
uses regressor outputs. Bins are accessed through an ``ifcbkit`` data
directory.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from ifcb_focus.features import preprocess_image
from ifcb_focus.training.utils import compute_features, skip_image


def score_bin_with_teacher(dd, pid, model, top_n=100, progress_bar=False):
    """
    Score a bin using a teacher model (classifier).

    This method:
    1. Uses predict_proba() from a classifier model (e.g., teacher model)
    2. Filters predictions by confidence threshold (0.66)
    3. Returns both score and average confidence

    Only the largest ROIs (by ADC area) are analyzed.

    Args:
        dd (ifcbkit.SyncIfcbDataDirectory): Data directory containing the bin.
        pid (str): Bin ID to score (e.g. 'D20230101T120000_IFCB123').
        model: Trained classifier (teacher model) with predict_proba().
        top_n (int): Number of largest ROIs to analyze. Default is 100.
        progress_bar (bool): Whether to show a progress bar. Default is False.

    Returns:
        tuple: (score, sample_confidence)
            - score: mean probability of the "good" class over high-confidence
              predictions
            - sample_confidence: mean confidence of those predictions

    Example:
        >>> from ifcbkit import SyncIfcbDataDirectory
        >>> import joblib
        >>> from ifcb_focus.scoring import score_bin_with_teacher
        >>> teacher_model = joblib.load('teacher_model.pkl')
        >>> dd = SyncIfcbDataDirectory('/path/to/data')
        >>> score, confidence = score_bin_with_teacher(dd, 'bin_id', teacher_model)
    """
    feature_names = None
    features_list = []

    # Per-target ROI metadata from the ADC: {target: {roi_id, x, y, width, height}}
    roi_meta = dd.list_images(pid)
    roi_metadata = pd.DataFrame.from_dict(roi_meta, orient='index')
    roi_metadata.index.name = 'roi_number'
    roi_metadata['roi_number'] = roi_metadata.index
    roi_metadata['area'] = roi_metadata['width'] * roi_metadata['height']
    roi_metadata = roi_metadata.sort_values(by='area', ascending=False).head(top_n)

    images = dd.read_images(pid)

    for roi_number in tqdm(roi_metadata['roi_number'], disable=not progress_bar):
        roi_number = int(roi_number)
        if roi_number not in images:
            continue
        raw = np.asarray(images[roi_number])
        if skip_image(raw):
            continue
        img = preprocess_image(raw)
        features = compute_features(img)
        if feature_names is None:
            feature_names = list(features.keys())
        features_list.append(np.array(list(features.values())))

    if not features_list:
        return 0.0, 0.0

    X = pd.DataFrame(np.array(features_list), columns=feature_names)
    y_probabilities = model.predict_proba(X)
    y = y_probabilities[:, 1]  # probability of being a 'good' ROI
    confidence = np.max(y_probabilities, axis=1)
    confidence_threshold = 0.66
    highconf = np.where(confidence > confidence_threshold)[0]

    score = np.mean(y[highconf]) if len(highconf) > 0 else 0.0
    sample_confidence = np.mean(confidence[highconf]) if len(highconf) > 0 else 0.0

    return score, sample_confidence
