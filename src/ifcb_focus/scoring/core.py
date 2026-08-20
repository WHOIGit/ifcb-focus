"""
Scoring functions for IFCB focus detection.

This module provides functions for scoring individual bins or batches of bins
using trained focus detection models. Bins are accessed through an ``ifcbkit``
data directory (``SyncIfcbDataDirectory``); callers pass the directory plus a
bin id, and ``ifcbkit`` locates the underlying raw files.
"""

import os
import pandas as pd

from ifcb_focus.features import extract_features, preprocess_image
from ifcb_focus.features.constants import STUDENT_MODEL_FEATURES


def score_bin(dd, pid, model, confidence_threshold=0.1, top_n=200,
              verbose=False, cache_dir=None, features_to_use=None):
    """
    Score a single IFCB bin for focus quality.

    Args:
        dd (ifcbkit.SyncIfcbDataDirectory): Data directory containing the bin.
        pid (str): Bin ID to score (e.g. 'D20230101T120000_IFCB123').
        model: Trained model (sklearn regressor) for prediction.
        confidence_threshold (float): Threshold for good/bad predictions.
            Default is 0.1.
        top_n (int): Number of ROIs to analyze. Default is 200.
        verbose (bool): Whether to print scoring details. Default is False.
        cache_dir (str, optional): Directory to cache extracted features.
        features_to_use (list, optional): Feature names to extract. If None,
            uses STUDENT_MODEL_FEATURES.

    Returns:
        float: Bin score between 0 and 1, where higher values indicate better
        focus. Score is ``good / (good + bad)`` where good = predictions above
        ``(1 - confidence_threshold)`` and bad = predictions below
        ``confidence_threshold``.

    Example:
        >>> from ifcbkit import SyncIfcbDataDirectory
        >>> import joblib
        >>> from ifcb_focus.scoring import score_bin
        >>> model = joblib.load('model.pkl')
        >>> dd = SyncIfcbDataDirectory('/path/to/data')
        >>> score = score_bin(dd, 'bin_id', model)
    """
    X = extract_slim_features(
        dd, pid,
        top_n=top_n,
        cache_dir=cache_dir,
        features_to_use=features_to_use,
    ).drop(columns=['pid']).to_numpy()

    if X.shape[0] == 0:
        return 0.0

    y = model.predict(X)
    bad = len(y[y < confidence_threshold])
    good = len(y[y > (1 - confidence_threshold)])
    score = good / (good + bad) if (good + bad) > 0 else 0.0

    if verbose:
        print(f"Scored bin {pid}: {score:.4f} (good: {good}, bad: {bad})")
    return score


def extract_slim_features(dd, pid, top_n=200, cache_dir=None,
                          features_to_use=None):
    """
    Extract a slim feature set from an IFCB bin.

    Args:
        dd (ifcbkit.SyncIfcbDataDirectory): Data directory containing the bin.
        pid (str): Bin ID.
        top_n (int): Number of ROIs to process. Default is 200.
        cache_dir (str, optional): Directory to cache/load features. If
            specified, features are saved to and loaded from this directory.
        features_to_use (list, optional): Feature names to extract. If None,
            uses STUDENT_MODEL_FEATURES.

    Returns:
        pandas.DataFrame: DataFrame with columns ['pid'] + feature names, one
        row per ROI image.

    Example:
        >>> from ifcbkit import SyncIfcbDataDirectory
        >>> from ifcb_focus.scoring import extract_slim_features
        >>> dd = SyncIfcbDataDirectory('/path/to/data')
        >>> features_df = extract_slim_features(dd, 'bin_id', top_n=100)
    """
    if features_to_use is None:
        features_to_use = STUDENT_MODEL_FEATURES

    if cache_dir is not None:
        features_path = os.path.join(cache_dir, f"{pid}_features.csv")
        if os.path.exists(features_path):
            return pd.read_csv(features_path)

    images = dd.read_images(pid)
    roi_numbers = list(images)

    features = []
    for roi in roi_numbers[:top_n]:
        roi_pid = f"{pid}_{roi:05d}"
        image = preprocess_image(images[roi])
        image_features = {'pid': roi_pid}
        image_features.update(extract_features(image, features_to_use))
        features.append(image_features)

    df = pd.DataFrame.from_records(features)
    if df.empty:
        df = pd.DataFrame(columns=['pid'] + list(features_to_use))

    if cache_dir is not None:
        df.to_csv(features_path, index=False, float_format='%.5f')

    return df


def score_directory(dd, model, features=None, verbose=True, cache_dir=None):
    """
    Score all bins in a data directory.

    Args:
        dd (ifcbkit.SyncIfcbDataDirectory): Data directory to process.
        model: Trained model for prediction.
        features (list, optional): Feature names to use. If None, uses
            STUDENT_MODEL_FEATURES.
        verbose (bool): Whether to print progress. Default is True.
        cache_dir (str, optional): Directory to cache features.

    Returns:
        pandas.DataFrame: DataFrame with columns ['pid', 'score', 'label']
        where label is 1 if score > 0.5, else 0.

    Example:
        >>> from ifcbkit import SyncIfcbDataDirectory
        >>> import joblib
        >>> from ifcb_focus.scoring import score_directory
        >>> model = joblib.load('model.pkl')
        >>> dd = SyncIfcbDataDirectory('/path/to/data')
        >>> results = score_directory(dd, model)
        >>> results.to_csv('scores.csv', index=False)
    """
    results = []

    for fs in dd.list():
        pid = fs['pid']
        score = score_bin(
            dd, pid, model,
            verbose=verbose,
            cache_dir=cache_dir,
            features_to_use=features,
        )
        results.append({
            'pid': pid,
            'score': score,
            'label': int(score > 0.5),
        })

    return pd.DataFrame.from_records(results)
