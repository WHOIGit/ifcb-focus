"""
Scoring functions for IFCB focus detection.

This module provides functions for scoring individual bins or batches of bins
using trained focus detection models.
"""

import os
import pandas as pd
from ifcb import DataDirectory, open_url

from ifcb_focus.features import extract_features, preprocess_image
from ifcb_focus.features.constants import STUDENT_MODEL_FEATURES


def score_bin(b, model, confidence_threshold=0.1, top_n=200, verbose=False, cache_dir=None):
    """
    Score a single IFCB bin for focus quality.

    Args:
        b: IFCB bin object (from pyifcb library).
        model: Trained model (sklearn RandomForestRegressor) for prediction.
        confidence_threshold (float): Threshold for determining good/bad predictions.
            Default is 0.1.
        top_n (int): Number of largest ROIs to analyze. Default is 200.
        verbose (bool): Whether to print scoring details. Default is False.
        cache_dir (str, optional): Directory to cache extracted features.

    Returns:
        float: Bin score between 0 and 1, where higher values indicate better focus.
            Score is calculated as: good / (good + bad), where:
            - good = count of predictions > (1 - confidence_threshold)
            - bad = count of predictions < confidence_threshold

    Example:
        >>> from ifcb import DataDirectory
        >>> import joblib
        >>> from ifcb_focus.scoring import score_bin
        >>>
        >>> model = joblib.load('model.pkl')
        >>> dd = DataDirectory('/path/to/data')
        >>> bin_data = dd['bin_id']
        >>> score = score_bin(bin_data, model)
        >>> print(f'Bin score: {score:.4f}')
    """
    X = extract_slim_features(b, top_n=top_n, cache_dir=cache_dir).drop(columns=['pid']).values
    if X.shape[0] == 0:
        return 0.0
    y = model.predict(X)
    # compute confidence as the mean of the predicted probabilities
    bad = len(y[y < confidence_threshold])
    good = len(y[y > (1 - confidence_threshold)])
    score = good / (good + bad) if (good + bad) > 0 else 0.0
    if verbose:
        print(f"Scored bin {b.lid}: {score:.4f} (good: {good}, bad: {bad})")
    return score


def extract_slim_features(b, top_n=200, cache_dir=None, features_to_use=None):
    """
    Extract slim feature set from an IFCB bin.

    Args:
        b: IFCB bin object (from pyifcb library).
        top_n (int): Number of largest ROIs to process. Default is 200.
        cache_dir (str, optional): Directory to cache/load features. If specified,
            features will be saved to and loaded from this directory.
        features_to_use (list, optional): List of feature names to extract.
            If None, uses STUDENT_MODEL_FEATURES.

    Returns:
        pandas.DataFrame: DataFrame with columns ['pid'] + feature names.

    Example:
        >>> from ifcb import DataDirectory
        >>> from ifcb_focus.scoring import extract_slim_features
        >>>
        >>> dd = DataDirectory('/path/to/data')
        >>> bin_data = dd['bin_id']
        >>> features_df = extract_slim_features(bin_data, top_n=100)
        >>> print(features_df.head())
    """
    if features_to_use is None:
        features_to_use = STUDENT_MODEL_FEATURES

    if cache_dir is not None:
        features_path = os.path.join(cache_dir, f"{b.lid}_features.csv")
        if os.path.exists(features_path):
            return pd.read_csv(features_path)

    roi_numbers = list(b.images.keys())
    features = []
    for roi in roi_numbers[:top_n]:
        roi_pid = f"{b.lid}_{roi:05d}"
        image = b.images[roi]
        image = preprocess_image(image)
        image_features = {'pid': roi_pid}
        image_features.update(extract_features(image, features_to_use))
        features.append(image_features)

    df = pd.DataFrame.from_records(features)
    if df.empty:
        df = pd.DataFrame(columns=['pid'] + list(features_to_use))

    if cache_dir is not None:
        df.to_csv(features_path, index=False, float_format='%.5f')

    return df


def score_remote_bin(host, pid, model, features=None):
    """
    Score a remote IFCB bin via HTTP.

    Args:
        host (str): Hostname of the IFCB data server.
        pid (str): Bin ID (e.g., 'D20230101T120000_IFCB123').
        model: Trained model for prediction.
        features (list, optional): Feature names to use. If None, uses STUDENT_MODEL_FEATURES.

    Returns:
        float: Bin score between 0 and 1.

    Example:
        >>> import joblib
        >>> from ifcb_focus.scoring import score_remote_bin
        >>>
        >>> model = joblib.load('model.pkl')
        >>> score = score_remote_bin('ifcb-data.whoi.edu', 'D20230101T120000_IFCB123', model)
        >>> print(f'Remote bin score: {score:.4f}')
    """
    if features is None:
        features = STUDENT_MODEL_FEATURES

    with open_url(f'https://{host}/data/{pid}.adc') as b:
        return score_bin(b, model)


def score_directory(data_dir, model, features=None, verbose=True, cache_dir=None):
    """
    Score all bins in a directory.

    Args:
        data_dir (str): Path to directory containing IFCB bins.
        model: Trained model for prediction.
        features (list, optional): Feature names to use. If None, uses STUDENT_MODEL_FEATURES.
        verbose (bool): Whether to print progress. Default is True.
        cache_dir (str, optional): Directory to cache features.

    Returns:
        pandas.DataFrame: DataFrame with columns ['pid', 'score', 'label'].
            - pid: Bin ID
            - score: Continuous score (0-1)
            - label: Binary label (1 if score > 0.5, else 0)

    Example:
        >>> import joblib
        >>> from ifcb_focus.scoring import score_directory
        >>>
        >>> model = joblib.load('model.pkl')
        >>> results = score_directory('/path/to/data', model, verbose=True)
        >>> print(results)
        >>> results.to_csv('scores.csv', index=False)
    """
    if features is None:
        features = STUDENT_MODEL_FEATURES

    dd = DataDirectory(data_dir)
    results = []

    def process_bin(b):
        score = score_bin(b, model, verbose=verbose, cache_dir=cache_dir)
        return {
            'pid': b.lid,
            'score': score,
            'label': int(score > 0.5)
        }

    for b in dd:
        results.append(process_bin(b))

    return pd.DataFrame.from_records(results)
